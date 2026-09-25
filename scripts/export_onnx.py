#!/usr/bin/env python3
"""Export a CDM checkpoint to ONNX for TensorRT, plus a parity reference.

The ONNX graph is the network only: [1, 4, H, W] float32 (ImageNet-normalized
RGB + inverse depth) -> [1, H, W] float32 inverse depth, at one fixed network
resolution. Pre- and post-processing (resize, normalization, inversion) stay
outside the graph and are reimplemented in C++, so the reference written here
records the tensors on both sides of the network as well as the final depth,
which lets the C++ side check each stage separately.

The network resolution follows the package's own resize rule
(util.transform.Resize, lower_bound, multiple of 14) for the camera resolution
given, so an engine built from this ONNX serves exactly one camera resolution.

The reference is produced by the unmodified `RGBDDepth.infer_depth` on the
unpatched model, in full FP32 (TF32 off). The exported model differs only in
that the DINOv2 positional-embedding interpolation is precomputed for the fixed
resolution; the script checks that this changes nothing before exporting.

Example:
    python scripts/export_onnx.py \
        --checkpoint models/cdm_d435.ckpt --image-width 640 --image-height 480 \
        --output models/cdm_d435_640x480_fp32.onnx \
        --reference-rgb Input/moving_robots/rgb/<stamp>.png \
        --reference-depth Input/moving_robots/depth/<stamp>.png --depth-factor 1000 \
        --reference-dir models/cdm_d435_640x480_reference
"""

import argparse
import hashlib
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from camera_depth_models import model_configs  # noqa: E402
from camera_depth_models.dinov2_layers import attention, block  # noqa: E402
from camera_depth_models.dpt import RGBDDepth  # noqa: E402
from camera_depth_models.util.transform import Resize  # noqa: E402

PATCH_SIZE = 14


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--encoder", default="vitl", choices=sorted(model_configs))
    p.add_argument("--output", required=True, help="ONNX path; weights go to <output>.data")
    p.add_argument("--image-width", type=int, required=True, help="camera image width the engine will serve")
    p.add_argument("--image-height", type=int, required=True, help="camera image height the engine will serve")
    p.add_argument("--input-size", type=int, default=518, help="infer_depth's input_size")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32",
                   help="weights/activations type of the graph; I/O stays float32. "
                        "TensorRT >= 10 builds strongly typed, so this sets the engine precision")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference-rgb", required=True, help="colour image (any format cv2 reads)")
    p.add_argument("--reference-depth", required=True, help="uint16 depth image")
    p.add_argument("--depth-factor", type=float, required=True, help="raw depth units per metre")
    p.add_argument("--reference-dir", required=True)
    p.add_argument("--resize-reference", action="store_true",
                   help="resize a reference frame of another resolution to the camera resolution "
                        "(colour INTER_AREA, depth INTER_NEAREST). The reference only has to be a "
                        "realistic input for the parity check, not a frame of the target camera")
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def network_size(image_w, image_h, input_size):
    """The (width, height) infer_depth resizes to, from the package's own rule."""
    resize = Resize(
        width=input_size,
        height=input_size,
        resize_target=True,
        keep_aspect_ratio=True,
        ensure_multiple_of=PATCH_SIZE,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_CUBIC,
    )
    w, h = resize.get_size(image_w, image_h)
    return int(w), int(h)


def load_model(encoder, checkpoint_path):
    """Same key handling as camera_depth_models.load_model, but a missing key is an error.

    Upstream loads with strict=False, so a checkpoint for another architecture
    loads silently with random weights. Unexpected keys (e.g. the unused RGB-only
    head) are reported, not fatal: the architecture never reads them.
    """
    model = RGBDDepth(**model_configs[encoder])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in checkpoint:
        states = {k[7:]: v for k, v in checkpoint["model"].items()}
    elif "state_dict" in checkpoint:
        states = {k[9:]: v for k, v in checkpoint["state_dict"].items()}
    else:
        states = checkpoint
    missing, unexpected = model.load_state_dict(states, strict=False)
    if missing:
        raise RuntimeError(f"checkpoint lacks {len(missing)} model keys, e.g. {missing[:5]}")
    if unexpected:
        prefixes = sorted({k.split(".")[0] for k in unexpected})
        print(f"[cdm] ignoring {len(unexpected)} checkpoint keys the model does not use (prefixes: {prefixes})")
    return model.eval()


def freeze_pos_embed(encoder, net_w, net_h):
    """Replace the bicubic pos-embedding interpolation by its value at (net_w, net_h).

    The interpolation runs on a parameter, so at a fixed resolution it is a
    constant; exporting it as a graph Resize would make TensorRT's cubic kernel
    part of the model. prepare_tokens_with_masks calls it as (x, w, h) with
    `B, nc, w, h = x.shape`, i.e. w is the height.
    """
    tokens = torch.zeros(1, (net_h // PATCH_SIZE) * (net_w // PATCH_SIZE) + 1, encoder.embed_dim,
                         device=encoder.pos_embed.device, dtype=encoder.pos_embed.dtype)
    with torch.no_grad():
        frozen = encoder.interpolate_pos_encoding(tokens, net_h, net_w).clone()
    encoder.register_buffer("frozen_pos_embed", frozen, persistent=False)

    def interpolate_pos_encoding(x, w, h):
        if (w, h) != (net_h, net_w):
            raise RuntimeError(f"exported for {net_w}x{net_h}, got {h}x{w}")
        return encoder.frozen_pos_embed.to(x.dtype)

    encoder.interpolate_pos_encoding = interpolate_pos_encoding


class ExportWrapper(nn.Module):
    """float32 in, float32 out, whatever the internal precision."""

    def __init__(self, model, dtype):
        super().__init__()
        self.model = model
        self.dtype = dtype

    def forward(self, rgbd):
        return self.model(rgbd.to(self.dtype)).float()


def write_f32(path, array):
    np.ascontiguousarray(array, dtype="<f4").tofile(path)


def main():
    args = parse_args()
    device = torch.device(args.device)

    # The reference is full FP32. cuDNN convolutions default to TF32 on Ampere+.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # xFormers kernels do not export; the plain attention path is numerically the same op.
    attention.XFORMERS_AVAILABLE = False
    block.XFORMERS_AVAILABLE = False

    net_w, net_h = network_size(args.image_width, args.image_height, args.input_size)
    print(f"[cdm] camera {args.image_width}x{args.image_height} -> network {net_w}x{net_h} "
          f"({net_w // PATCH_SIZE}x{net_h // PATCH_SIZE} patches)")

    model = load_model(args.encoder, args.checkpoint).to(device)

    # --- reference, from the unmodified inference path -------------------------
    bgr = cv2.imread(args.reference_rgb, cv2.IMREAD_COLOR)
    raw_depth = cv2.imread(args.reference_depth, cv2.IMREAD_UNCHANGED)
    if bgr is None or raw_depth is None:
        raise FileNotFoundError("cannot read the reference rgb/depth images")
    if raw_depth.dtype != np.uint16 or raw_depth.ndim != 2:
        raise ValueError(f"reference depth must be single-channel uint16, got {raw_depth.dtype} {raw_depth.shape}")
    if raw_depth.shape != bgr.shape[:2]:
        raise ValueError(f"reference colour {bgr.shape[1]}x{bgr.shape[0]} and depth "
                         f"{raw_depth.shape[1]}x{raw_depth.shape[0]} differ in size")
    if bgr.shape[:2] != (args.image_height, args.image_width) and args.resize_reference:
        print(f"[cdm] resizing the reference frame {bgr.shape[1]}x{bgr.shape[0]} -> "
              f"{args.image_width}x{args.image_height}")
        size = (args.image_width, args.image_height)
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
        raw_depth = cv2.resize(raw_depth, size, interpolation=cv2.INTER_NEAREST)
    if bgr.shape[:2] != (args.image_height, args.image_width):
        raise ValueError(f"reference images are {bgr.shape[1]}x{bgr.shape[0]} / "
                         f"{raw_depth.shape[1]}x{raw_depth.shape[0]}, the engine is for "
                         f"{args.image_width}x{args.image_height} (see --resize-reference)")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth_m = raw_depth.astype(np.float32) / args.depth_factor

    # infer_depth calls self.forward() directly, which module hooks do not see,
    # so the tensors are captured by shadowing forward on the instance.
    captured = {}
    original_forward = model.forward

    def capturing_forward(x):
        captured["input"] = x.detach().clone()
        out = original_forward(x)
        captured["output"] = out.detach().clone()
        return out

    model.forward = capturing_forward
    with np.errstate(divide="ignore"):
        ref_depth = model.infer_depth(rgb, depth_m, input_size=args.input_size)
    del model.forward
    ref_input = captured["input"]
    if tuple(ref_input.shape) != (1, 4, net_h, net_w):
        raise RuntimeError(f"infer_depth fed {tuple(ref_input.shape)}, expected (1, 4, {net_h}, {net_w})")

    # --- the exported model ----------------------------------------------------
    freeze_pos_embed(model.pretrained, net_w, net_h)
    freeze_pos_embed(model.depth_pretrained, net_w, net_h)
    with torch.no_grad():
        frozen_out = model(ref_input)
    diff = (frozen_out - captured["output"]).abs().max().item()
    print(f"[cdm] frozen pos-embed vs original: max |diff| = {diff:.3e}")
    if diff > 1e-5:
        raise RuntimeError("freezing the positional embedding changed the output")

    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    wrapper = ExportWrapper(model.to(dtype), dtype).eval()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = args.output + ".export"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_onnx = os.path.join(tmp_dir, "model.onnx")
    print(f"[cdm] exporting {args.precision} ONNX (opset {args.opset}) ...")
    # In eval mode nn.MultiheadAttention takes a fused fast path
    # (aten::_native_multi_head_attention) that has no ONNX export; the regular
    # path computes the same attention from exportable ops.
    torch.backends.mha.set_fastpath_enabled(False)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (ref_input.float(),),
            tmp_onnx,
            input_names=["rgbd"],
            output_names=["inverse_depth"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    # The weights exceed protobuf's 2 GB limit, so the exporter scatters them
    # into one file per tensor. Gather them into a single <output>.data.
    import onnx

    graph = onnx.load(tmp_onnx, load_external_data=True)
    data_name = os.path.basename(args.output) + ".data"
    if os.path.exists(os.path.join(out_dir, data_name)):
        os.remove(os.path.join(out_dir, data_name))
    onnx.save_model(graph, args.output, save_as_external_data=True,
                    all_tensors_to_one_file=True, location=data_name, size_threshold=1024)
    onnx.checker.check_model(args.output)
    for name in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, name))
    os.rmdir(tmp_dir)
    print(f"[cdm] ONNX ready: {args.output} (+ {data_name})")

    # --- write the reference ---------------------------------------------------
    os.makedirs(args.reference_dir, exist_ok=True)
    cv2.imwrite(os.path.join(args.reference_dir, "rgb.png"), bgr)
    write_f32(os.path.join(args.reference_dir, "depth_m.f32"), depth_m)
    write_f32(os.path.join(args.reference_dir, "input_rgbd.f32"), ref_input.cpu().numpy())
    write_f32(os.path.join(args.reference_dir, "inverse_depth.f32"), captured["output"].cpu().numpy())
    write_f32(os.path.join(args.reference_dir, "depth.f32"), ref_depth)
    meta = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "encoder": args.encoder,
        "precision": args.precision,
        "opset": args.opset,
        "input_size": args.input_size,
        "image_size": [args.image_width, args.image_height],
        "network_size": [net_w, net_h],
        "reference_rgb": os.path.abspath(args.reference_rgb),
        "reference_depth": os.path.abspath(args.reference_depth),
        "depth_factor": args.depth_factor,
        "reference_resized": bool(args.resize_reference),
        "torch": torch.__version__,
        "files": {
            "rgb.png": "BGR uint8, as read",
            "depth_m.f32": "HxW float32 metres fed to infer_depth",
            "input_rgbd.f32": "1x4xhxw float32 network input",
            "inverse_depth.f32": "1xhxw float32 network output",
            "depth.f32": "HxW float32 metres returned by infer_depth (inf where the network output 0)",
        },
    }
    with open(os.path.join(args.reference_dir, "reference.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[cdm] reference ready: {args.reference_dir}")


if __name__ == "__main__":
    main()
