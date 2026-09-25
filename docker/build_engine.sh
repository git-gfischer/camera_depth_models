#!/usr/bin/env bash
# Build and verify a CDM TensorRT engine inside the cdm-trt container:
#
#   checkpoint --[export_onnx.py]--> ONNX + parity reference --[trtexec]--> engine --[cdm_probe]--> verified
#
# Each step is skipped when its output exists, so re-running is cheap; the
# probe always runs. An engine serves one camera resolution and is tied to the
# TensorRT version and GPU that built it.
#
# Settings come from an engine config, CDM_CONFIG (default /config/engine.yaml,
# i.e. config/engine.yaml in the repository). The same file tells cdm_dataset
# and cdm_realsense which engine to load and at what resolution.
set -euo pipefail

CDM_ROOT="${CDM_ROOT:-/opt/cdm}"
PY="${CDM_PY:-/opt/cdm-py/bin/python}"
CONFIG="${CDM_CONFIG:-/config/engine.yaml}"
[[ -f "${CONFIG}" ]] || { echo "[cdm][ERROR] engine config not found: ${CONFIG}" >&2; exit 1; }

# Read the config into shell variables. Unknown keys are errors, so a typo
# cannot fall back to a default.
SETTINGS="$("${PY}" - "${CONFIG}" "${CDM_ROOT}" <<'PYEOF'
import os, shlex, sys
import yaml

path, root = sys.argv[1], sys.argv[2]
with open(path) as f:
    cfg = yaml.safe_load(f) or {}
base = os.path.dirname(os.path.abspath(path))
required = ["model", "width", "height", "precision", "input_size", "engine"]
optional = {
    "checkpoint": None, "reference_dir": None,
    "reference_rgb": os.path.join(root, "assets/example_data/color_12.png"),
    "reference_depth": os.path.join(root, "assets/example_data/depth_12.png"),
    "reference_depth_factor": 1000, "opset": 17, "no_tf32": False, "workspace_mb": 4096,
}
unknown = sorted(set(cfg) - set(required) - set(optional))
missing = [k for k in required if cfg.get(k) in (None, "")]
if unknown or missing:
    sys.exit(f"{path}: unknown keys {unknown}, missing keys {missing}")
if cfg["precision"] not in ("fp16", "fp32"):
    sys.exit(f"{path}: precision must be fp16 or fp32")
if not str(cfg["engine"]).endswith(".engine"):
    sys.exit(f"{path}: engine must end in .engine")
v = {**optional, **{k: val for k, val in cfg.items() if val not in (None, "")}}
resolve = lambda p: p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))
engine = resolve(str(v["engine"]))
models = os.path.dirname(engine)
tag = f"cdm_{v['model']}_{int(v['width'])}x{int(v['height'])}"
out = {
    "CAMERA": v["model"], "WIDTH": int(v["width"]), "HEIGHT": int(v["height"]),
    "PRECISION": v["precision"], "INPUT_SIZE": int(v["input_size"]), "OPSET": int(v["opset"]),
    "ENGINE": engine, "ONNX": engine[: -len(".engine")] + ".onnx", "MODELS": models,
    "CHECKPOINT": resolve(v["checkpoint"]) if v["checkpoint"] else os.path.join(models, f"cdm_{v['model']}.ckpt"),
    # FP32 PyTorch whatever the precision, so shared between precisions by default.
    "REFERENCE": resolve(v["reference_dir"]) if v["reference_dir"] else os.path.join(models, tag + "_reference"),
    "REF_RGB": resolve(str(v["reference_rgb"])), "REF_DEPTH": resolve(str(v["reference_depth"])),
    "DEPTH_FACTOR": v["reference_depth_factor"], "WORKSPACE_MB": int(v["workspace_mb"]),
    "NO_TF32": "1" if v["no_tf32"] else "",
}
for k, val in out.items():
    print(f"{k}={shlex.quote(str(val))}")
PYEOF
)" || { echo "[cdm][ERROR] invalid engine config ${CONFIG}" >&2; exit 1; }
eval "${SETTINGS}"
STEM="${ENGINE%.engine}"
# What the ONNX and engine at STEM were built from. A rerun with other settings
# must not quietly reuse them.
BUILD_SETTINGS="model=${CAMERA} size=${WIDTH}x${HEIGHT} precision=${PRECISION} input_size=${INPUT_SIZE} opset=${OPSET} no_tf32=${NO_TF32:-0} checkpoint=${CHECKPOINT}"

log() { echo "[cdm] $*"; }
die() { echo "[cdm][ERROR] $*" >&2; exit 1; }

nvidia-smi -L >/dev/null 2>&1 || die "no GPU visible; run the container with --gpus all"
mkdir -p "${MODELS}"
if [[ -f "${STEM}.settings" ]]; then
  [[ "$(cat "${STEM}.settings")" == "${BUILD_SETTINGS}" ]] || die "${STEM}.* was built with
    $(cat "${STEM}.settings")
  but ${CONFIG} now asks for
    ${BUILD_SETTINGS}
  Give the engine a new name in the config, or delete ${STEM}.{engine,onnx,onnx.data,settings}."
elif [[ -f "${ONNX}" || -f "${ENGINE}" ]]; then
  die "${ONNX} or ${ENGINE} exists without ${STEM}.settings, so what it was built from is unknown; delete it or pick another engine name"
else
  echo "${BUILD_SETTINGS}" > "${STEM}.settings"
fi

# ---------------------------------------------------------------------------
# 1. checkpoint
# ---------------------------------------------------------------------------
if [[ ! -f "${CHECKPOINT}" ]]; then
  REPO="depth-anything/camera-depth-model-${CAMERA}"
  log "checkpoint not found at ${CHECKPOINT}; looking it up in ${REPO}"
  read -r NAME SHA < <(curl -fsSL "https://huggingface.co/api/models/${REPO}/tree/main" | "${PY}" -c '
import json, sys
ckpts = [f for f in json.load(sys.stdin) if f["path"].endswith(".ckpt")]
names = [f["path"] for f in ckpts]
if len(ckpts) != 1:
    sys.exit("expected one .ckpt in the repository, found %s" % names)
print(ckpts[0]["path"], ckpts[0]["lfs"]["oid"])') || die "cannot list ${REPO} (unknown model in ${CONFIG}?)"
  log "downloading ${NAME} (weights licensed CC-BY-NC-4.0)"
  curl -fL --retry 5 -C - -o "${CHECKPOINT}.part" "https://huggingface.co/${REPO}/resolve/main/${NAME}"
  echo "${SHA}  ${CHECKPOINT}.part" | sha256sum -c - >/dev/null || die "checksum mismatch for ${NAME}"
  mv "${CHECKPOINT}.part" "${CHECKPOINT}"
fi
log "checkpoint: ${CHECKPOINT}"

# ---------------------------------------------------------------------------
# 2. ONNX + parity reference
# ---------------------------------------------------------------------------
if [[ ! -f "${ONNX}" ]]; then
  "${PY}" "${CDM_ROOT}/scripts/export_onnx.py" \
    --checkpoint "${CHECKPOINT}" \
    --image-width "${WIDTH}" --image-height "${HEIGHT}" \
    --input-size "${INPUT_SIZE}" --precision "${PRECISION}" --opset "${OPSET}" \
    --output "${ONNX}" \
    --reference-rgb "${REF_RGB}" --reference-depth "${REF_DEPTH}" \
    --depth-factor "${DEPTH_FACTOR}" --resize-reference \
    --reference-dir "${REFERENCE}"
fi

# ---------------------------------------------------------------------------
# 3. engine
# ---------------------------------------------------------------------------
# TensorRT >= 10 builds strongly typed: the precision is the ONNX's. TF32 is
# still allowed for fp32 matmuls unless no_tf32 is set. The workspace suffix
# must be M ("MiB" is parsed as bytes).
if [[ ! -f "${ENGINE}" ]]; then
  log "building ${PRECISION} engine on $(nvidia-smi --query-gpu=name --format=csv,noheader) ..."
  trtexec --onnx="${ONNX}" --saveEngine="${ENGINE}" --skipInference \
    ${NO_TF32:+--noTF32} --memPoolSize="workspace:${WORKSPACE_MB}M" \
    > "${STEM}.trtexec.log" 2>&1 || { tail -20 "${STEM}.trtexec.log" >&2; die "trtexec failed (log: ${STEM}.trtexec.log)"; }
  log "engine: ${ENGINE}"
fi

# ---------------------------------------------------------------------------
# 4. verify against the Python reference
# ---------------------------------------------------------------------------
cdm_probe --engine "${ENGINE}" --reference "${REFERENCE}" --input-size "${INPUT_SIZE}" \
  || die "the engine does not reproduce the Python model; see the probe output above"
log "ready: ${ENGINE}"
