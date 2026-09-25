# Camera Depth Models (CDM)

This repository is a lightweight fork of the CDM module released in [ByteDance-Seed/manip-as-in-sim-suite](https://github.com/ByteDance-Seed/manip-as-in-sim-suite), a depth estimation library that leverages Vision Transformer encoders to turn noisy RGB-D sensor readings into clean, metric depth maps.   



## About This Fork

CDM runs here as a C++ TensorRT runtime in one Docker image, `cdm-trt`. Compose
defines three services that share that image: `engine` (build and verify a
TensorRT engine), `dataset` (run on a recorded sequence), and `realsense` (run
live on an Intel RealSense).

## Installation

You need an NVIDIA GPU, a driver with CUDA 12.8 support, Docker, and the
NVIDIA Container Toolkit.

| Path | What it is |
|---|---|
| `docker/Dockerfile` | CUDA 12.8, TensorRT, OpenCV, librealsense (libusb backend) |
| `docker/docker-compose.yaml` | services `engine`, `dataset`, `realsense` |
| `docker/build_engine.sh` | in the container: checkpoint → ONNX → engine → parity check (`cdm_build_engine`) |
| `scripts/export_onnx.py` | checkpoint → ONNX (network only, fixed resolution) + a parity reference |
| `cpp/` | C++ library (`cdm::DepthCompleter`) and tools `cdm_dataset`, `cdm_realsense`, `cdm_probe` |
| `config/` | host paths (`paths.env`) and the settings of each service (`*.yaml`) |

All commands below run from the repository root. From `docker/`, drop the
`-f docker/docker-compose.yaml`.

Compose reads host paths from `config/paths.env` through the symlink
`docker/.env` (gitignored, so create it once):

```bash
ln -sfn ../config/paths.env docker/.env
```

Before the first build, set `CDM_DATASETS` in `config/paths.env` to your
datasets directory and `CDM_UID` / `CDM_GID` to the output of `id -u` / `id -g`.
Relative paths in `paths.env` are relative to `docker/`. Paths in the YAML
files are container paths (`/data/...`, `/models/...`, `/output/...`).

Build the image (shared by every service):

```bash
docker compose -f docker/docker-compose.yaml build engine
```

Each service has a profile of its own name, so `docker compose up` alone starts
nothing. Run one service at a time:

| Service | Command | What it does |
|---|---|---|
| `engine` | `docker compose -f docker/docker-compose.yaml run --rm engine` | download the checkpoint if needed, build the engine, verify it with `cdm_probe` |
| `dataset` | `docker compose -f docker/docker-compose.yaml run --rm dataset` | run CDM on the sequence in `config/dataset.yaml` |
| `realsense` | `docker compose -f docker/docker-compose.yaml run --rm realsense` | stream a RealSense live and complete its depth |
| `realsense` | `docker compose -f docker/docker-compose.yaml run --rm realsense --list` | list connected cameras |

Arguments after the service name override that service's YAML. Unknown keys are
errors.

```bash
docker compose -f docker/docker-compose.yaml run --rm dataset --max-frames 10 --show
```

For several datasets, keep one YAML per sequence in `config/` and pick one with
`CDM_DATASET_CONFIG=other.yaml docker compose -f docker/docker-compose.yaml run --rm dataset`.

## Configuration

| File | Sets |
|---|---|
| `config/paths.env` | host directories mounted as `/data` (datasets, read-only), `/models`, `/output`; your UID/GID; which config file each service uses |
| `config/engine.yaml` | checkpoint model, camera resolution, precision, engine file. Also tells the other two services which engine to load and at what resolution |
| `config/dataset.yaml` | the sequence, its depth factor, the camera intrinsics, output directory and options |
| `config/realsense.yaml` | camera serial, fps, recording, display |

The YAML keys are the tools' command-line options with `_` for `-`.

**Camera intrinsics are not used by CDM**: the network sees images only. The
`camera` block of a dataset config is checked against the frame size and
written to `<output>/camera.yaml`, so the CDM depth can be back-projected later
with the calibration of the camera that took it. RealSense recordings write
the device's factory intrinsics in the same format, which a dataset config can
name directly (`camera: /data/<recording>/camera.yaml`).

### `engine`

An engine serves **one camera resolution** and only runs on the TensorRT version
and GPU that built it. The tools refuse a frame of another resolution rather
than resize it. The builder skips steps whose output exists and records the
settings each engine was built from; it refuses to reuse an engine when
`engine.yaml` asks for different ones.

The last step of `engine` runs `cdm_probe`, which compares the engine with the
reference model on one frame (by default the example frame, resized to the
camera resolution). It checks three stages separately (preprocessing, network,
end to end) and fails if the median depth error exceeds 0.5 % or the 99th
percentile exceeds 2 %.

### `dataset`

A TUM-style sequence (`associations.txt`: `rgb_ts rgb_path depth_ts depth_path`),
or two folders of RGB and uint16 depth images paired by file name, as set in
`config/dataset.yaml`:

```bash
docker compose -f docker/docker-compose.yaml run --rm dataset
```

The output directory then holds `depth/` (CDM depth as uint16 PNGs, same depth
factor as the input), `associations.txt` (the input RGB paired with the CDM
depth, so the output is a sequence of its own; its RGB paths are absolute as the
tool saw them, i.e. `/data/...`), `viz/` (RGB | sensor | CDM), `timings.csv`,
`camera.yaml` and `run.yaml` (every setting in effect).

### `realsense`

```bash
xhost +local:docker                       # once per session, lets the container open a window
docker compose -f docker/docker-compose.yaml run --rm realsense
docker compose -f docker/docker-compose.yaml run --rm realsense --list  # connected cameras
```

Colour and depth stream at the engine's resolution and depth is aligned to
colour. With `record: /output/<name>` in `config/realsense.yaml` (or
`--record /output/<name>`), every frame is saved as `rgb/`, `depth/` (raw
aligned sensor depth), `cdm/`, `associations.txt` and `camera.yaml`; the
recording replays through the dataset service. `no_display: true` prints
statistics instead of opening a window. The container runs privileged, as
root, with `/dev` mounted, because the libusb backend needs the USB device
nodes; recordings are therefore owned by root.

### Output conventions

- Depth is metres internally, and uint16 PNGs at the input's depth factor on disk.
- 0 means no depth. Where the network predicts zero inverse depth, the tools
  write 0; a depth beyond the uint16 range is also written as 0 and counted,
  never saturated to a false maximum.
- The output is the network's depth at every pixel, including the holes in the
  sensor depth: CDM completes depth, it does not only denoise the measured pixels.

## Overview

Camera Depth Models are sensor-specific depth networks trained to produce clean, simulation-like depth maps from noisy real-world inputs. By bridging the visual gap between simulation and reality, CDMs allow robotic policies trained in simulation to operate on real hardware with minimal adaptation.

### Key Features

- **Metric Depth Estimation** – produces absolute depth in meters.
- **Multi-Camera Support** – tuned checkpoints for Intel RealSense D405/D435/L515, Stereolabs ZED 2i, and Azure Kinect.
- **Real-time Ready** – lightweight inference suitable for robot control loops.
- **Sim-to-Real Transfer** – outputs depth maps that mimic the noise profile of simulation.

## Architecture

CDM relies on a dual-branch Vision Transformer design:

- **RGB Branch** extracts semantic context from RGB images.
- **Depth Branch** processes noisy depth measurements.
- **Cross-Attention Fusion** blends semantic cues and scale cues.
- **DPT Decoder** reconstructs the final metric depth map.

Supported ViT encoder sizes:

- `vits`: 64 features / 384 channels
- `vitb`: 128 features / 768 channels
- `vitl`: 256 features / 1024 channels *(all released checkpoints use this configuration)*
- `vitg`: 384 features / 1536 channels

## Training Pipeline

The upstream authors train CDMs using synthetic datasets augmented with learned camera-specific noise models:

1. **Noise Modeling** – learn hole/value noise patterns from real sensor captures.
2. **Synthetic Data Generation** – apply the noise models to clean simulation depth.
3. **CDM Training** – train the ViT-based model on this synthetic-but-realistic corpus.

Datasets include HyperSim, DREDS, HISS, and IRS (over 280k images).

## Supported Cameras

We distribute checkpoints for:

- Intel RealSense D405 / D435 / L515
- Stereolabs ZED 2i (Quality + Neural modes)
- Microsoft Azure Kinect

## Performance

The released CDMs achieve state-of-the-art accuracy on metric depth estimation:

- Higher accuracy than prompt-guided monocular depth estimators.
- Strong zero-shot generalization across camera hardware.
- Fast enough for closed-loop manipulation policies.

## Citation

If you use Camera Depth Models in your research, please cite the original paper:

```bibtex
@article{liu2025manipulation,
  title={Manipulation as in Simulation: Enabling Accurate Geometry Perception in Robots},
  author={Liu, Minghuan and Zhu, Zhengbang and Han, Xiaoshen and Hu, Peng and Lin, Haotong and
          Li, Xinyao and Chen, Jingxiao and Xu, Jiafeng and Yang, Yichu and Lin, Yunfeng and
          Li, Xinghang and Yu, Yong and Zhang, Weinan and Kong, Tao and Kang, Bingyi},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is distributed under the Apache 2.0 License. See `LICENSE` for the full text.
