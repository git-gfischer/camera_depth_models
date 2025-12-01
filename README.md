# Camera Depth Models (CDM)

This repository is a lightweight fork of the CDM module released in [ByteDance-Seed/manip-as-in-sim-suite](https://github.com/ByteDance-Seed/manip-as-in-sim-suite), a depth estimation library that leverages Vision Transformer encoders to turn noisy RGB-D sensor readings into clean, metric depth maps.   



## About This Fork
The goal of the fork is to make the camera depth models easy to install and use for inference as a standalone Python package. To this end:
- The original models are wrapped in a `camera_depth_models` package
- The inference API has been slighly simplified: it handles conversions internally so you can just call `model.infer_depth` and get a metric depth estimate.

## Getting Started

### Installation

```bash
pip install -e .
```

### Download Pretrained Checkpoints

Pretrained CDM models for each supported camera are available on Hugging Face:  
https://huggingface.co/collections/depth-anything/camera-depth-models-68b521181dedd223f4b020db

Download the checkpoints you need (for example, `cdm_d435.ckpt`) and place them wherever you keep large model files.

### Example Script

We provide a minimal example in `scripts/main.py`. Update the `model_path` as needed and run. The script loads a model with `camera_depth_models.load_model`, runs `infer_depth`, and displays the RGB input, raw sensor depth, and the estimated metric depth side-by-side.

### Python API

```python
import cv2
import numpy as np

from camera_depth_models import load_model

device = "cuda"
model = load_model("vitl", "path/to/model.ckpt", device)

rgb = cv2.imread("assets/example_data/color_12.png")[:, :, ::-1]  # BGR -> RGB
depth = cv2.imread("assets/example_data/depth_12.png", cv2.IMREAD_UNCHANGED) / 1000.0

# Returns metric depth (meters), not inverse depth
pred_depth = model.infer_depth(rgb, depth, input_size=518)
```

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
