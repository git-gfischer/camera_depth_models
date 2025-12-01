from camera_depth_models.dpt import RGBDDepth

import torch

model_configs = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },  # Small
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },  # Base
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },  # Large
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },  # Giant
}


def load_model(encoder: str, model_path: str, device: str) -> RGBDDepth:
    """Load and initialize the RGBD depth estimation model.

    Args:
        encoder: Model encoder type ('vits', 'vitb', 'vitl', 'vitg')
        model_path: Path to the model checkpoint file

    Returns:
        torch.nn.Module: Loaded model in evaluation mode
    """
    # Initialize model with configuration for specified encoder
    model = RGBDDepth(**model_configs[encoder])

    # Load checkpoint and extract state dict
    checkpoint = torch.load(model_path, map_location="cpu")
    if "model" in checkpoint:
        # Handle checkpoints that wrap state dict in 'model' key
        # Remove 'module.' prefix if present (from DataParallel training)
        states = {k[7:]: v for k, v in checkpoint["model"].items()}
    elif "state_dict" in checkpoint:
        states = checkpoint["state_dict"]
        states = {k[9:]: v for k, v in states.items()}
    else:
        # Direct state dict checkpoint
        states = checkpoint

    # Load weights and move to device
    model.load_state_dict(states, strict=False)
    model = model.to(device).eval()

    print(f"Model loaded: {encoder} from {model_path}")
    return model
