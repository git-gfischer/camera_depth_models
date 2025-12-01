import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from camera_depth_models import load_model


if __name__ == "__main__":
    rgb_path = "assets/example_data/color_12.png"
    depth_path = "assets/example_data/depth_12.png"
    model_path = "assets/checkpoints/cdm_d435.ckpt"

    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    # assumes the depth images is saved as uint16 in millimeters
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.

    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model("vitl", model_path, device)

    depth_estimated = model.infer_depth(rgb, depth)

    depth_display_max = float(max(np.nanmax(depth), np.nanmax(depth_estimated)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis("off")
    axes[1].imshow(depth, cmap="magma", vmin=0, vmax=depth_display_max)
    axes[1].set_title("Depth (m)")
    axes[1].axis("off")
    axes[2].imshow(depth_estimated, cmap="magma", vmin=0, vmax=depth_display_max)
    axes[2].set_title("Estimated Depth (m)")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()
