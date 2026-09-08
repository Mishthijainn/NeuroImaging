"""Grad-CAM for the 3D spatial stream of CAFN.

Backprop starts from the *fused* classification logit by default, so the
resulting activation map reflects which anatomical zones drove the
multimodal decision (not just the unimodal spatial head) -- gradients flow
from `logits_fused` back through the cross-attention fusion module into the
spatial encoder's last residual block.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM3D:
    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self.target_layer = target_layer or model.spatial_encoder.last_conv
        self._activations = None
        self._gradients = None
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def generate(
        self,
        mri_volume: torch.Tensor,
        vep_signal: torch.Tensor,
        target_class: int | torch.Tensor | None = None,
        logits_key: str = "logits_fused",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cam, target_class): cam is (B, D, H, W) in [0, 1]."""
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        outputs = self.model(mri_volume, vep_signal)
        logits = outputs[logits_key]

        if target_class is None:
            target_class = logits.argmax(dim=1)
        elif isinstance(target_class, int):
            target_class = torch.full(
                (logits.shape[0],), target_class, dtype=torch.long, device=logits.device
            )

        selected = logits.gather(1, target_class.view(-1, 1)).sum()
        selected.backward()

        if was_training:
            self.model.train()

        activations = self._activations  # (B, C, d, h, w)
        gradients = self._gradients  # (B, C, d, h, w)
        if activations is None or gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients; check target_layer")

        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))  # (B, 1, d, h, w)

        target_size = mri_volume.shape[2:]
        cam = F.interpolate(cam, size=target_size, mode="trilinear", align_corners=False)
        cam = cam.squeeze(1)  # (B, D, H, W)

        cam_min = cam.amin(dim=(1, 2, 3), keepdim=True)
        cam_max = cam.amax(dim=(1, 2, 3), keepdim=True)
        denom = (cam_max - cam_min).clamp(min=1e-8)
        cam = (cam - cam_min) / denom

        return cam.detach(), target_class.detach()


def compute_gradcam(
    model: nn.Module,
    mri_volume: torch.Tensor,
    vep_signal: torch.Tensor,
    target_class: int | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-shot convenience wrapper around GradCAM3D that cleans up its hooks."""
    cam_engine = GradCAM3D(model)
    try:
        return cam_engine.generate(mri_volume, vep_signal, target_class)
    finally:
        cam_engine.remove_hooks()


def plot_gradcam_slices(
    volume: np.ndarray,
    cam: np.ndarray,
    save_path: str,
    title: str = "Grad-CAM: Visual Cortex Activation",
) -> None:
    """Overlay a Grad-CAM heatmap on the mid axial/coronal/sagittal slices."""
    if volume.shape != cam.shape:
        raise ValueError(f"volume shape {volume.shape} != cam shape {cam.shape}")

    d, h, w = volume.shape
    slices = [
        (volume[d // 2, :, :], cam[d // 2, :, :], "Axial"),
        (volume[:, h // 2, :], cam[:, h // 2, :], "Coronal"),
        (volume[:, :, w // 2], cam[:, :, w // 2], "Sagittal"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (vol_slice, cam_slice, name) in zip(axes, slices):
        ax.imshow(vol_slice, cmap="gray")
        ax.imshow(cam_slice, cmap="jet", alpha=0.5)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
