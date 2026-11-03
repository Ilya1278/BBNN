from __future__ import annotations

from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def get_target_layer(model: nn.Module, target_layer: str = "auto") -> nn.Module:
    """
    Автовыбор target_layer:
      - 'auto' → последний nn.Conv2d в модели
        (для MotorcycleCNN — последний conv-блок,
         для resnet18 — layer4[-1].conv2)
      - иначе — точное имя слоя из model.named_modules()
    """
    if target_layer != "auto":
        for name, module in model.named_modules():
            if name == target_layer:
                return module
        raise ValueError(f"target_layer='{target_layer}' не найден в модели")

    last_conv: Optional[nn.Module] = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("В модели нет ни одного nn.Conv2d")
    return last_conv

def _to_normalized_tensor(
    image: Union[np.ndarray, torch.Tensor], device: torch.device
) -> torch.Tensor:

    if isinstance(image, torch.Tensor):
        t = image
        if t.dim() == 3:
            t = t.unsqueeze(0)
        return t.to(device).float()

    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Ожидался HWC RGB ndarray, получено {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        x = arr.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        return t.to(device).float()

    raise TypeError(f"Неподдерживаемый тип изображения: {type(image)}")

class GradCAM:

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        # full_backward_hook — не-deprecated вариант register_backward_hook
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    # ---------- hooks ----------
    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        # grad_out[0] — градиент по выходу target_layer, форма [B, C, H', W']
        self.gradients = grad_out[0].detach()

    def remove_hooks(self):

        self._fwd_handle.remove()
        self._bwd_handle.remove()
        self.activations = None
        self.gradients = None

    # ---------- main ----------
    def generate(
        self,
        image: Union[np.ndarray, torch.Tensor],
        target_class: Optional[int] = None,
    ) -> np.ndarray:

        device = next(self.model.parameters()).device
        self.model.eval()

        tensor = _to_normalized_tensor(image, device)

        # сбрасываем накопленные значения перед forward
        self.activations = None
        self.gradients = None
        self.model.zero_grad(set_to_none=True)

        logits = self.model(tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[0, target_class]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("hooks не захватили activations/gradients")

        # α_k = global average pool градиентов по spatial dim
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # [B, C, 1, 1]
        # L = ReLU( Σ_k α_k * A_k )
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B, 1, H', W']
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=tensor.shape[2:], mode="bilinear", align_corners=False
        )

        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32)

    def overlay(
        self, image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        img = image.copy()

        hm = cv2.resize(heatmap.astype(np.float32), (img.shape[1], img.shape[0]))
        hm = np.uint8(np.clip(hm, 0, 1) * 255)
        colored = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)  # cv2 даёт BGR

        return cv2.addWeighted(img, 1.0 - alpha, colored, alpha, 0)