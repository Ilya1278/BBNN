"""Vanilla saliency + Integrated Gradients. Через backward по входу."""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch

from src.xai.gradcam import _to_normalized_tensor


def vanilla_saliency(
    model: torch.nn.Module,
    image: Union[np.ndarray, torch.Tensor],
    target_class: Optional[int] = None,
) -> np.ndarray:
    """
    |d(output[target_class]) / d(input)| по каждому пикселю.
    Возвращает HxW float [0,1] (max по каналам).
    """
    device = next(model.parameters()).device
    model.eval()

    tensor = _to_normalized_tensor(image, device).clone().detach().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = model(tensor)
    if target_class is None:
        target_class = int(logits.argmax(dim=1).item())

    logits[0, target_class].backward()

    grad = tensor.grad[0].detach().cpu().numpy()   # [3, H, W]
    sal = np.max(np.abs(grad), axis=0)             # [H, W]

    sal = sal - sal.min()
    if sal.max() > 0:
        sal = sal / sal.max()
    return sal.astype(np.float32)


def integrated_gradients(
    model: torch.nn.Module,
    image: Union[np.ndarray, torch.Tensor],
    target_class: Optional[int] = None,
    steps: int = 32,
) -> np.ndarray:
    """
    IG с baseline=zeros. Аппроксимация Римана: steps+1 точек, включая 0 и 1.
    Возвращает HxW float [0,1].
    """
    device = next(model.parameters()).device
    model.eval()

    tensor = _to_normalized_tensor(image, device).detach()

    if target_class is None:
        with torch.no_grad():
            target_class = int(model(tensor).argmax(dim=1).item())

    baseline = torch.zeros_like(tensor)
    total_grads = torch.zeros_like(tensor)

    for i in range(steps + 1):
        alpha = i / steps
        interpolated = baseline + alpha * (tensor - baseline)
        interpolated = interpolated.clone().detach().requires_grad_(True)

        model.zero_grad(set_to_none=True)
        logits = model(interpolated)
        logits[0, target_class].backward()
        total_grads += interpolated.grad.detach()

    avg_grads = total_grads / (steps + 1)
    ig = (tensor - baseline) * avg_grads           # [1, 3, H, W]
    ig = ig[0].detach().cpu().numpy()              # [3, H, W]
    ig = np.sum(np.abs(ig), axis=0)                # [H, W]

    ig = ig - ig.min()
    if ig.max() > 0:
        ig = ig / ig.max()
    return ig.astype(np.float32)