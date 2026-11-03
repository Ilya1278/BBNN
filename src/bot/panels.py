"""Отрисовка XAI-панели 2×3 из НОРМАЛИЗОВАННОГО тензора.

Копия логики make_xai_panel (src/xai/visualize.py), но принимает тензор
напрямую — чтобы (а) не делать двойной clf.transform, (б) гарантировать
идентичный формат для clean и adv.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

log = logging.getLogger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denorm_to_uint8(x_norm: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalized ImageNet → HWC uint8 [0,255]."""
    x = x_norm.detach().cpu().numpy().transpose(1, 2, 0)
    x = x * _STD + _MEAN
    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def forward_probs(clf, x_norm: torch.Tensor) -> tuple[int, str, float, dict[str, float]]:
    """Forward через clf.model. x_norm: (1,3,H,W). → (idx, label, conf, probs_dict)."""
    with torch.no_grad():
        logits = clf.model(x_norm.to(clf.device))
        probs = torch.softmax(logits, dim=1)[0].detach().cpu()
    idx = int(probs.argmax().item())
    label = clf.CLASSES[idx]
    conf = float(probs[idx].item())
    probs_dict = {clf.CLASSES[i]: float(probs[i].item()) for i in range(len(clf.CLASSES))}
    return idx, label, conf, probs_dict


def build_xai_panel(
    x_norm: torch.Tensor,
    clf,
    out_path: Path,
    ig_steps: int = 32,
    alpha: float = 0.5,
    dpi: int = 150,
    figsize: tuple[float, float] = (12, 8),
) -> tuple[str, float, dict[str, float]]:
    """Панель 2×3 из НОРМАЛИЗОВАННОГО тензора.

    Формат идентичен make_xai_panel:
      Row 1: Original | Grad-CAM | Overlay
      Row 2: Vanilla Saliency | Integrated Gradients | Probabilities
      Заголовок: Predicted: <label>  conf=<conf>  (<model_mode>)

    Возвращает (label, confidence, probs_dict).
    """
    from src.xai.gradcam import GradCAM, get_target_layer
    from src.xai.saliency import vanilla_saliency, integrated_gradients

    if x_norm.dim() == 3:
        x_norm = x_norm.unsqueeze(0)
    x_norm = x_norm.to(clf.device)

    idx, label, conf, probs_dict = forward_probs(clf, x_norm)

    target_layer = get_target_layer(clf.model, "auto")
    gradcam = GradCAM(clf.model, target_layer)
    try:
        hm_gradcam = gradcam.generate(x_norm, target_class=idx)
        img_uint8 = denorm_to_uint8(x_norm[0])
        overlay = gradcam.overlay(img_uint8, hm_gradcam, alpha=alpha)
    finally:
        gradcam.remove_hooks()

    hm_sal = vanilla_saliency(clf.model, x_norm, target_class=idx)
    hm_ig = integrated_gradients(clf.model, x_norm, target_class=idx, steps=ig_steps)

    fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=dpi)
    axes[0, 0].imshow(img_uint8);                 axes[0, 0].set_title("Original")
    axes[0, 1].imshow(hm_gradcam, cmap="jet");    axes[0, 1].set_title("Grad-CAM")
    axes[0, 2].imshow(overlay);                   axes[0, 2].set_title("Overlay")
    axes[1, 0].imshow(hm_sal, cmap="hot");        axes[1, 0].set_title("Vanilla Saliency")
    axes[1, 1].imshow(hm_ig, cmap="hot");         axes[1, 1].set_title("Integrated Gradients")

    keys = list(probs_dict.keys())
    axes[1, 2].barh(keys, [probs_dict[k] for k in keys])
    axes[1, 2].set_title("Probabilities")
    axes[1, 2].set_xlim(0.0, 1.0)

    for ax in axes.ravel():
        ax.axis("off")
    axes[1, 2].axis("on")

    fig.suptitle(f"Predicted: {label}  conf={conf:.3f}  ({clf.model_mode})")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return label, conf, probs_dict