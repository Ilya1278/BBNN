"""Сборка финального XAI-панно 2×3."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import matplotlib
matplotlib.use("Agg")  # без GUI, иначе на Windows может подвиснуть
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.xai.gradcam import GradCAM, get_target_layer
from src.xai.saliency import integrated_gradients, vanilla_saliency


def _load_as_uint8_hwc(image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    if isinstance(image, (str, Path)):
        return np.array(Image.open(image).convert("RGB"))
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Ожидался HWC RGB, получено {arr.shape}")
        return arr
    raise TypeError(f"Неподдерживаемый тип: {type(image)}")


def _class_id(classifier, label_str: str) -> int:
    classes = getattr(classifier, "CLASSES", ["not_motorcycle", "motorcycle"])
    if label_str in classes:
        return classes.index(label_str)
    return 0


def make_xai_panel(
    image: Union[str, Path, np.ndarray, Image.Image],
    model,
    classifier,
    out_path: Union[str, Path],
    ig_steps: int = 32,
    alpha: float = 0.5,
    dpi: int = 150,
    figsize=(12, 8),
) -> Path:
    """
    Panel 2x3:
      [original | gradcam | overlay]
      [saliency | ig      | probs bar]

    Работает через публичный API MotorcycleClassifier:
      - predict(image) → dict {label, confidence, probs, model_mode}
      - transform(PIL) → нормализованный тензор
      - model, device, CLASSES — свойства класса.
    """
    img = _load_as_uint8_hwc(image)

    # 1) предсказание через публичный контракт
    pred = classifier.predict(img)
    label_str = pred["label"]
    conf = pred.get("confidence", None)
    probs_dict = pred.get("probs", None)
    class_id = _class_id(classifier, label_str)

    # 2) нормализованный тензор через classifier.transform (Resize+ToTensor+Normalize)
    pil = Image.fromarray(img)
    tensor = classifier.transform(pil).unsqueeze(0).to(classifier.device)

    # 3) Grad-CAM
    target_layer = get_target_layer(model, "auto")
    cam = GradCAM(model, target_layer)
    try:
        heatmap = cam.generate(tensor, target_class=class_id)
    finally:
        cam.remove_hooks()
    overlay = cam.overlay(img, heatmap, alpha=alpha)

    # 4) saliency + IG
    sal = vanilla_saliency(model, tensor, target_class=class_id)
    ig = integrated_gradients(model, tensor, target_class=class_id, steps=ig_steps)

    # 5) панель
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    axes[0, 0].imshow(img);         axes[0, 0].set_title("Original");           axes[0, 0].axis("off")
    axes[0, 1].imshow(heatmap, cmap="jet"); axes[0, 1].set_title("Grad-CAM");   axes[0, 1].axis("off")
    axes[0, 2].imshow(overlay);     axes[0, 2].set_title("Overlay");            axes[0, 2].axis("off")
    axes[1, 0].imshow(sal, cmap="hot"); axes[1, 0].set_title("Vanilla Saliency"); axes[1, 0].axis("off")
    axes[1, 1].imshow(ig, cmap="hot");  axes[1, 1].set_title("Integrated Gradients"); axes[1, 1].axis("off")

    if isinstance(probs_dict, dict) and probs_dict:
        classes = getattr(classifier, "CLASSES", list(probs_dict.keys()))
        order = [c for c in classes if c in probs_dict]
        vals = [float(probs_dict[c]) for c in order]
        axes[1, 2].barh(order, vals, color="#4477aa")
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_title("Probabilities")
    else:
        axes[1, 2].text(0.5, 0.5, "no probs", ha="center")
        axes[1, 2].axis("off")

    title = f"Predicted: {label_str}"
    if conf is not None:
        title += f"  conf={conf:.3f}"
    if "model_mode" in pred:
        title += f"  ({pred['model_mode']})"
    fig.suptitle(title)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path