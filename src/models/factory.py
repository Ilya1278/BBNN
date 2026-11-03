"""Фабрика моделей: единственная точка переключения scratch / pretrained."""
from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm

from src.models.scratch_cnn import MotorcycleCNN


def build_model(cfg: dict, num_classes: int = 2) -> nn.Module:
    """
    Единый интерфейс. Код обучения НЕ знает о режиме — просто вызывает фабрику.
    cfg["model"]["mode"] in {"scratch", "pretrained"}.
    """
    mode = cfg["model"]["mode"]

    if mode == "scratch":
        dropout = float(cfg["model"].get("dropout", 0.5))
        return MotorcycleCNN(num_classes=num_classes, dropout=dropout)

    if mode == "pretrained":
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1
        m = tvm.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    raise ValueError(f"Unknown mode: {mode!r} (expected 'scratch' or 'pretrained')")