"""Единая обёртка инференса для XAI / Attack / Bot агентов."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.models.factory import build_model


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MotorcycleClassifier:
    CLASSES = ["not_motorcycle", "motorcycle"]

    def __init__(self, checkpoint_path: str, cfg: dict, device: str = "cuda"):
        self.cfg = cfg
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        # --- приватные поля; доступ снаружи через @property (см. ниже) ---
        self._device = torch.device(device)
        self._model_mode: str = cfg["model"]["mode"]
        self._checkpoint_path = str(checkpoint_path)

        num_classes = int(cfg["model"].get("num_classes", 2))
        image_size = int(cfg.get("data", {}).get("image_size", 224))

        model = build_model(cfg, num_classes=num_classes)

        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception:
            # fallback для чекпоинтов с нетривиальными объектами
            # (наш формат безопасен, но подстраховываемся на будущее)
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ("model_state", "state_dict", "model"):
                if key in state:
                    state = state[key]
                    break
        model.load_state_dict(state)
        model.eval().to(self._device)
        self._model = model

        self._transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    # ------------------------------------------------------------------
    # Публичный доступ для XAI / Attack агентов
    # ------------------------------------------------------------------
    @property
    def model(self) -> nn.Module:
        """Underlying-модель в eval() режиме, на self.device.
        Разрешено делать forward и backward (Attack Agent)."""
        return self._model

    @property
    def device(self) -> torch.device:
        """Устройство, на котором лежат модель и веса."""
        return self._device

    @property
    def transform(self):
        """torchvision Compose: Resize → ToTensor → Normalize(ImageNet).
        Атаку делать ПОСЛЕ transform, на нормализованных тензорах."""
        return self._transform

    @property
    def model_mode(self) -> str:
        """'scratch' | 'pretrained'. Совпадает с полем в predict()."""
        return self._model_mode

    # ------------------------------------------------------------------
    # Внутренний хелпер
    # ------------------------------------------------------------------
    def _to_tensor(self, image: Union[np.ndarray, str, Path]) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            arr = image
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"Expected HWC RGB ndarray, got shape {arr.shape}")
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr, mode="RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        return self._transform(img)

    # ------------------------------------------------------------------
    # Публичный инференс — КОНТРАКТ СТАБИЛЕН
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, image: Union[np.ndarray, str]) -> dict:
        x = self._to_tensor(image).unsqueeze(0).to(self._device)
        logits = self._model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        return {
            "label": self.CLASSES[idx],
            "confidence": float(probs[idx]),
            "probs": {self.CLASSES[i]: float(probs[i]) for i in range(len(self.CLASSES))},
            "model_mode": self._model_mode,
        }

    @torch.no_grad()
    def predict_batch(self, images: list) -> list[dict]:
        if len(images) == 0:
            return []
        batch = torch.stack([self._to_tensor(im) for im in images]).to(self._device)
        logits = self._model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        out = []
        for i in range(probs.shape[0]):
            idx = int(probs[i].argmax())
            out.append({
                "label": self.CLASSES[idx],
                "confidence": float(probs[i, idx]),
                "probs": {self.CLASSES[j]: float(probs[i, j]) for j in range(len(self.CLASSES))},
                "model_mode": self._model_mode,
            })
        return out