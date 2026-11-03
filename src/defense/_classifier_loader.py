"""Загрузка MotorcycleClassifier по реальной сигнатуре (§2 INTERFACES.md):

    MotorcycleClassifier(checkpoint_path: str, cfg: dict, device: str = "cuda")

`cfg` — распарсенный YAML из runs/{run_id}/config.yaml (содержит augment.*,
нужные для сборки transform внутри classifier).
"""
import yaml
import torch


def load_classifier(checkpoint: str, model_config: str, device: str | None = None):
    with open(model_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from src.models.classifier import MotorcycleClassifier
    clf = MotorcycleClassifier(checkpoint_path=checkpoint, cfg=cfg, device=device)
    clf.model.eval()
    return clf