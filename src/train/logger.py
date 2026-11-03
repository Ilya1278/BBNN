"""CSV-логгер эпох + снапшот конфига + снапшот манифеста."""
from __future__ import annotations
import json
import csv
import shutil
from datetime import datetime
from pathlib import Path

import yaml


class RunLogger:
    COLUMNS = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr", "time_sec"]

    def __init__(
        self,
        cfg: dict,
        runs_dir: str | Path = "runs",
        run_name: str = "scratch",
    ):
        self.cfg = cfg
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{run_name}_{ts}"
        self.run_dir = Path(runs_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # снапшот конфига
        with open(self.run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

        # CSV с шапкой
        self.csv_path = self.run_dir / "metrics.csv"
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(self.COLUMNS)

    # --- снапшот манифеста ДО старта обучения ---
    def snapshot_manifest(self, manifest_path: str | Path) -> Path:
        """
        Простая копия манифеста в run_dir. ПУТИ НЕ ТРОГАЕМ —
        MotorcycleDataset ожидает относительные ('train/...').
        Связка с data/processed/ делается через root= у датасета.
        """
        src = Path(manifest_path)
        if not src.exists():
            raise FileNotFoundError(f"manifest not found: {src}")
        dst = self.run_dir / "manifest_snapshot.json"
        shutil.copy2(src, dst)
        return dst

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float,
        time_sec: float,
    ) -> None:
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, train_loss, val_loss, train_acc, val_acc, lr, time_sec]
            )