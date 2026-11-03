from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .overfit_detector import detect_overfitting


def synthetic_overfit():
    epochs = np.arange(1, 21)
    train_loss = 1.2 * np.exp(-0.15 * epochs) + 0.05
    val_loss = 1.3 * np.exp(-0.18 * epochs) + 0.08
    mask = epochs > 10
    base = val_loss[9]
    k = epochs[mask] - 10
    val_loss[mask] = base + 0.03 * k + 0.005 * k ** 2
    return pd.DataFrame({
        "epoch": epochs, "train_loss": train_loss, "val_loss": val_loss,
        "train_acc": 0.5 + 0.02 * epochs, "val_acc": 0.5 + 0.015 * epochs,
        "lr": 0.001 * np.ones_like(epochs, dtype=float),
        "time_sec": 10.0 * np.ones_like(epochs, dtype=float),
    })


def synthetic_clean():
    epochs = np.arange(1, 21)
    train_loss = 1.2 * np.exp(-0.15 * epochs) + 0.05
    val_loss = 1.3 * np.exp(-0.16 * epochs) + 0.07
    return pd.DataFrame({
        "epoch": epochs, "train_loss": train_loss, "val_loss": val_loss,
        "train_acc": 0.5 + 0.02 * epochs, "val_acc": 0.5 + 0.018 * epochs,
        "lr": 0.001 * np.ones_like(epochs, dtype=float),
        "time_sec": 10.0 * np.ones_like(epochs, dtype=float),
    })


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "reports" / "_synthetic"
    out.mkdir(parents=True, exist_ok=True)

    overfit = synthetic_overfit()
    clean = synthetic_clean()
    overfit.to_csv(out / "synthetic_overfit.csv", index=False)
    clean.to_csv(out / "synthetic_clean.csv", index=False)

    print("synthetic_overfit:", detect_overfitting(overfit, window=5))
    print()
    print("synthetic_clean:", detect_overfitting(clean, window=5))


if __name__ == "__main__":
    main()
