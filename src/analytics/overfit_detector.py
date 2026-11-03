from __future__ import annotations

import numpy as np
import pandas as pd


def detect_overfitting(metrics_df, window=5, slope_eps=1e-4):
    """Slope-based детектор. Окно поиска начинается СТРОГО после best_epoch,
    чтобы минимум val_loss не попадал внутрь overfit-окна.
    """
    df = metrics_df.sort_values("epoch").reset_index(drop=True)

    if len(df) == 0:
        return {
            "best_epoch": -1, "best_val_loss": float("nan"),
            "overfit_start_epoch": None, "underfit_until_epoch": None,
            "interpretation": "Пустой metrics.csv: анализировать нечего.",
        }

    best_idx = int(df["val_loss"].idxmin())
    best_epoch = int(df.loc[best_idx, "epoch"])
    best_val_loss = float(df.loc[best_idx, "val_loss"])

    val = df["val_loss"].to_numpy(dtype=float)
    train = df["train_loss"].to_numpy(dtype=float)
    epochs = df["epoch"].to_numpy(dtype=int)

    overfit_start_epoch = None
    # Окно начинается СТРОГО после best_epoch, чтобы минимум val_loss
    # не попадал внутрь окна (иначе детектор срабатывает раньше best_epoch).
    start_i = best_idx + window
    if start_i < len(df):
        x = np.arange(window, dtype=float)
        for i in range(start_i, len(df)):
            v = val[i - window + 1: i + 1]
            t = train[i - window + 1: i + 1]
            v_slope = np.polyfit(x, v, 1)[0]
            t_slope = np.polyfit(x, t, 1)[0]
            val_growing = v_slope > slope_eps and (v[-1] - v[0]) > 0
            train_falling = t_slope < -slope_eps and (t[-1] - t[0]) < 0
            if val_growing and train_falling:
                overfit_start_epoch = int(epochs[i])
                break

    underfit_until_epoch = None
    if len(df) >= 2:
        drops = (train[:-1] - train[1:]) / np.maximum(train[:-1], 1e-12)
        fast_drops = drops > 0.10
        if np.any(fast_drops):
            last_fast = int(np.where(fast_drops)[0][-1])
            underfit_until_epoch = int(epochs[last_fast + 1])

    parts = [f"Лучшая эпоха: {best_epoch}, min val_loss={best_val_loss:.4f}."]
    if overfit_start_epoch is None:
        parts.append(
            f"Устойчивого переобучения не найдено: val_loss не показывает "
            f"положительного тренда на окне {window} эпох после best_epoch "
            f"при падающем train_loss."
        )
    else:
        parts.append(
            f"Переобучение начинается около эпохи {overfit_start_epoch}: "
            f"val_loss в среднем растёт на окне {window} эпох, train_loss продолжает падать."
        )
    if underfit_until_epoch is None:
        parts.append("Быстрой фазы падения train_loss (>10% за эпоху) не обнаружено.")
    else:
        parts.append(
            f"До эпохи {underfit_until_epoch} включительно train_loss падал быстро "
            f"(>10% за эпоху), после — темп замедлился."
        )
    parts.append(f"Рекомендуется использовать model_best.pt из эпохи {best_epoch}.")

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "overfit_start_epoch": overfit_start_epoch,
        "underfit_until_epoch": underfit_until_epoch,
        "interpretation": " ".join(parts),
    }
