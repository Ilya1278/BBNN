from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from .overfit_detector import detect_overfitting
from .plots import (
    plot_accuracy_curves,
    plot_confusion_matrix,
    plot_epoch_time,
    plot_loss_curves,
    plot_lr_schedule,
    plot_overfit_zone,
    plot_per_class_metrics,
)


def _load_yaml(path):
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def generate_report(run_dir, out_dir, y_true=None, y_pred=None):
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    run_id = run_dir.name

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_csv(run_dir / "metrics.csv")
    run_cfg = _load_yaml(run_dir / "config.yaml")

    root = Path(__file__).resolve().parents[2]
    acfg = _load_yaml(root / "configs" / "analytics.yaml")

    window = int(acfg.get("window", 5))
    dpi = int(acfg.get("dpi", 150))
    figsize = tuple(acfg.get("figsize", [10, 6]))
    style = acfg.get("style", "seaborn-v0_8-whitegrid")

    try:
        plt.style.use(style)
    except Exception:
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except Exception:
            pass

    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["axes.grid"] = True

    plot_loss_curves(metrics_df, fig_dir / "loss_curves.png")
    plot_accuracy_curves(metrics_df, fig_dir / "acc_curves.png")
    plot_overfit_zone(metrics_df, fig_dir / "overfit_zone.png")
    plot_lr_schedule(metrics_df, fig_dir / "lr_schedule.png")
    plot_epoch_time(metrics_df, fig_dir / "epoch_time.png")

    det = detect_overfitting(metrics_df, window=window)

    mode = (
        (run_cfg.get("model") or {}).get("mode")
        or run_cfg.get("mode")
        or "unknown"
    )

    df = metrics_df.sort_values("epoch").reset_index(drop=True)
    last = df.iloc[-1]
    total_time = float(df["time_sec"].sum()) if "time_sec" in df.columns else float("nan")

    lines = []
    lines.append(f"# Отчёт по run: `{run_id}`")
    lines.append("")
    lines.append(f"- Режим модели: `{mode}`")
    lines.append(f"- Количество эпох: `{len(df)}`")
    lines.append(f"- Best epoch: `{det['best_epoch']}`")
    lines.append(f"- Best val_loss: `{det['best_val_loss']:.4f}`")
    lines.append("")

    lines.append("## Финальные метрики")
    lines.append("")
    lines.append("| Метрика | Значение |")
    lines.append("|---|---:|")
    if "val_acc" in df.columns:
        lines.append(f"| val_acc | {float(last['val_acc']):.4f} |")
    if "val_loss" in df.columns:
        lines.append(f"| val_loss | {float(last['val_loss']):.4f} |")
    if "train_acc" in df.columns:
        lines.append(f"| train_acc | {float(last['train_acc']):.4f} |")
    if "train_loss" in df.columns:
        lines.append(f"| train_loss | {float(last['train_loss']):.4f} |")
    lines.append(f"| Время обучения, сек | {total_time:.2f} |")
    lines.append(f"| Время обучения, мин | {total_time / 60.0:.2f} |")
    lines.append("")

    lines.append("## Интерпретация")
    lines.append("")
    lines.append(det["interpretation"])
    lines.append("")
    lines.append(
        f"- `best_epoch`: `{det['best_epoch']}`\n"
        f"- `overfit_start_epoch`: `{det['overfit_start_epoch']}`\n"
        f"- `underfit_until_epoch`: `{det['underfit_until_epoch']}`"
    )
    lines.append("")

    lines.append("## Графики")
    lines.append("")
    lines.append("![Loss curves](figures/loss_curves.png)")
    lines.append("")
    lines.append("![Accuracy curves](figures/acc_curves.png)")
    lines.append("")
    lines.append("![Overfit zone](figures/overfit_zone.png)")
    lines.append("")
    lines.append("![LR schedule](figures/lr_schedule.png)")
    lines.append("")
    lines.append("![Epoch time](figures/epoch_time.png)")
    lines.append("")

    if y_true is not None and y_pred is not None:
        classes = (
            (run_cfg.get("data") or {}).get("classes")
            or run_cfg.get("classes")
        )
        if not classes:
            classes = [str(i) for i in sorted(set(y_true) | set(y_pred))]
        plot_confusion_matrix(y_true, y_pred, fig_dir / "confusion_matrix.png", classes)
        plot_per_class_metrics(y_true, y_pred, fig_dir / "per_class_metrics.png")
        lines.append("## Test-метрики")
        lines.append("")
        lines.append("![Confusion matrix](figures/confusion_matrix.png)")
        lines.append("")
        lines.append("![Per-class metrics](figures/per_class_metrics.png)")
        lines.append("")

    lines.append("## Вывод")
    lines.append("")
    reason = f"минимум `val_loss={det['best_val_loss']:.4f}` достигнут на эпохе `{det['best_epoch']}`"
    if det["overfit_start_epoch"] is not None:
        reason += f"; после эпохи `{det['overfit_start_epoch']}` наблюдается устойчивое переобучение"
    lines.append(f"Рекомендуется использовать `model_best.pt` из эпохи `{det['best_epoch']}`: {reason}.")
    lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
