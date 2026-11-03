# bootstrap_analytics.py — создаёт структуру Analytics Agent
from pathlib import Path

FILES = {}

FILES["src/analytics/__init__.py"] = '"""Analytics Agent for BBNN."""\n'

FILES["src/analytics/plots.py"] = '''from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


DEFAULT_DPI = 150
DEFAULT_FIGSIZE = (10, 6)


def _ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _finish(fig, out_path):
    _ensure_parent(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=plt.rcParams.get("savefig.dpi", DEFAULT_DPI), bbox_inches="tight")
    plt.close(fig)


def _best_epoch(df):
    idx = df["val_loss"].idxmin()
    return int(df.loc[idx, "epoch"])


def plot_loss_curves(metrics_df, out_path):
    df = metrics_df.sort_values("epoch").reset_index(drop=True)
    best_epoch = _best_epoch(df)
    fig, ax = plt.subplots(figsize=plt.rcParams.get("figure.figsize", DEFAULT_FIGSIZE))
    ax.plot(df["epoch"], df["train_loss"], marker="o", linewidth=2, label="train_loss")
    ax.plot(df["epoch"], df["val_loss"], marker="s", linewidth=2, label="val_loss")
    ax.axvline(best_epoch, color="green", linestyle="--", linewidth=1.5, label=f"best epoch = {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss curves")
    ax.legend()
    _finish(fig, out_path)


def plot_accuracy_curves(metrics_df, out_path):
    df = metrics_df.sort_values("epoch").reset_index(drop=True)
    best_epoch = _best_epoch(df)
    fig, ax = plt.subplots(figsize=plt.rcParams.get("figure.figsize", DEFAULT_FIGSIZE))
    ax.plot(df["epoch"], df["train_acc"], marker="o", linewidth=2, label="train_acc")
    ax.plot(df["epoch"], df["val_acc"], marker="s", linewidth=2, label="val_acc")
    ax.axvline(best_epoch, color="green", linestyle="--", linewidth=1.5, label=f"best epoch = {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy curves")
    ax.legend()
    _finish(fig, out_path)


def plot_overfit_zone(metrics_df, out_path):
    from .overfit_detector import detect_overfitting
    df = metrics_df.sort_values("epoch").reset_index(drop=True)
    best_epoch = _best_epoch(df)
    det = detect_overfitting(df, window=5)
    overfit_start = det.get("overfit_start_epoch")
    fig, ax = plt.subplots(figsize=plt.rcParams.get("figure.figsize", DEFAULT_FIGSIZE))
    ax.plot(df["epoch"], df["train_loss"], marker="o", linewidth=2, label="train_loss")
    ax.plot(df["epoch"], df["val_loss"], marker="s", linewidth=2, label="val_loss")
    ax.axvline(best_epoch, color="green", linestyle="--", linewidth=1.5, label=f"best epoch = {best_epoch}")
    if overfit_start is not None:
        ax.axvspan(overfit_start, df["epoch"].max(), color="red", alpha=0.15, label="overfit zone")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Overfitting zone")
    ax.legend()
    _finish(fig, out_path)


def plot_lr_schedule(metrics_df, out_path):
    df = metrics_df.sort_values("epoch").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=plt.rcParams.get("figure.figsize", DEFAULT_FIGSIZE))
    ax.plot(df["epoch"], df["lr"], marker="o", linewidth=2, label="lr")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("LR schedule")
    ax.legend()
    _finish(fig, out_path)


def plot_confusion_matrix(y_true, y_pred, out_path, classes):
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=plt.rcParams.get("figure.figsize", DEFAULT_FIGSIZE))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    _finish(fig, out_path)


def plot_per_class_metrics(y_true, y_pred, out_path):
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=plt.rcParams.get("figure.figsize", DEFAULT_FIGSIZE))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in labels])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-class metrics")
    ax.legend()
    _finish(fig, out_path)


def plot_epoch_time(metrics_df, out_path):
    df = metrics_df.sort_values("epoch").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=plt.rcParams.get("figure.figsize", DEFAULT_FIGSIZE))
    ax.bar(df["epoch"], df["time_sec"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time, sec")
    ax.set_title("Epoch time")
    _finish(fig, out_path)
'''

FILES["src/analytics/overfit_detector.py"] = '''from __future__ import annotations

import numpy as np
import pandas as pd


def detect_overfitting(metrics_df, window=5, slope_eps=1e-4):
    """Slope-based детектор переобучения. Возвращает dict со словами
    best_epoch, best_val_loss, overfit_start_epoch, underfit_until_epoch,
    interpretation.
    """
    df = metrics_df.sort_values("epoch").reset_index(drop=True)

    if len(df) == 0:
        return {
            "best_epoch": -1, "best_val_loss": float("nan"),
            "overfit_start_epoch": None, "underfit_until_epoch": None,
            "interpretation": "Пустой metrics.csv: анализировать нечего.",
        }

    best_idx = df["val_loss"].idxmin()
    best_epoch = int(df.loc[best_idx, "epoch"])
    best_val_loss = float(df.loc[best_idx, "val_loss"])

    val = df["val_loss"].to_numpy(dtype=float)
    train = df["train_loss"].to_numpy(dtype=float)
    epochs = df["epoch"].to_numpy(dtype=int)

    overfit_start_epoch = None
    if len(df) >= window:
        x = np.arange(window, dtype=float)
        for i in range(window - 1, len(df)):
            v = val[i - window + 1 : i + 1]
            t = train[i - window + 1 : i + 1]
            v_slope = np.polyfit(x, v, 1)[0]
            t_slope = np.polyfit(x, t, 1)[0]
            val_growing = v_slope > slope_eps and (v[-1] - v[0]) > 0
            train_falling = t_slope < -slope_eps and (t[-1] - t[0]) < 0
            if val_growing and train_falling:
                overfit_start_epoch = int(epochs[i - window + 1])
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
            f"положительного тренда на окне {window} эпох при падающем train_loss."
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
'''

FILES["src/analytics/report.py"] = '''from __future__ import annotations

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
        f"- `best_epoch`: `{det['best_epoch']}`\\n"
        f"- `overfit_start_epoch`: `{det['overfit_start_epoch']}`\\n"
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

    (out_dir / "summary.md").write_text("\\n".join(lines), encoding="utf-8")
'''

FILES["src/analytics/selftest.py"] = '''from __future__ import annotations

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
'''

FILES["scripts/analyze.py"] = '''from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analytics.report import generate_report


def _run_test_eval(run_dir):
    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
        from src.models.classifier import MotorcycleClassifier
    except Exception as exc:
        raise RuntimeError("classifier not ready, run Model Agent first") from exc

    import yaml

    cfg_path = ROOT / "configs" / "analytics.yaml"
    cfg = {}
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    test_dir = ROOT / cfg.get("test_dir", "data/processed/test")
    if not test_dir.exists():
        raise RuntimeError(f"test dir not found: {test_dir}. classifier not ready, run Model Agent first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(str(test_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=int(cfg.get("batch_size", 32)), shuffle=False, num_workers=0)

    num_classes = len(dataset.classes)
    try:
        model = MotorcycleClassifier(num_classes=num_classes)
    except TypeError:
        model = MotorcycleClassifier()

    ckpt_path = run_dir / "model_best.pt"
    if not ckpt_path.exists():
        raise RuntimeError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    else:
        state = ckpt

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(y.numpy().tolist())
    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description="Analytics Agent for BBNN")
    parser.add_argument("--run", required=True, help="Path to runs/{run_id}")
    parser.add_argument("--eval-test", action="store_true", help="Eval on test set")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run)
    run_id = run_dir.name
    out_dir = Path(args.out) if args.out else ROOT / "reports" / run_id

    y_true = y_pred = None
    if args.eval_test:
        try:
            y_true, y_pred = _run_test_eval(run_dir)
        except RuntimeError as exc:
            print(f"[--eval-test] {exc}", file=sys.stderr)
            print("[--eval-test] Пропускаю confusion matrix, остальной отчёт будет построен.")
        except Exception as exc:
            print(f"[--eval-test] Ошибка: {exc}", file=sys.stderr)
            print("[--eval-test] Пропускаю confusion matrix, остальной отчёт будет построен.")

    generate_report(run_dir, out_dir, y_true=y_true, y_pred=y_pred)
    print(f"Report written to: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
'''

FILES["configs/analytics.yaml"] = """window: 5
dpi: 150
figsize: [10, 6]
style: seaborn-v0_8-whitegrid
test_dir: data/processed/test
batch_size: 32
num_classes: 2
"""


def main():
    root = Path(__file__).resolve().parent
    for rel, content in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        print("wrote", rel)
    print("done")


if __name__ == "__main__":
    main()