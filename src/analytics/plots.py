from __future__ import annotations

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
