"""Entry point обучения.

    python scripts/train.py --config configs/train_scratch.yaml

Data Agent владеет src/data/augment.py — здесь только импорт его публичных
функций get_train_transform(cfg) / get_val_transform(cfg).
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- Data Agent ---
from src.data.dataset import MotorcycleDataset  # noqa: E402


def _resolve_transforms_factory(cfg: dict):
    """
    Возвращает (train_tf, val_tf, source_name).
    Data Agent: src/data/augment.py, сигнатура get_*_transform(cfg).
    Локальный фолбэк на torchvision — только если модуль не импортируется.
    """
    image_size = int(cfg.get("data", {}).get("image_size", 224))

    try:
        import src.data.augment as aug  # type: ignore
    except Exception:  # noqa: BLE001
        aug = None

    def _pick(*names):
        if aug is None:
            return None
        for n in names:
            fn = getattr(aug, n, None)
            if callable(fn):
                return fn
        return None

    train_fn = _pick("get_train_transform", "get_train_transforms")
    val_fn = _pick("get_val_transform", "get_val_transforms")

    if train_fn is not None and val_fn is not None:
        # Data Agent ожидает cfg целиком (см. cfg["augment"][...])
        return train_fn(cfg), val_fn(cfg), "src.data.augment"

    # локальный фолбэк (в src/data/* не лезем)
    from torchvision import transforms

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf, "scripts/train.py (local fallback)"


from src.models.factory import build_model  # noqa: E402
from src.train.logger import RunLogger  # noqa: E402
from src.train.loop import train_one_epoch, validate  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_dataset(manifest_path, split, transform, processed_dir):
    """
    MotorcycleDataset(manifest_path, split, transform=None, root=None).
    root — абсолютный префикс, приклеивается к относительному filepath.
    """
    root = None
    if processed_dir:
        root = str(Path(processed_dir).resolve())
    return MotorcycleDataset(manifest_path, split, transform=transform, root=root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["train"].get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = cfg["model"]["mode"]
    run_name = cfg.get("paths", {}).get("run_name", mode)
    print(f"[BBNN] mode={mode} run_name={run_name} device={device} seed={seed}")

    # ---------- logger + снапшот ДО старта ----------
    runs_dir = cfg.get("paths", {}).get("runs_dir", "runs")
    ckpt_dir = Path(cfg.get("paths", {}).get("checkpoints_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(cfg, runs_dir=runs_dir, run_name=run_name)
    print(f"[BBNN] run_id={logger.run_id} -> {logger.run_dir}")

    manifest_source = cfg["data"]["manifest"]
    snapshot_path = logger.snapshot_manifest(manifest_source)
    print(f"[BBNN] manifest snapshot: {snapshot_path}  (source: {manifest_source})")

    # --- sanity check: filepath из снапшота + root существует ---
    import json as _json
    _root = Path(cfg["data"]["processed_dir"]).resolve()
    _records = _json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    _missing = [
        r["filepath"] for r in _records
        if not (_root / r["filepath"]).exists()
    ]
    if _missing:
        print(f"[BBNN] FATAL: {len(_missing)} файлов не найдены под {_root}. Примеры:")
        for _m in _missing[:5]:
            print(f"         {_m}")
        sys.exit(1)
    print(f"[BBNN] snapshot sanity: all {len(_records)} paths exist under {_root}")

    # ---------- data: читаем СНАПШОТ ----------
    bs = int(cfg["data"].get("batch_size", 64))
    nw = int(cfg["data"].get("num_workers", 4))
    processed_dir = cfg["data"].get("processed_dir")

    train_tf, val_tf, tf_src = _resolve_transforms_factory(cfg)
    print(f"[BBNN] transforms from: {tf_src}")

    train_ds = _build_dataset(str(snapshot_path), "train", train_tf, processed_dir)
    val_ds = _build_dataset(str(snapshot_path), "val", val_tf, processed_dir)
    print(f"[BBNN] train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw,
        pin_memory=True, drop_last=False, persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw,
        pin_memory=True, drop_last=False, persistent_workers=nw > 0,
    )

    # ---------- model / optim ----------
    num_classes = int(cfg["model"].get("num_classes", 2))
    model = build_model(cfg, num_classes=num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[BBNN] params={n_params/1e6:.2f}M")

    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=float(cfg["train"].get("label_smoothing", 0.1))
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    epochs = int(cfg["train"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_ckpt_run = logger.run_dir / "model_best.pt"
    best_ckpt_global = ckpt_dir / f"{logger.run_id}_best.pt"

    patience = int(cfg["train"].get("early_stopping_patience", 10))
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0

    # ---------- training ----------
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, amp
        )
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device, amp
        )

        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()
        dt = time.time() - t0

        logger.log_epoch(epoch, train_loss, val_loss, train_acc, val_acc, lr_now, dt)
        print(
            f"[{epoch:03d}/{epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"lr={lr_now:.6f} time={dt:.1f}s"
        )

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            state = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "mode": mode,
                "run_name": run_name,
                "config": cfg,
            }
            torch.save(state, best_ckpt_run)
            torch.save(state, best_ckpt_global)
            print(f"  ↳ saved best (val_loss={val_loss:.4f}) @ epoch {epoch}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[BBNN] early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"[BBNN] done. best_epoch={best_epoch} best_val_loss={best_val:.4f}")
    print(f"[BBNN] best checkpoint: {best_ckpt_run}")
    print(f"[BBNN] manifest snapshot: {snapshot_path}")


if __name__ == "__main__":
    main()