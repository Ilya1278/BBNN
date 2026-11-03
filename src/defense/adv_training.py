"""Adversarial Training: обучение на смеси чистых и FGSM-примеров."""
import csv
import os
import time

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.attacks.fgsm import fgsm_attack
from src.defense._classifier_loader import load_classifier
from src.data.augment import get_train_transform, get_val_transform
from src.data.dataset import MotorcycleDataset


_WEIGHT_KEYS = ("model_state", "state_dict", "model_state_dict", "weights")


def _assert_no_overwrite(out_dir, original_checkpoint):
    if not original_checkpoint:
        return
    out_abs = os.path.abspath(out_dir)
    ckpt_abs = os.path.abspath(original_checkpoint)
    ckpt_dir = os.path.dirname(ckpt_abs)
    try:
        common = os.path.commonpath([out_abs, ckpt_dir])
    except ValueError:
        return
    if common == ckpt_dir:
        raise RuntimeError(
            f"REFUSING to write adv_trained checkpoint into {out_abs}: "
            f"it collides with original checkpoint dir {ckpt_dir}. "
            f"Set defense.adv_training.out_dir to a separate folder (runs/adv_training/)."
        )


def _detect_ckpt_format(path):
    if not path or not os.path.exists(path):
        return None
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for k in _WEIGHT_KEYS:
            if k in obj and isinstance(obj[k], dict):
                return k
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return "raw"
    return None


def _save_ckpt(save_path, state_dict, meta, original_path):
    fmt = _detect_ckpt_format(original_path)
    if fmt in (None, "raw"):
        torch.save(state_dict, save_path)
    else:
        torch.save({fmt: state_dict, **meta}, save_path)


def _load_model_cfg(cfg):
    with open(cfg["model_config"], "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _eval_acc(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def _eval_adv_acc(model, loader, device, eps, eps_space):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.no_grad():
            x_req = x.detach().clone().requires_grad_(True)
        x_adv = fgsm_attack(model, x_req, y, eps=eps, eps_space=eps_space).detach()
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def _try_dataset(manifest_path, root, split, transform):
    try:
        return MotorcycleDataset(
            manifest_path=manifest_path, split=split, transform=transform, root=root
        )
    except (ValueError, TypeError):
        return None


def train_with_adv_examples(
    cfg,
    train_manifest,
    val_manifest,
    out_dir,
    eps=0.03,
    adv_ratio=0.5,
    epochs=30,
    lr=5e-4,
    eps_space="normalized",
    device=None,
    seed=42,
    val_every=3,
    val_adv_every=6,
    num_workers_override=None,
    use_amp=True,
):
    original_ckpt = cfg.get("checkpoint")
    _assert_no_overwrite(out_dir, original_ckpt)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "model_adv_trained.pt")
    assert original_ckpt is None or os.path.abspath(save_path) != os.path.abspath(original_ckpt)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    clf = load_classifier(checkpoint=original_ckpt, model_config=cfg["model_config"], device=device)
    model = clf.model

    model_cfg = _load_model_cfg(cfg)
    train_tf = get_train_transform(model_cfg)
    val_tf = get_val_transform(model_cfg)

    bs = cfg["data"]["batch_size"]
    nw = num_workers_override if num_workers_override is not None else cfg["data"]["num_workers"]
    root = cfg["data"]["processed_dir"]

    train_ds = _try_dataset(train_manifest, root, "train", train_tf)
    if train_ds is None:
        raise RuntimeError(f"MotorcycleDataset(split='train') пуст для {train_manifest}")
    val_ds = _try_dataset(val_manifest, root, "val", val_tf)
    if val_ds is None:
        print("WARNING: split='val' отсутствует, использую 'test' для model selection")
        val_ds = _try_dataset(val_manifest, root, "test", val_tf)
    if val_ds is None:
        raise RuntimeError(f"Не нашёл ни val-, ни test-сплит в {val_manifest}")

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True,
        persistent_workers=(nw > 0),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    amp_enabled = use_amp and device != "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    print(f"[adv] device={device} amp={amp_enabled} nw={nw} "
          f"epochs={epochs} val_every={val_every} val_adv_every={val_adv_every}",
          flush=True)

    best_val, rows = -1.0, []
    best_adv_at_best = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        te = time.time()
        model.train()
        run_loss, nb = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # FGSM в fp32 (стабильность), без autocast
            x_req = x.detach().clone().requires_grad_(True)
            x_adv = fgsm_attack(model, x_req, y, eps=eps, eps_space=eps_space).detach()

            n = x.size(0)
            n_adv = int(round(n * adv_ratio))
            if 0 < n_adv < n:
                perm = torch.randperm(n, device=device)
                a, c = perm[:n_adv], perm[n_adv:]
                x_mix = torch.cat([x_adv[a], x[c]], 0)
                y_mix = torch.cat([y[a], y[c]], 0)
            elif n_adv >= n:
                x_mix, y_mix = x_adv, y
            else:
                x_mix, y_mix = x.detach(), y

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(x_mix)
                loss = criterion(logits, y_mix)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            run_loss += loss.item()
            nb += 1
        scheduler.step()

        train_loss = run_loss / max(nb, 1)
        t_train = time.time() - te

        # ---- val (sparse) ----
        val_clean = None
        val_adv = None
        last_epoch = epoch == epochs
        t_val = 0.0
        if epoch % val_every == 0 or last_epoch:
            tv = time.time()
            val_clean = _eval_acc(model, val_loader, device)
            if epoch % val_adv_every == 0 or last_epoch:
                val_adv = _eval_adv_acc(model, val_loader, device, eps, eps_space)
            t_val = time.time() - tv

        dt = time.time() - t0
        elapsed_epoch = time.time() - te
        eta = (epochs - epoch) * elapsed_epoch
        rows.append(dict(
            epoch=epoch, train_loss=train_loss,
            val_clean_acc=val_clean if val_clean is not None else float("nan"),
            val_adv_acc=val_adv if val_adv is not None else float("nan"),
            time_sec=dt,
        ))

        msg = (f"[adv] {epoch}/{epochs} loss={train_loss:.4f} "
               f"train={t_train:.0f}s val={t_val:.0f}s total={dt:.0f}s eta={eta:.0f}s")
        if val_clean is not None:
            msg += f" clean={val_clean:.4f}"
        if val_adv is not None:
            msg += f" adv={val_adv:.4f}"
        print(msg, flush=True)

        if val_clean is not None and val_clean > best_val:
            best_val = val_clean
            best_adv_at_best = val_adv if val_adv is not None else best_adv_at_best
            meta = {
                "epoch": epoch,
                "val_clean_acc": val_clean,
                "val_adv_acc": val_adv,
                "config": cfg,
                "adv_training": {
                    "eps": eps, "adv_ratio": adv_ratio,
                    "epochs": epochs, "lr": lr,
                },
            }
            _save_ckpt(save_path, model.state_dict(), meta, original_ckpt)

    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    return {
        "checkpoint": save_path,
        "best_val_clean_acc": best_val,
        "best_val_adv_acc": best_adv_at_best,
        "epochs": epochs,
        "elapsed_sec": time.time() - t0,
        "eps": eps,
        "adv_ratio": adv_ratio,
        "out_dir": os.path.abspath(out_dir),
    }