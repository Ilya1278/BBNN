"""Один шаг train / validate с AMP."""
from __future__ import annotations

import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, amp: bool = True):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=amp):
            logits = model(images)
            loss = criterion(logits, targets)

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += bs
        pbar.set_postfix(loss=f"{total_loss / max(total, 1):.4f}")

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, amp: bool = True):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc="val", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=amp):
            logits = model(images)
            loss = criterion(logits, targets)

        bs = images.size(0)
        total_loss += loss.item() * bs
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += bs

        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())

    y_pred = torch.cat(all_preds).numpy() if all_preds else torch.empty(0, dtype=torch.long).numpy()
    y_true = torch.cat(all_targets).numpy() if all_targets else torch.empty(0, dtype=torch.long).numpy()
    return total_loss / max(total, 1), correct / max(total, 1), y_true, y_pred