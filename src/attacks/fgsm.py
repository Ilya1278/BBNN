from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["not_motorcycle", "motorcycle"]
# ---------------------------------------------------------------- FGSM
def _eps_to_normalized(eps: float, eps_space: str, std, device, dtype):
    """Вернуть per-channel eps-тензор формы (1,3,1,1) в норм. пространстве."""
    if eps_space == "normalized":
        return torch.full((1, 3, 1, 1), float(eps), device=device, dtype=dtype)
    elif eps_space == "pixel":
        std_t = torch.as_tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1)
        return torch.full((1, 3, 1, 1), float(eps), device=device, dtype=dtype) / std_t
    else:
        raise ValueError(f"unknown eps_space={eps_space!r}")


def fgsm_attack(model, images, labels, eps, criterion=None,
                eps_space="normalized", std=None):

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    eps_t = _eps_to_normalized(eps, eps_space, std, images.device, images.dtype)

    x = images.detach().clone().requires_grad_(True)
    logits = model(x)
    loss = criterion(logits, labels)

    model.zero_grad(set_to_none=True)
    loss.backward()

    with torch.no_grad():
        x_adv = x + eps_t * x.grad.sign()
    return x_adv.detach()

# ---------------------------------------------------------------- evaluation

def evaluate_attack(classifier, loader, eps,
                    eps_space="normalized", std=None,
                    criterion=None):

    model = classifier.model
    device = classifier.device
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    n_total = 0
    n_clean_correct = 0
    n_adv_correct = 0
    n_adv_success = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            clean_preds = model(images).argmax(1)
        correct_mask = clean_preds.eq(labels)

        if eps > 0:
            x_adv = fgsm_attack(model, images, labels, eps, criterion,
                                eps_space=eps_space, std=std)
        else:
            x_adv = images

        with torch.no_grad():
            adv_preds = model(x_adv).argmax(1)

        n_total        += labels.numel()
        n_clean_correct += int(correct_mask.sum().item())
        n_adv_correct   += int(adv_preds.eq(labels).sum().item())
        n_adv_success   += int((correct_mask & adv_preds.ne(labels)).sum().item())

    clean_acc = n_clean_correct / max(n_total, 1)
    adv_acc   = n_adv_correct   / max(n_total, 1)
    asr       = n_adv_success   / max(n_clean_correct, 1)

    return {
        "attack": "fgsm",
        "eps": float(eps),
        "eps_space": eps_space,
        "asr": float(asr),
        "n_samples": int(n_total),
        "n_clean_correct": int(n_clean_correct),
        "n_adv_success": int(n_adv_success),
        "clean_acc": float(clean_acc),
        "adv_acc": float(adv_acc),
    }

def evaluate_sweep(classifier, loader, eps_list,
                   eps_space="normalized", std=None, progress=True):
    out = []
    for eps in eps_list:
        m = evaluate_attack(classifier, loader, eps,
                            eps_space=eps_space, std=std)
        if progress:
            print(f"[attack] eps={eps:<6}  asr={m['asr']:.4f}  "
                  f"clean={m['clean_acc']:.4f}  adv={m['adv_acc']:.4f}")
        out.append(m)
    return out

# ---------------------------------------------------------------- panels

def _unnormalize(img_tensor, mean, std):
    """(3,H,W) норм. тензор -> HWC float [0,1]."""
    m = torch.as_tensor(mean, dtype=img_tensor.dtype).view(3, 1, 1)
    s = torch.as_tensor(std,  dtype=img_tensor.dtype).view(3, 1, 1)
    x = img_tensor.detach().cpu() * s + m
    return x.clamp(0, 1).permute(1, 2, 0).numpy()

def _save_panel(img_clean, img_adv, y_true, y_clean, y_adv,
                mean, std, out_path: Path):
    x_clean = _unnormalize(img_clean, mean, std)
    x_adv   = _unnormalize(img_adv,   mean, std)

    # perturbation в норм. пространстве, растянутая для видимости
    delta = (img_adv - img_clean).detach().cpu()
    delta_vis = (delta * 4.0 * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.5))
    ax[0].imshow(x_clean)
    ax[0].set_title(f"clean\ntrue={CLASSES[y_true]}\npred={CLASSES[y_clean]}")
    ax[1].imshow(x_adv)
    ax[1].set_title(f"adv (FGSM)\npred={CLASSES[y_adv]}")
    ax[2].imshow(delta_vis)
    ax[2].set_title("perturbation (norm-space ×4)")
    ax[3].imshow(np.abs(x_adv - x_clean) * 8.0)
    ax[3].set_title("|adv − clean| ×8")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)

def save_success_panels(classifier, loader, eps, out_dir, cfg_train,
                        eps_space="normalized", max_per_class=1):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean = cfg_train["augment"]["imagenet_mean"]
    std  = cfg_train["augment"]["imagenet_std"]

    model = classifier.model
    device = classifier.device
    model.eval()
    criterion = nn.CrossEntropyLoss()

    saved = []
    counters = {"moto": 0, "nonmoto": 0, "failure": 0}
    done = lambda: (counters["moto"]    >= max_per_class and
                    counters["nonmoto"] >= max_per_class and
                    counters["failure"] >= 1)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # 1) clean forward — под no_grad
        with torch.no_grad():
            clean_preds = model(images).argmax(1)
        correct_mask = clean_preds.eq(labels)

        # 2) FGSM — ВНЕ no_grad, иначе backward не построит граф
        x_adv = fgsm_attack(model, images, labels, eps, criterion,
                            eps_space=eps_space, std=std)

        # 3) adv forward — снова можно под no_grad
        with torch.no_grad():
            adv_preds = model(x_adv).argmax(1)

        success = correct_mask & adv_preds.ne(labels)
        fail    = correct_mask & adv_preds.eq(labels)

        if counters["moto"] < max_per_class:
            sel = success & labels.eq(1)
            if sel.any():
                i = int(sel.nonzero(as_tuple=True)[0][0])
                saved.append(_save_panel(
                    images[i], x_adv[i],
                    labels[i].item(), clean_preds[i].item(), adv_preds[i].item(),
                    mean, std,
                    out_dir / f"moto_success_{counters['moto']+1:02d}.png"))
                counters["moto"] += 1

        if counters["nonmoto"] < max_per_class:
            sel = success & labels.eq(0)
            if sel.any():
                i = int(sel.nonzero(as_tuple=True)[0][0])
                saved.append(_save_panel(
                    images[i], x_adv[i],
                    labels[i].item(), clean_preds[i].item(), adv_preds[i].item(),
                    mean, std,
                    out_dir / f"nonmoto_success_{counters['nonmoto']+1:02d}.png"))
                counters["nonmoto"] += 1

        if counters["failure"] < 1 and fail.any():
            i = int(fail.nonzero(as_tuple=True)[0][0])
            saved.append(_save_panel(
                images[i], x_adv[i],
                labels[i].item(), clean_preds[i].item(), adv_preds[i].item(),
                mean, std, out_dir / "failure_01.png"))
            counters["failure"] += 1

        if done():
            break

    return saved