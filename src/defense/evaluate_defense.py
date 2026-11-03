"""Оценка защиты: ASR до и после применения defense_fn."""
import torch
from src.attacks.fgsm import fgsm_attack


def evaluate_defense(
    classifier,
    loader,
    defense_fn,
    eps: float,
    eps_space: str = "normalized",
    defense_name: str = "none",
) -> dict:
    model, device = classifier.model, classifier.device
    model.eval()
    total = clean_correct = adv_correct_before = adv_correct_after = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred_clean = model(x).argmax(1)
        # FGSM на лету
        x_req = x.detach().clone().requires_grad_(True)
        x_adv = fgsm_attack(model, x_req, y, eps=eps, eps_space=eps_space).detach()
        with torch.no_grad():
            pred_adv = model(x_adv).argmax(1)
            x_def = defense_fn(x_adv) if defense_fn is not None else x_adv
            pred_def = model(x_def).argmax(1)

        mask = pred_clean == y                      # «уязвимые» = корректно классифицированные чистые
        total += y.numel()
        clean_correct += (pred_clean == y).sum().item()
        adv_correct_before += ((pred_adv == y) & mask).sum().item()
        adv_correct_after += ((pred_def == y) & mask).sum().item()

    clean_acc = clean_correct / max(total, 1)
    asr_before = 1.0 - adv_correct_before / max(clean_correct, 1)
    asr_after = 1.0 - adv_correct_after / max(clean_correct, 1)
    return {
        "defense": defense_name,
        "eps": float(eps),
        "clean_acc": clean_acc,
        "adv_acc_before_defense": adv_correct_before / max(total, 1),
        "adv_acc_after_defense": adv_correct_after / max(total, 1),
        "asr_before": asr_before,
        "asr_after": asr_after,
        "asr_reduction": asr_before - asr_after,
    }