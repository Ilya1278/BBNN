#!/usr/bin/env python
"""
CLI для FGSM-атаки на MotorcycleClassifier.

Примеры:
  python scripts/run_attack.py --config configs/attack_fgsm.yaml --eps 0.03
  python scripts/run_attack.py --config configs/attack_fgsm.yaml \
          --eps-sweep 0.005 0.01 0.02 0.03 0.05 0.1
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.augment import get_val_transform
from src.data.dataset import MotorcycleDataset
from src.models.classifier import MotorcycleClassifier
from src.attacks.fgsm import evaluate_attack, evaluate_sweep, save_success_panels


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_loader(attack_cfg, cfg_train, split, batch_size, num_workers):
    transform = get_val_transform(cfg_train)   # albumentations, val
    ds = MotorcycleDataset(
        manifest_path=attack_cfg["data"]["manifest"],
        split=split,
        transform=transform,
        root=attack_cfg["data"]["processed_dir"],
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--eps-sweep", nargs="+", type=float, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-panels", action="store_true")
    args = ap.parse_args()

    if (args.eps is None) == (args.eps_sweep is None):
        ap.error("укажи ровно одно из --eps / --eps-sweep")

    attack_cfg = load_yaml(args.config)
    cfg_train  = load_yaml(attack_cfg["model_config"])

    set_seed(attack_cfg.get("seed", 42))

    output_dir = Path(attack_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    eps_space = attack_cfg.get("attack", {}).get("eps_space", "normalized")
    std       = cfg_train["augment"]["imagenet_std"]

    print(f"[attack] ckpt       = {attack_cfg['checkpoint']}")
    print(f"[attack] cfg_train  = {attack_cfg['model_config']}")
    print(f"[attack] manifest   = {attack_cfg['data']['manifest']} "
          f"(split={attack_cfg['data']['split']})")
    print(f"[attack] eps_space  = {eps_space}")

    classifier = MotorcycleClassifier(
        checkpoint_path=attack_cfg["checkpoint"],
        cfg=cfg_train,
        device=args.device,
    )
    print(f"[attack] model_mode = {classifier.model_mode}")
    print(f"[attack] device     = {classifier.device}")

    loader = build_loader(
        attack_cfg, cfg_train,
        split=attack_cfg["data"]["split"],
        batch_size=attack_cfg["data"]["batch_size"],
        num_workers=attack_cfg["data"]["num_workers"],
    )

    # --- sweep ---
    if args.eps_sweep is not None:
        results = evaluate_sweep(classifier, loader, args.eps_sweep,
                                 eps_space=eps_space, std=std)
        out_path = output_dir / "attack_metrics_sweep.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[attack] wrote {out_path}")
        # быстрая сводка «опасного» порога
        dangerous = [m["eps"] for m in results if m["asr"] >= 0.5]
        if dangerous:
            print(f"[attack] ASR>=0.5 начиная с eps={min(dangerous)}")
        else:
            print(f"[attack] ASR<0.5 во всём диапазоне {args.eps_sweep}")
        return

    # --- single ---
    eps = args.eps
    metrics = evaluate_attack(classifier, loader, eps,
                              eps_space=eps_space, std=std)
    out_path = output_dir / "attack_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[attack] wrote {out_path}")
    print(f"[attack] ASR={metrics['asr']:.4f}  "
          f"clean={metrics['clean_acc']:.4f}  adv={metrics['adv_acc']:.4f}")

    if not args.no_panels:
        examples_dir = output_dir / "attack_examples"
        saved = save_success_panels(
            classifier=classifier, loader=loader, eps=eps,
            out_dir=examples_dir, cfg_train=cfg_train,
            eps_space=eps_space,
        )
        for p in saved:
            print(f"[attack] panel: {p}")


if __name__ == "__main__":
    main()