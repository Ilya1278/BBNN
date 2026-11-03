"""Entry point: запуск защит (JPEG, Feature Squeezing, Adversarial Training) и оценка."""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.defense._classifier_loader import load_classifier
from src.data.augment import get_val_transform
from src.data.dataset import MotorcycleDataset
from src.defense.jpeg_compress import jpeg_compress_tensor, DEFAULT_MEAN, DEFAULT_STD
from src.defense.feature_squeeze import feature_squeeze_tensor, _backend_name
from src.defense.adv_training import train_with_adv_examples
from src.defense.evaluate_defense import evaluate_defense


# ---------- helpers ----------

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _results_path(cfg):
    return os.path.join(cfg["output_dir"], "defense_metrics.json")


def _load_existing_results(cfg):
    p = _results_path(cfg)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[resume] failed to read {p}: {e} — starting fresh")
        return {}


def _save_metrics(cfg, results):
    if "meta" not in results:
        results["meta"] = {
            "eps": cfg["attack"]["eps"],
            "eps_space": cfg["attack"]["eps_space"],
            "checkpoint": cfg["checkpoint"],
            "split": cfg["data"]["split"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    with open(_results_path(cfg), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _test_loader(cfg):
    with open(cfg["model_config"], "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    transform = get_val_transform(model_cfg)
    ds = MotorcycleDataset(
        manifest_path=cfg["data"]["manifest"],
        split=cfg["data"]["split"],
        transform=transform,
        root=cfg["data"]["processed_dir"],
    )
    return DataLoader(ds, batch_size=cfg["data"]["batch_size"],
                      shuffle=False, num_workers=cfg["data"]["num_workers"])


# ---------- methods ----------

def _run_jpeg(cfg, clf, loader):
    out = {}
    for q in cfg["defense"]["jpeg"]["qualities"]:
        name = f"jpeg_{q}"
        print(f"[{name}] running...", flush=True)
        fn = lambda x, q=q: jpeg_compress_tensor(x, quality=q)
        out[name] = evaluate_defense(clf, loader, fn, cfg["attack"]["eps"],
                                     cfg["attack"]["eps_space"], defense_name=name)
        print(f"[{name}] ASR {out[name]['asr_before']:.4f} -> {out[name]['asr_after']:.4f}",
              flush=True)
    return out


def _run_squeeze(cfg, clf, loader):
    print(f"[squeeze] median filter backend: {_backend_name()}", flush=True)
    out = {}
    for bd in cfg["defense"]["squeeze"]["bit_depths"]:
        for m in cfg["defense"]["squeeze"]["median_sizes"]:
            name = f"squeeze_{bd}_{m}"
            print(f"[{name}] running...", flush=True)
            fn = lambda x, bd=bd, m=m: feature_squeeze_tensor(
                x, bit_depth=bd, median_filter_size=m)
            out[name] = evaluate_defense(clf, loader, fn, cfg["attack"]["eps"],
                                         cfg["attack"]["eps_space"], defense_name=name)
            print(f"[{name}] ASR {out[name]['asr_before']:.4f} -> {out[name]['asr_after']:.4f}",
                  flush=True)
    return out


def _run_adv_training(cfg):
    out_dir = cfg["defense"]["adv_training"]["out_dir"]
    orig = cfg["checkpoint"]
    hash_before = _sha256(orig) if os.path.exists(orig) else None

    at_cfg = cfg["defense"]["adv_training"]
    res = train_with_adv_examples(
        cfg=cfg,
        train_manifest=cfg["data"]["manifest"],
        val_manifest=cfg["data"]["manifest"],
        out_dir=out_dir,
        eps=at_cfg["eps"],
        adv_ratio=at_cfg["adv_ratio"],
        epochs=at_cfg["epochs"],
        lr=at_cfg["lr"],
        eps_space=cfg["attack"]["eps_space"],
        val_every=at_cfg.get("val_every", 3),
        val_adv_every=at_cfg.get("val_adv_every", 6),
        num_workers_override=at_cfg.get("num_workers"),
        use_amp=at_cfg.get("use_amp", True),
    )
    if hash_before is not None:
        hash_after = _sha256(orig)
        if hash_before != hash_after:
            raise SystemExit(
                f"FATAL: original checkpoint {orig} was modified during adv_training! "
                f"{hash_before[:12]} != {hash_after[:12]}"
            )
        print("ORIGINAL CHECKPOINT UNCHANGED \u2713")
    return res


def _eval_adv_trained(cfg, loader):
    ckpt = os.path.join(cfg["defense"]["adv_training"]["out_dir"], "model_adv_trained.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"Adv-trained checkpoint not found: {ckpt}. "
                         f"Run --method adv_training first.")
    clf = load_classifier(checkpoint=ckpt, model_config=cfg["model_config"])
    return evaluate_defense(clf, loader, None, cfg["attack"]["eps"],
                            cfg["attack"]["eps_space"], defense_name="adv_training")


# ---------- examples ----------

def _denorm_to_pil(x, mean=DEFAULT_MEAN, std=DEFAULT_STD):
    m = np.array(mean).reshape(3, 1, 1)
    s = np.array(std).reshape(3, 1, 1)
    arr = (x.detach().cpu().numpy() * s + m).clip(0, 1)
    arr = (arr.transpose(1, 2, 0) * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def _save_examples(cfg, clf, loader):
    if "jpeg" not in cfg["defense"] or "squeeze" not in cfg["defense"]:
        return
    from src.attacks.fgsm import fgsm_attack
    out = os.path.join(cfg["output_dir"], "defense_examples")
    os.makedirs(out, exist_ok=True)
    eps = cfg["attack"]["eps"]
    q = cfg["defense"]["jpeg"]["qualities"][-1]
    bd = cfg["defense"]["squeeze"]["bit_depths"][-1]
    m = cfg["defense"]["squeeze"]["median_sizes"][-1]
    clf.model.eval()
    for x, y in loader:
        x, y = x.to(clf.device), y.to(clf.device)
        with torch.no_grad():
            pc = clf.model(x).argmax(1)
        x_req = x.detach().clone().requires_grad_(True)
        x_adv = fgsm_attack(clf.model, x_req, y, eps=eps,
                            eps_space=cfg["attack"]["eps_space"]).detach()
        with torch.no_grad():
            pa = clf.model(x_adv).argmax(1)
            xj = jpeg_compress_tensor(x_adv, quality=q)
            xsq = feature_squeeze_tensor(x_adv, bit_depth=bd, median_filter_size=m)
            pj = clf.model(xj).argmax(1)
            psv = clf.model(xsq).argmax(1)
        for i in range(x.size(0)):
            if pc[i] == y[i] and pa[i] != y[i] and pj[i] == y[i] and psv[i] == y[i]:
                _denorm_to_pil(x[i]).save(f"{out}/clean.png")
                _denorm_to_pil(x_adv[i]).save(f"{out}/adversarial.png")
                _denorm_to_pil(xj[i]).save(f"{out}/defended_jpeg.png")
                _denorm_to_pil(xsq[i]).save(f"{out}/defended_squeeze.png")
                print(f"defense_examples saved (class={int(y[i])})")
                return
    print("WARNING: no sample satisfied the showcase criterion")


# ---------- plots ----------

def _fig_dir(cfg):
    d = os.path.join(cfg["output_dir"], "figures")
    os.makedirs(d, exist_ok=True)
    return d


def _bar(path, labels, before, after, title, ylabel="ASR"):
    x = np.arange(len(labels))
    w = 0.38
    plt.figure(figsize=(max(6, len(labels) * 1.3), 4))
    plt.bar(x - w / 2, before, w, label="ASR before", color="#d9534f")
    plt.bar(x + w / 2, after, w, label="ASR after", color="#5cb85c")
    plt.xticks(x, labels, rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _make_plots(cfg, results):
    d = _fig_dir(cfg)
    if results.get("jpeg"):
        keys = list(results["jpeg"].keys())
        _bar(os.path.join(d, "jpeg_comparison.png"),
             [k.replace("jpeg_", "q=") for k in keys],
             [results["jpeg"][k]["asr_before"] for k in keys],
             [results["jpeg"][k]["asr_after"] for k in keys],
             "JPEG defense: ASR before/after")
    if results.get("squeeze"):
        keys = list(results["squeeze"].keys())
        _bar(os.path.join(d, "squeeze_comparison.png"),
             [k.replace("squeeze_", "b") for k in keys],
             [results["squeeze"][k]["asr_before"] for k in keys],
             [results["squeeze"][k]["asr_after"] for k in keys],
             "Feature Squeezing: ASR before/after")
    if results.get("adv_training"):
        at = results["adv_training"]
        _bar(os.path.join(d, "adv_training_comparison.png"), ["adv_training"],
             [at["asr_before"]], [at["asr_after"]],
             "Adversarial Training: ASR before/after")
    # summary — по всем доступным группам
    labels, vals = [], []
    for grp in ("jpeg", "squeeze"):
        for k, v in results.get(grp, {}).items():
            labels.append(k)
            vals.append(v["asr_reduction"])
    if results.get("adv_training"):
        labels.append("adv_training")
        vals.append(results["adv_training"]["asr_reduction"])
    if labels:
        plt.figure(figsize=(max(7, len(labels) * 1.1), 4))
        plt.bar(labels, vals, color="#337ab7")
        plt.ylabel("ASR reduction")
        plt.title("ASR reduction by defense")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(d, "asr_reduction_summary.png"), dpi=150)
        plt.close()


# ---------- report ----------

def _write_report(cfg, results):
    path = os.path.join(cfg["output_dir"], "defense_report.md")
    meta = results.get("meta", {})
    lines = [
        "# Defense report",
        "",
        f"eps = {meta.get('eps', '?')}, eps_space = {meta.get('eps_space', '?')}",
        "",
        "| method | params | clean_acc | adv_acc(before) | adv_acc(after) | ASR before | ASR after | ASR reduction |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def row(name, m):
        return (f"| {name} | {m.get('defense','')} | {m['clean_acc']:.4f} | "
                f"{m['adv_acc_before_defense']:.4f} | {m['adv_acc_after_defense']:.4f} | "
                f"{m['asr_before']:.4f} | {m['asr_after']:.4f} | {m['asr_reduction']:.4f} |")

    if "baseline" in results:
        lines.append(row("baseline", results["baseline"]))
    for _, v in results.get("jpeg", {}).items():
        lines.append(row("JPEG", v))
    for _, v in results.get("squeeze", {}).items():
        lines.append(row("Squeeze", v))
    if "adv_training" in results:
        lines.append(row("AdvTraining", results["adv_training"]))

    if "adv_training" in results:
        at = results["adv_training"]
        lines += [
            "",
            "## Adversarial training",
            "",
            f"- epochs: {at.get('epochs', 30)}",
            f"- время: {at.get('elapsed_sec', 0):.0f} c",
            f"- clean_acc: {at['clean_acc']:.4f}",
            f"- ASR reduction: {at['asr_reduction']:.4f}",
            "",
        ]
    else:
        lines += ["", "_adv_training ещё не посчитан._", ""]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", choices=["jpeg", "squeeze", "adv_training", "all"], default="all")
    ap.add_argument("--evaluate-all", action="store_true")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    cfg.setdefault("output_dir", "reports/defense/")
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # adv_training — отдельный процесс, только обучение
    if args.method == "adv_training":
        res = _run_adv_training(cfg)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    # резюмируем уже посчитанное
    results = _load_existing_results(cfg)
    if results:
        done = [k for k in ("baseline", "jpeg", "squeeze", "adv_training") if k in results]
        print(f"[resume] existing metrics: {done}")

    clf = load_classifier(checkpoint=cfg["checkpoint"], model_config=cfg["model_config"])
    loader = _test_loader(cfg)

    # baseline всегда пересчитываем (быстро, sanity)
    results["baseline"] = evaluate_defense(
        clf, loader, None,
        cfg["attack"]["eps"], cfg["attack"]["eps_space"],
        defense_name="none",
    )
    print(f"[baseline] ASR={results['baseline']['asr_before']:.4f} "
          f"(sanity target ~0.7701)")

    save_path = _results_path(cfg)

    if args.method in ("jpeg", "all"):
        results["jpeg"] = _run_jpeg(cfg, clf, loader)
        _save_metrics(cfg, results)
        _make_plots(cfg, results)
        print(f"[save] {save_path} (jpeg)")

    if args.method in ("squeeze", "all"):
        results["squeeze"] = _run_squeeze(cfg, clf, loader)
        _save_metrics(cfg, results)
        _make_plots(cfg, results)
        print(f"[save] {save_path} (squeeze)")

    if args.evaluate_all:
        if not results.get("jpeg"):
            results["jpeg"] = _run_jpeg(cfg, clf, loader)
            _save_metrics(cfg, results)
            _make_plots(cfg, results)
        if not results.get("squeeze"):
            results["squeeze"] = _run_squeeze(cfg, clf, loader)
            _save_metrics(cfg, results)
            _make_plots(cfg, results)
        if not results.get("adv_training"):
            results["adv_training"] = _eval_adv_trained(cfg, loader)
            _save_metrics(cfg, results)
            _make_plots(cfg, results)
        _save_examples(cfg, clf, loader)
        _write_report(cfg, results)
        print(f"Done. See {cfg['output_dir']}")
        return

    if not {"jpeg", "squeeze", "adv_training"} <= set(results.keys()):
        print("Partial run finished. Files saved incrementally in "
              f"{cfg['output_dir']}. Next: --method adv_training, then --evaluate-all.")
        return


if __name__ == "__main__":
    main()