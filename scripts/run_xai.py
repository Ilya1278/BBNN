"""Entry point XAI Agent.

Примеры:
  python scripts/run_xai.py --checkpoint runs/<run_id>/model_best.pt \
                            --config runs/<run_id>/config.yaml \
                            --image data/processed/test/motorcycle/0001.jpg \
                            --out reports/xai/sample1.png

  python scripts/run_xai.py --checkpoint ... --config ... \
                            --from-manifest data/processed/manifest.json \
                            --split test --n 20 --out-dir reports/xai/

  python scripts/run_xai.py --checkpoint ... --config ... \
                            --from-manifest data/processed/manifest.json \
                            --split test --sanity \
                            --out-dir reports/xai/sanity/
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.classifier import MotorcycleClassifier  # noqa: E402
from src.xai.gradcam import GradCAM, get_target_layer  # noqa: E402
from src.xai.visualize import make_xai_panel           # noqa: E402


# ---------- утилиты ----------

def load_cfg(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_classifier(checkpoint: str, config_path: str, device: str = "cuda"):
    cfg = load_cfg(config_path)
    clf = MotorcycleClassifier(checkpoint_path=checkpoint, cfg=cfg, device=device)
    return clf, cfg


def load_manifest_items(manifest_path: str, split: str) -> list[dict]:
    """
    Manifest: плоский список dict'ов с полями filepath/label/source/sha1.
    filepath — относительный, начинается со '<split>/...'.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Ожидался плоский список в manifest.json")
    prefix = split + "/"
    return [r for r in data if str(r.get("filepath", "")).startswith(prefix)]


def full_path(cfg: dict, filepath: str) -> str:
    base = Path(cfg.get("data", {}).get("processed_dir", "data/processed"))
    return str(base / filepath)


def heatmap_focus_score(hm: np.ndarray) -> dict:
    """Метрики концентрации heatmap. Осмысленный: top20_mass ≥ 0.55, cvb ≥ 1.5."""
    h, w = hm.shape
    flat = hm.astype(np.float64).flatten()
    p = flat / (flat.sum() + 1e-12)
    k = max(1, int(0.2 * flat.size))
    top20_mass = float(np.sort(p)[-k:].sum())

    ch, cw = h // 4, w // 4
    center = hm[ch:h - ch, cw:w - cw].mean()
    mask = np.ones_like(hm, dtype=bool)
    mask[ch:h - ch, cw:w - cw] = False
    border = hm[mask].mean() + 1e-8
    return {"top20_mass": top20_mass, "center_vs_border": float(center / border)}


def _gradcam_heatmap_for_diag(classifier, img_uint8: np.ndarray, class_id: int) -> np.ndarray:
    layer = get_target_layer(classifier.model, "auto")
    cam = GradCAM(classifier.model, layer)
    try:
        t = classifier.transform(Image.fromarray(img_uint8)).unsqueeze(0).to(classifier.device)
        return cam.generate(t, target_class=class_id)
    finally:
        cam.remove_hooks()


# ---------- режимы ----------

def run_single(args, classifier, cfg):
    if not args.out:
        raise SystemExit("--out обязателен вместе с --image")
    make_xai_panel(args.image, classifier.model, classifier, args.out)
    print(f"Saved {args.out}")


def run_batch(args, classifier, cfg):
    if not args.out_dir:
        raise SystemExit("--out-dir обязателен вместе с --from-manifest")
    items = load_manifest_items(args.from_manifest, args.split)
    if not items:
        raise SystemExit(f"В манифесте нет записей для split='{args.split}'")
    paths = [full_path(cfg, it["filepath"]) for it in items]

    random.seed(42)
    random.shuffle(paths)
    if args.n:
        paths = paths[:args.n]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(paths, 1):
        out = out_dir / f"sample{i:02d}.png"
        make_xai_panel(p, classifier.model, classifier, out)
        print(f"[{i}/{len(paths)}] {out}")


def run_sanity(args, classifier, cfg):
    items = load_manifest_items(args.from_manifest, args.split)
    moto    = [it for it in items if int(it.get("label", -1)) == 1]
    nonmoto = [it for it in items if int(it.get("label", -1)) == 0]

    if len(moto) < 5 or len(nonmoto) < 5:
        raise SystemExit(
            f"Недостаточно данных в split='{args.split}': "
            f"moto={len(moto)}, nonmoto={len(nonmoto)} (нужно ≥5 каждого)"
        )

    random.seed(42)
    random.shuffle(moto)
    random.shuffle(nonmoto)

    picks = [(full_path(cfg, it["filepath"]), "motorcycle",     f"moto_{i:02d}")
             for i, it in enumerate(moto[:5], 1)]
    picks += [(full_path(cfg, it["filepath"]), "not_motorcycle", f"nonmoto_{i:02d}")
              for i, it in enumerate(nonmoto[:5], 6)]

    out_dir = Path(args.out_dir or "reports/xai/sanity")
    out_dir.mkdir(parents=True, exist_ok=True)

    correct = 0
    rows = []
    for path, true_label, name in picks:
        img = np.array(Image.open(path).convert("RGB"))
        pred = classifier.predict(img)
        ok = (pred["label"] == true_label)
        correct += int(ok)

        try:
            class_id = classifier.CLASSES.index(true_label)
            hm = _gradcam_heatmap_for_diag(classifier, img, class_id)
            diag = heatmap_focus_score(hm)
        except Exception as e:
            diag = {"error": str(e)}

        out_path = out_dir / f"{name}.png"
        make_xai_panel(img, classifier.model, classifier, out_path)
        rows.append((name, true_label, pred["label"], pred["confidence"], ok, diag))
        print(f"[{name}] true={true_label:14s} pred={pred['label']:14s} "
              f"conf={pred['confidence']:.3f} ok={ok} diag={diag}")

    acc = correct / len(picks)
    print(f"\nSanity accuracy: {correct}/{len(picks)} = {acc:.1%}")

    summary = out_dir / "summary.txt"
    with open(summary, "w", encoding="utf-8") as f:
        f.write(f"Sanity accuracy: {correct}/{len(picks)} = {acc:.1%}\n")
        f.write(f"model_mode = {classifier.model_mode}\n")
        f.write(f"device     = {classifier.device}\n\n")
        for r in rows:
            f.write(f"{r[0]}: true={r[1]} pred={r[2]} conf={r[3]:.3f} "
                    f"ok={r[4]} diag={r[5]}\n")
    print(f"Summary: {summary}")


def main():
    p = argparse.ArgumentParser(description="XAI for MotorcycleClassifier")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--image")
    p.add_argument("--from-manifest")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int)
    p.add_argument("--out")
    p.add_argument("--out-dir")
    p.add_argument("--sanity", action="store_true",
                   help="5 мотоциклов + 5 не-мотоциклов с диагностикой")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    classifier, cfg = build_classifier(args.checkpoint, args.config, args.device)
    print(f"model_mode = {classifier.model_mode}, device = {classifier.device}")

    if args.sanity:
        run_sanity(args, classifier, cfg)
    elif args.image:
        run_single(args, classifier, cfg)
    elif args.from_manifest:
        run_batch(args, classifier, cfg)
    else:
        raise SystemExit("Укажи --image, --from-manifest или --sanity")


if __name__ == "__main__":
    main()