from __future__ import annotations

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
        from src.models.classifier import MotorcycleClassifier
        from src.data.dataset import MotorcycleDataset
        from src.data.augment import get_val_transform
    except Exception as exc:
        raise RuntimeError("classifier not ready, run Model Agent first") from exc

    import yaml

    acfg_path = ROOT / "configs" / "analytics.yaml"
    acfg = (
        yaml.safe_load(acfg_path.read_text(encoding="utf-8")) or {}
        if acfg_path.exists() else {}
    )

    run_cfg_path = run_dir / "config.yaml"
    if not run_cfg_path.exists():
        raise RuntimeError(f"run config not found: {run_cfg_path}")
    run_cfg = yaml.safe_load(run_cfg_path.read_text(encoding="utf-8")) or {}

    # manifest: предпочитаем снапшот из run — он отражает реальный split обучения
    manifest_snapshot = run_dir / "manifest_snapshot.json"
    manifest_path = (
        manifest_snapshot if manifest_snapshot.exists()
        else ROOT / run_cfg.get("data", {}).get("manifest", "data/processed/manifest.json")
    )
    if not Path(manifest_path).exists():
        raise RuntimeError(f"manifest not found: {manifest_path}")

    processed_dir = ROOT / run_cfg.get("data", {}).get("processed_dir", "data/processed")

    ckpt_path = run_dir / "model_best.pt"
    if not ckpt_path.exists():
        raise RuntimeError(f"checkpoint not found: {ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ВАЖНО: используем альбументационный val-трансформ — его ожидает MotorcycleDataset
    transform = get_val_transform(run_cfg)

    dataset = MotorcycleDataset(
        manifest_path=str(manifest_path),
        split="test",
        transform=transform,
        root=str(processed_dir),
    )
    print(f"[--eval-test] test samples: {len(dataset)}")

    if len(dataset) == 0:
        raise RuntimeError(
            "test split empty — проверь manifest_snapshot.json: filepath должен "
            "начинаться с 'test/' и быть относительным (см. INTERFACES.md, раздел 2)"
        )

    loader = DataLoader(
        dataset,
        batch_size=int(acfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=0,
    )

    # MotorcycleClassifier сам загружает чекпоинт из checkpoint_path
    classifier = MotorcycleClassifier(
        checkpoint_path=str(ckpt_path),
        cfg=run_cfg,
        device=device,
    )

    model = classifier.model  # уже в eval(), на device
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(classifier.device)
            out = model(x)
            if isinstance(out, dict):
                out = out.get("logits", out)
            pred = out.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(y.numpy().tolist())

    print(f"[--eval-test] collected {len(y_true)} predictions on device={device}")
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
