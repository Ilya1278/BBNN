import argparse
import os
import sys
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.collect import collect
from src.data.clean import clean
from src.data.split import split
from src.data.report import generate_report
from src.data.utils import load_config, ensure_dir, load_json


STAGES = ["collect", "clean", "split", "report"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--from", dest="from_stage", default="collect",
                        choices=STAGES,
                        help="С какой стадии начать: collect|clean|split|report")
    parser.add_argument("--only", dest="only_stage", default=None,
                        choices=STAGES,
                        help="Выполнить только одну стадию")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for p in cfg["paths"].values():
        ensure_dir(p)

    start = STAGES.index(args.from_stage)

    def should_run(stage: str) -> bool:
        if args.only_stage:
            return stage == args.only_stage
        return STAGES.index(stage) >= start

    raw_path = os.path.join(cfg["paths"]["interim"], "raw_manifest.json")
    cleaned_path = os.path.join(cfg["paths"]["interim"], "cleaned_manifest.json")
    manifest_path = os.path.join(cfg["paths"]["processed"], "manifest.json")

    raw = cleaned = manifest = None

    if should_run("collect"):
        print("1/4 Collecting...")
        raw = collect(cfg)
    else:
        if os.path.exists(raw_path):
            raw = load_json(raw_path)
            print(f"[skip] collect: loaded {len(raw)} from {raw_path}")
        else:
            raise RuntimeError(f"missing {raw_path}; run collect first")

    if should_run("clean"):
        print("2/4 Cleaning...")
        cleaned = clean(cfg, raw)
    else:
        if os.path.exists(cleaned_path):
            cleaned = load_json(cleaned_path)
            print(f"[skip] clean: loaded {len(cleaned)} from {cleaned_path}")
        else:
            raise RuntimeError(f"missing {cleaned_path}; run clean first")

    if should_run("split"):
        # --- safety backup перед перезаписью manifest.json
        backup_path = os.path.join(cfg["paths"]["processed"], "manifest_v1_hard9.json")
        if os.path.exists(manifest_path) and not os.path.exists(backup_path):
            shutil.copy2(manifest_path, backup_path)
            print(f"[backup] saved {backup_path}")

        print("3/4 Splitting...")
        manifest = split(cfg, cleaned)
    else:
        if os.path.exists(manifest_path):
            manifest = load_json(manifest_path)
            print(f"[skip] split: loaded {len(manifest)} from {manifest_path}")
        else:
            raise RuntimeError(f"missing {manifest_path}; run split first")

    if should_run("report"):
        print("4/4 Reporting...")
        generate_report(cfg, manifest)

    print("Done. Check data/processed/ and reports/")


if __name__ == "__main__":
    main()