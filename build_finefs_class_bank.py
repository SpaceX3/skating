#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from finefs_class_bank import build_class_banks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FineFS train-only coarse-class banks")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/media/v100/disk3t/finefs_pocr_classifier/experiments/e11_video_grain_split/cliplet_manifest.jsonl"),
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("/media/v100/disk3t/finefs_pocr_classifier/features/videomae_base_1s_stride05"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/media/v100/disk3t/skating/finefs_c1_class_bank_first_token"),
    )
    args = parser.parse_args()
    report = build_class_banks(args.manifest, args.feature_root, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
