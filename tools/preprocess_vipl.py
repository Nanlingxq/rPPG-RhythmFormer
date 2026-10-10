"""Preprocess VIPL-HR without starting model training."""

import argparse
import sys
from types import SimpleNamespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config
from dataset.data_loader.VIPLHRLoader import VIPLHRLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    args = parser.parse_args()

    config = get_config(SimpleNamespace(config_file=args.config_file))
    splits = ("train", "test") if args.split == "all" else (args.split,)

    for split in splits:
        data_config = getattr(config, split.upper()).DATA
        if not data_config.DO_PREPROCESS:
            print(f"Skipping {split}: DO_PREPROCESS is False")
            continue
        print(f"Preprocessing {split} split")
        VIPLHRLoader(
            name=split,
            data_path=data_config.DATA_PATH,
            config_data=data_config,
        )


if __name__ == "__main__":
    main()
