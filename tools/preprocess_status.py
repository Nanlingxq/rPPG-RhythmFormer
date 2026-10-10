"""Show VIPL-HR preprocessing progress."""

from pathlib import Path
import shutil

import numpy as np


ROOT = Path("/data/new_data/lwk/HREstimation/preprocessed/VIPL-HR")
EXPECTED = {"train": 10252, "test": 2680}


def size_gb(path):
    return sum(item.stat().st_size for item in path.rglob("*.npy")) / 1e9


def main():
    counts = {}
    for split, expected in EXPECTED.items():
        split_root = ROOT / split
        count = len(list(split_root.rglob("*_input*.npy"))) if split_root.exists() else 0
        counts[split] = count
        percentage = count / expected * 100
        print(f"{split:5s}: {count:5d}/{expected} ({percentage:6.2f}%)")

    total = sum(counts.values())
    total_expected = sum(EXPECTED.values())
    print(f"total : {total:5d}/{total_expected} ({total / total_expected * 100:6.2f}%)")
    print(f"cache : {size_gb(ROOT):.1f} GB")

    disk = shutil.disk_usage("/data")
    print(f"/data free: {disk.free / 1e9:.1f} GB")

    inputs = list(ROOT.rglob("*_input*.npy"))
    if inputs:
        sample = np.load(inputs[0], mmap_mode="r")
        print(f"sample : shape={sample.shape}, dtype={sample.dtype}")


if __name__ == "__main__":
    main()
