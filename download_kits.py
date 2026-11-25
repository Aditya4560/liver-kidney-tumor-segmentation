"""Download KiTS23 kidney tumor CT cases using the official kits23 package."""

import subprocess
import sys
import os
from pathlib import Path

# Step 1: Install kits23 if not already installed
print("Checking kits23 package...")
try:
    import kits23
    print("kits23 already installed.")
except ImportError:
    print("Installing kits23...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kits23"])
    import kits23

# Step 2: Find where kits23 wants to store data
from kits23.configuration.labels import KITS_HF_REPO
from kits23.configuration.paths import KiTS23_DATASET_DIR
print(f"Official KiTS23 HF repo: {KITS_HF_REPO}")
print(f"Official download dir  : {KiTS23_DATASET_DIR}")

# Step 3: Download only first 50 cases using huggingface_hub
from huggingface_hub import hf_hub_download

NUM_CASES = 50
OUT_DIR = Path("data/kits_small")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nDownloading {NUM_CASES} cases to {OUT_DIR.resolve()} ...")
print("(This may take a while depending on your internet speed.)\n")

success, failed = 0, []

for i in range(NUM_CASES):
    folder = f"case_{i:05d}"
    for fname in ["imaging.nii.gz", "segmentation.nii.gz"]:
        dest = OUT_DIR / folder / fname
        if dest.exists():
            print(f"[skip] {folder}/{fname} already downloaded")
            continue
        # Try multiple known path structures in the HF repo
        tried = []
        downloaded = False
        for prefix in [f"dataset/{folder}", folder]:
            path_in_repo = f"{prefix}/{fname}"
            tried.append(path_in_repo)
            try:
                hf_hub_download(
                    repo_id=KITS_HF_REPO,
                    filename=path_in_repo,
                    repo_type="dataset",
                    local_dir=str(OUT_DIR),
                )
                # Move to expected location if stored under dataset/ subdir
                nested = OUT_DIR / "dataset" / folder / fname
                if nested.exists() and not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    nested.rename(dest)
                print(f"[✓] {folder}/{fname}")
                success += 1
                downloaded = True
                break
            except Exception as e:
                continue
        if not downloaded:
            print(f"[✗] {folder}/{fname}  (tried: {tried})")
            failed.append(f"{folder}/{fname}")

print(f"\n=== Done: {success} files downloaded, {len(failed)} failed ===")
if failed:
    print(f"Failed ({len(failed)} files):", failed[:10], "..." if len(failed) > 10 else "")

print(f"\nDataset at: {OUT_DIR.resolve()}")
print("\nTo train your model, run:")
print(f'  python -m src.train --model unet --dataset kits --data-dir data/kits_small --epochs 30 --batch-size 8 --save-dir outputs/kits_unet')
