#!/usr/bin/env python3
"""
CleanFID per seed (images/ only) with index filtering.

- For each <fake_root>/<seed>/images, take only files whose numeric index < 100
  (works for names like '00042.png' or 'sample_00042.png').
- Compute FID twice: Inception V3 and CLIP ViT-B/32.
- Print per-seed counts and scores, then mean/std across seeds.
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import os
import ssl
import certifi
import numpy as np
import re
import tempfile
import shutil

# ---------- Robust SSL setup ----------
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
def _cafile_context(*args, **kwargs):
    kwargs.setdefault("cafile", certifi.where())
    return ssl.create_default_context(*args, **kwargs)
ssl._create_default_https_context = _cafile_context
# --------------------------------------

# Optional caches
os.environ.setdefault("CLEANFID_CACHE", str(Path.home() / ".cache" / "clean-fid"))
os.environ.setdefault("TORCH_HOME", str(Path.home() / ".cache" / "torch"))
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

from cleanfid import fid as cleanfid
from cleanfid import features as cf_features

BACKBONES = ("inception_v3", "clip_vit_b_32")
INDEX_LIMIT = 100          # only indices < 100
USE_SYMLINKS = True        # fall back to copy if symlinks fail

# accept "sample_00042.png" or "00042.png" (also supports negative, though those will be filtered out by < INDEX_LIMIT)
INDEX_RE = re.compile(r"^(?:sample_)?(-?\d+)\.(?:png|jpg|jpeg|bmp|webp|tif|tiff)$", re.IGNORECASE)

def preload_extractors():
    for name in BACKBONES:
        try:
            cf_features.build_feature_extractor(name)
        except Exception as e:
            print(f"[warn] Preload for {name} failed: {e}")

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}

def list_targets(fake_root: Path) -> List[Tuple[str, Path]]:
    """
    Build targets as (seed_name, seed_images_dir) where seed_images_dir is exactly:
      <fake_root>/<seed>/images
    Also supports <fake_root>/images (no seed level).
    Only files directly inside images/ are considered.
    """
    targets: List[Tuple[str, Path]] = []

    # Case A: --fake has images/ directly
    root_images = fake_root / "images"
    if root_images.is_dir() and any(is_image_file(x) for x in root_images.iterdir() if x.is_file()):
        targets.append((fake_root.name, root_images))

    # Case B: seed dirs like 12346/, 12347/, ...
    for seed_dir in sorted([x for x in fake_root.iterdir() if x.is_dir()]):
        img_dir = seed_dir / "images"
        if img_dir.is_dir() and any(is_image_file(x) for x in img_dir.iterdir() if x.is_file()):
            targets.append((seed_dir.name, img_dir))

    return targets

def parse_index(name: str):
    """Return integer index from filename, or None if it doesn't match."""
    m = INDEX_RE.match(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None

def build_filtered_view(src_dir: Path, limit: int) -> Tuple[Path, int]:
    """
    Create a temporary folder containing symlinks (or copies) to images in src_dir
    whose parsed index < limit. Returns (temp_dir_path, count_selected).
    """
    tmp = Path(tempfile.mkdtemp(prefix="fid_subset_"))
    selected = 0
    for f in sorted([x for x in src_dir.iterdir() if x.is_file() and is_image_file(x)]):
        idx = parse_index(f.name)
        if idx is None or idx >= limit:
            continue
        dst = tmp / f.name  # keep original filename
        try:
            if USE_SYMLINKS:
                os.symlink(f, dst)
            else:
                shutil.copy2(f, dst)
        except Exception:
            # fall back to copy if symlink not permitted
            shutil.copy2(f, dst)
        selected += 1
    return tmp, selected

def compute_cleanfid(real: Path, fake_dir: Path, batch_size: int, num_workers: int, mode: str, backbone: str) -> float:
    return cleanfid.compute_fid(
        str(fake_dir), str(real),
        mode=mode,
        batch_size=batch_size,
        num_workers=num_workers,
        model_name=backbone,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, type=Path, help="Folder of real images")
    ap.add_argument("--fake", required=True, type=Path, help="Folder containing seeds (each with images/)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--mode", type=str, default="clean",
                    choices=["clean", "legacy_pytorch", "legacy_tensorflow"])
    ap.add_argument("--csv", type=Path, default=None, help="Optional CSV path for results")
    args = ap.parse_args()

    real = args.real
    fake_root = args.fake
    if not real.is_dir():
        raise FileNotFoundError(f"Real folder not found: {real}")
    if not fake_root.is_dir():
        raise FileNotFoundError(f"Fake folder not found: {fake_root}")

    preload_extractors()

    targets = list_targets(fake_root)
    if not targets:
        raise RuntimeError(
            f"No seed images/ folders found under {fake_root}.\n"
            f"Expected layout like: <fake_root>/<seed>/images/*.png"
        )

    print(f"\nCleanFID results (mode={args.mode}, batch_size={args.batch_size})")
    print(f"Real: {real}")
    print(f"Fake root: {fake_root}\n")
    print(f"{'Seed (images/)':40s}  {'Count':>7s}  {'FID (Inception)':>16s}  {'FID (CLIP ViT-B/32)':>22s}")

    rows = []
    inc_vals, clip_vals = [], []

    temp_dirs: List[Path] = []
    try:
        for seed_name, img_dir in targets:
            # Build a filtered temp view with index < INDEX_LIMIT
            tmp_dir, count = build_filtered_view(img_dir, INDEX_LIMIT)
            temp_dirs.append(tmp_dir)

            if count == 0:
                print(f"{(seed_name + '/images'):40s}  {count:7d}  {'-':>16s}  {'-':>22s}")
                continue

            fid_inc  = compute_cleanfid(real, tmp_dir, args.batch_size, args.num_workers, args.mode, "inception_v3")
            fid_clip = compute_cleanfid(real, tmp_dir, args.batch_size, args.num_workers, args.mode, "clip_vit_b_32")

            print(f"{(seed_name + '/images'):40s}  {count:7d}  {fid_inc:16.4f}  {fid_clip:22.4f}")
            rows.append((seed_name, count, fid_inc, fid_clip))
            inc_vals.append(fid_inc)
            clip_vals.append(fid_clip)

        if not rows:
            print("\nNo successful evaluations.")
            return

        inc_mean, inc_std = float(np.mean(inc_vals)), float(np.std(inc_vals, ddof=0))
        clip_mean, clip_std = float(np.mean(clip_vals)), float(np.std(clip_vals, ddof=0))

        print("\nAggregate across seeds (images/ only, index < %d):" % INDEX_LIMIT)
        print(f"  Inception  -> mean: {inc_mean:.4f}, std: {inc_std:.4f}")
        print(f"  CLIP ViT-B -> mean: {clip_mean:.4f}, std: {clip_std:.4f}")

        if args.csv:
            import csv
            with open(args.csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["seed", "count_used", "fid_inception_v3", "fid_clip_vit_b_32", "mode", "batch_size", "num_workers"])
                for seed, cnt, fi, fc in rows:
                    w.writerow([seed, cnt, fi, fc, args.mode, args.batch_size, args.num_workers])
                w.writerow(["__aggregate_mean__", "", inc_mean, clip_mean, args.mode, args.batch_size, args.num_workers])
                w.writerow(["__aggregate_std__",  "", inc_std,  clip_std,  args.mode, args.batch_size, args.num_workers])
            print(f"\nSaved CSV to: {args.csv}")

    finally:
        # clean up temp dirs
        for d in temp_dirs:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass

if __name__ == "__main__":
    main()