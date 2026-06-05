"""Training-free hallucination detector for the Cards dataset.

A generated Cards image is a 2x2 grid of playing cards. Each of the four
quadrants is template-matched (``cv2.TM_CCOEFF_NORMED``) against the set of
standard card templates. If the *worst* quadrant match falls below
``--threshold``, the image cannot be explained by any valid card and is counted
as *hallucinated* (wrong symbol count/color, missing or conflicting symbols,
noise, etc.). See paper Section 5.2.1 / Appendix I.

Card templates are not bundled here (they ship with the Cards dataset on
HuggingFace, under ``Cards/templates``). Point ``--template-dir`` at that folder.

Usage
-----
    python -m evaluation.cards_validator \
        --gen-dir /path/to/generated/cards \
        --template-dir /path/to/Cards/templates \
        [--threshold 0.95] [--color]

``--gen-dir`` may be repeated (one folder per seed). Reports per-folder and
aggregated (mean +/- std) hallucination rate.
"""
import argparse
import glob
import os

import cv2
import numpy as np
from tqdm import tqdm


def load_templates(directory, use_color=False):
    flag = cv2.IMREAD_COLOR if use_color else cv2.IMREAD_GRAYSCALE
    paths = sorted(glob.glob(os.path.join(directory, "*.png")))
    templates = [cv2.imread(p, flag) for p in paths]
    templates = [t for t in templates if t is not None]
    if not templates:
        raise ValueError(f"No .png templates found in {directory}")
    return templates


def quadrant_scores(img, templates, use_color):
    """Return the best match score for each of the 4 quadrants."""
    if use_color:
        grid = img
    else:
        grid = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = grid.shape[:2]
    th, tw = h // 2, w // 2

    scores = []
    for y in (0, th):
        for x in (0, tw):
            tile = grid[y:y + th, x:x + tw]
            best = -1.0
            for tmpl in templates:
                if tile.shape != tmpl.shape:
                    continue
                best = max(best, cv2.matchTemplate(tile, tmpl, cv2.TM_CCOEFF_NORMED).max())
            scores.append(best)
    return scores


def validate_dirs(images_dirs, template_dir, threshold=0.95, use_color=False,
                  noise_threshold=None):
    templates = load_templates(template_dir, use_color)
    rates = []
    for folder in images_dirs:
        paths = sorted(glob.glob(os.path.join(folder, "*.png")))
        if not paths:
            print(f"[warn] no PNGs in {folder}, skipping")
            continue
        hallucinated = total = 0
        for path in tqdm(paths, desc=os.path.basename(folder.rstrip("/")) or "cards"):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            # Optional: skip near-uniform noise tiles before scoring.
            if noise_threshold is not None and np.std(img) < noise_threshold:
                continue
            total += 1
            if min(quadrant_scores(img, templates, use_color)) < threshold:
                hallucinated += 1
        if total:
            rate = 100 * hallucinated / total
            rates.append(rate)
            print(f"  {folder}: {hallucinated}/{total} hallucinated ({rate:.2f}%)")

    if rates:
        print("\n=== Cards validation ===")
        print(f"  hallucination%: {np.mean(rates):.2f} +/- {np.std(rates):.2f}")
    return rates


def main():
    p = argparse.ArgumentParser(description="Cards hallucination detector (template matching).")
    p.add_argument("--gen-dir", action="append", required=True,
                   help="Folder of generated 2x2-grid PNGs. Repeat for multiple seeds.")
    p.add_argument("--template-dir", required=True,
                   help="Folder of card templates (ships with the Cards dataset).")
    p.add_argument("--threshold", type=float, default=0.95,
                   help="Min per-quadrant match score for a valid card.")
    p.add_argument("--color", action="store_true", help="Match on color instead of grayscale.")
    p.add_argument("--noise-threshold", type=float, default=None,
                   help="If set, skip images with pixel std below this (pure-noise filter).")
    args = p.parse_args()

    validate_dirs(args.gen_dir, args.template_dir, args.threshold, args.color, args.noise_threshold)


if __name__ == "__main__":
    main()
