"""Training-free hallucination detector for the Shapes dataset.

Shapes images are 64x64 grayscale, split into three vertical regions; a valid
image holds at most one shape (triangle / square / pentagon) per region. We
threshold to binary and classify each region by its white-pixel count. An image
is *hallucinated* if it is blank/near-blank or contains more shapes than allowed
(duplicates, shapes in the wrong region, malformed shapes). See paper Sec 5.2.1.

Usage
-----
    python -m evaluation.shapes_validator --gen-dir /path/to/generated/shapes

``--gen-dir`` may be repeated (one folder per seed). Reports per-folder and
aggregated (mean +/- std) hallucination rate.
"""
import argparse
import os
from collections import defaultdict

import cv2
import numpy as np

# White-pixel-count tolerances per shape, calibrated to give 100% accuracy on
# the real Shapes data (see paper). Region order: triangle | square | pentagon.
TOLERANCES = {"triangle": (63, 67), "square": (60, 68), "pentagon": (68, 72)}
REGION_SHAPE = {0: "triangle", 1: "square", 2: "pentagon"}


def analyze_image(img_path):
    """Return (list_of_detected_shapes, error). error is None when the image is valid."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.shape != (64, 64):
        return None, "invalid image shape"

    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = img[:63, :63]
    if np.sum(img == 255) < 50:
        return None, "black image"

    regions = [img[:, 0:21], img[:, 21:42], img[:, 42:63]]
    detected = []
    for i, region in enumerate(regions):
        shape = REGION_SHAPE[i]
        white = np.sum(region == 255)
        if TOLERANCES[shape][0] <= white <= TOLERANCES[shape][1]:
            detected.append(shape)

    if len(detected) == 0:
        return None, "black image"
    if len(detected) > 3:
        return None, "more than 3 shapes"
    return detected, None


def process_directory(directory_path):
    shape_counts = defaultdict(int)
    invalid = []
    files = [f for f in os.listdir(directory_path) if f.endswith(".png")]
    for name in files:
        result, error = analyze_image(os.path.join(directory_path, name))
        if error:
            invalid.append((name, error))
        else:
            for shape in result:
                shape_counts[shape] += 1
    return {"shape_counts": dict(shape_counts), "invalid": invalid, "total": len(files)}


def validate_dirs(images_dirs):
    rates = []
    for folder in images_dirs:
        if not os.path.isdir(folder):
            print(f"[warn] not a directory, skipping: {folder}")
            continue
        res = process_directory(folder)
        if res["total"] == 0:
            continue
        rate = 100 * len(res["invalid"]) / res["total"]
        rates.append(rate)
        print(f"  {folder}: {len(res['invalid'])}/{res['total']} hallucinated "
              f"({rate:.2f}%)  shapes={res['shape_counts']}")
    if rates:
        print("\n=== Shapes validation ===")
        print(f"  hallucination%: {np.mean(rates):.2f} +/- {np.std(rates):.2f}")
    return rates


def main():
    p = argparse.ArgumentParser(description="Shapes hallucination detector (rule-based).")
    p.add_argument("--gen-dir", action="append", required=True,
                   help="Folder of generated 64x64 PNGs. Repeat for multiple seeds.")
    args = p.parse_args()
    validate_dirs(args.gen_dir)


if __name__ == "__main__":
    main()
