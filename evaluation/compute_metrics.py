"""Fidelity / diversity metrics for generated images.

Thin wrapper over the bundled ``fld`` library to report the metrics used in the
paper: FID (Inception features), CLIP-FID (CLIP features), FLD [Jiralerspong et
al. 2023], and improved precision / recall in Inception or CLIP feature space.

Features are extracted once per feature space and reused across metrics.

Usage
-----
    python -m evaluation.compute_metrics \
        --train-dir /path/to/train_images \
        --test-dir  /path/to/test_images \
        --gen-dir   /path/to/generated \
        --metrics fid clip_fid fld precision recall

``--gen-dir`` auto-detects ``inference_*/images`` or ``<digits>/images``
subfolders (one per seed) and concatenates them; otherwise it uses the folder
(or its ``images/`` subdir) directly.

Note: FID/CLIP-FID measure overall fidelity+diversity; the paper additionally
computes FLD on non-hallucinated samples only. Run the per-dataset validator
first and point ``--gen-dir`` at the kept (non-hallucinated) images to reproduce
that setting.
"""
import argparse
import json
import os
from datetime import datetime

import torch

from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from fld.features.CLIPFeatureExtractor import CLIPFeatureExtractor
from fld.metrics.FID import FID
from fld.metrics.FLD import FLD
from fld.metrics.PrecisionRecall import PrecisionRecall

ALL_METRICS = ["fid", "clip_fid", "fld", "precision", "recall"]


def list_inference_image_dirs(root):
    """Return the generated-image folders under ``root`` (one per seed if present)."""
    out = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            sub = os.path.join(root, name)
            p = os.path.join(sub, "images")
            if os.path.isdir(sub) and (name.startswith("inference_") or name.isdigit()) and os.path.isdir(p):
                out.append(p)
    if out:
        return out
    images_subdir = os.path.join(root, "images")
    return [images_subdir] if os.path.isdir(images_subdir) else [root]


def concat_features(extractor, dirs, extension="png"):
    feats = [extractor.get_dir_features(d, extension=extension) for d in dirs]
    return feats[0] if len(feats) == 1 else torch.cat(feats, dim=0)


def main():
    p = argparse.ArgumentParser(description="Compute FID / CLIP-FID / FLD / precision / recall.")
    p.add_argument("--train-dir", required=True, help="Reference training images.")
    p.add_argument("--test-dir", required=True, help="Held-out test images (used by FLD).")
    p.add_argument("--gen-dir", required=True, help="Generated images (auto-detects seed subfolders).")
    p.add_argument("--metrics", nargs="+", default=ALL_METRICS, choices=ALL_METRICS)
    p.add_argument("--train-ext", default="png")
    p.add_argument("--test-ext", default="png")
    p.add_argument("--gen-ext", default="png")
    p.add_argument("--out-json", default=None, help="Optional path to dump results as JSON.")
    args = p.parse_args()

    needs_clip = any(m in ("clip_fid",) for m in args.metrics)
    # Inception covers fid/fld/precision/recall; CLIP covers clip_fid.
    extractors = {"inception": InceptionFeatureExtractor()}
    if needs_clip:
        extractors["clip"] = CLIPFeatureExtractor()

    gen_dirs = list_inference_image_dirs(args.gen_dir)
    print("[info] generated folders:", *gen_dirs, sep="\n  ")

    feats = {}
    for space, ex in extractors.items():
        feats[space] = {
            "train": ex.get_dir_features(args.train_dir, extension=args.train_ext),
            "test": ex.get_dir_features(args.test_dir, extension=args.test_ext),
            "gen": concat_features(ex, gen_dirs, extension=args.gen_ext),
        }

    results = {}
    inc = feats["inception"]
    if "fid" in args.metrics:
        results["fid"] = FID().compute_metric(inc["train"], inc["test"], inc["gen"])
    if "fld" in args.metrics:
        results["fld"] = FLD().compute_metric(inc["train"], inc["test"], inc["gen"])
    if "precision" in args.metrics:
        results["precision"] = PrecisionRecall(mode="Precision").compute_metric(inc["train"], inc["test"], inc["gen"])
    if "recall" in args.metrics:
        results["recall"] = PrecisionRecall(mode="Recall").compute_metric(inc["train"], inc["test"], inc["gen"])
    if "clip_fid" in args.metrics:
        cl = feats["clip"]
        results["clip_fid"] = FID().compute_metric(cl["train"], cl["test"], cl["gen"])

    print("\n=== Metrics ===")
    for k in args.metrics:
        if k in results:
            print(f"  {k:>10}: {float(results[k]):.4f}")

    if args.out_json:
        payload = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "paths": {"train": args.train_dir, "test": args.test_dir, "gen": gen_dirs},
                   "metrics": {k: float(v) for k, v in results.items()}}
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    main()
