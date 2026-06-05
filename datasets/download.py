"""Download the proposed VSM benchmark datasets (Cards, ChessImages) from the Hub.

The two datasets live under a single HuggingFace repo, one subfolder each, with
the images shipped as ``.tar`` archives (uploading a few large files is far
faster than hundreds of thousands of tiny PNGs):

    mbhosale/VSM
    ├── Cards/         card_imgs.tar  templates.tar
    └── ChessImages/   train_images.tar  test_images.tar  train_fen.json  test_fen.json

This script downloads one subfolder, lifts its contents up into ``--out``,
extracts the tars, and renames the Cards folders so the on-disk layout matches
what ``datasets/dataset.py`` expects:

    <out>/                       (chess)            <out>/            (cards)
      ├── train_images/ *.png                         ├── train/ *.png
      ├── test_images/  *.png                          └── templates/ *.png
      ├── train_fen.json
      └── test_fen.json

Usage
-----
    python -m datasets.download --dataset chess --out ./data/ChessImages
    python -m datasets.download --dataset cards --out ./data/Cards
    # keep the .tar files around instead of deleting them after extraction:
    python -m datasets.download --dataset cards --out ./data/Cards --no-extract
"""
import argparse
import glob
import os
import shutil
import tarfile

HF_REPO_ID = "mbhosale/VSM"
SUBFOLDER = {"chess": "ChessImages", "cards": "Cards"}

# Rename extracted folders to the names the loaders / validators expect.
RENAME = {
    "cards": {"card_imgs": "train", "resized_templates_png": "templates"},
    "chess": {},
}


def _extract_tars(folder, cleanup=True):
    for tar in sorted(glob.glob(os.path.join(folder, "*.tar"))):
        print(f"  extracting {os.path.basename(tar)} ...")
        with tarfile.open(tar) as t:
            t.extractall(folder)
        if cleanup:
            os.remove(tar)


def download(dataset, out_dir, repo_id=HF_REPO_ID, revision=None, extract=True):
    from huggingface_hub import snapshot_download

    subfolder = SUBFOLDER[dataset]
    snap = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"{subfolder}/*"],
    )

    # Lift <snap>/<subfolder>/* up into out_dir (strip the leading subfolder/).
    src = os.path.join(snap, subfolder)
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(src):
        s, d = os.path.join(src, name), os.path.join(out_dir, name)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    if extract:
        _extract_tars(out_dir)
        for old, new in RENAME[dataset].items():
            old_p, new_p = os.path.join(out_dir, old), os.path.join(out_dir, new)
            if os.path.isdir(old_p) and not os.path.exists(new_p):
                os.rename(old_p, new_p)

    print(f"Downloaded {subfolder} from {repo_id} -> {out_dir}")
    return out_dir


def main():
    p = argparse.ArgumentParser(description="Download VSM benchmark datasets from HuggingFace.")
    p.add_argument("--dataset", required=True, choices=list(SUBFOLDER))
    p.add_argument("--out", required=True, help="Local output directory.")
    p.add_argument("--repo-id", default=HF_REPO_ID)
    p.add_argument("--revision", default=None)
    p.add_argument("--no-extract", dest="extract", action="store_false",
                   help="Keep the downloaded .tar archives instead of extracting them.")
    args = p.parse_args()
    download(args.dataset, args.out, args.repo_id, args.revision, args.extract)


if __name__ == "__main__":
    main()
