"""Minimal PyTorch datasets for the proposed Cards and ChessImages benchmarks.

These are framework-agnostic loaders for quickly training *your own* model on the
proposed datasets, or for iterating over the images for evaluation. They return
images normalized to [-1, 1] in CHW order.

For training with the two reference codebases instead, use their native loaders:
  * latent-diffusion: ``ldm.data.chess.ChessDataset`` (configured via the YAMLs
    under ``latent-diffusion/configs/latent-diffusion/chess-*.yaml``).
  * DDPM-IP: pass the image folder to ``--data_dir`` (see DDPM-IP/scripts).

Expected on-disk layout (as downloaded by ``datasets/download.py``):

    ChessImages/
      ├── train_images/ *.png      ├── test_images/ *.png
      └── train_fen.json           └── test_fen.json     # {"<name>": "<FEN>"}
    Cards/
      ├── train/ *.png             └── test/ *.png
"""
import glob
import json
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def _load_image(path, size):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0   # -> [-1, 1]
    return np.transpose(arr, (2, 0, 1))                      # HWC -> CHW


class ImageFolderDataset(Dataset):
    """Generic folder-of-PNGs dataset (used for Cards and as a base for Chess)."""

    def __init__(self, root, split="train", size=None, exts=(".png", ".jpg")):
        self.dir = os.path.join(root, split) if os.path.isdir(os.path.join(root, split)) else root
        self.paths = sorted(p for e in exts for p in glob.glob(os.path.join(self.dir, f"*{e}")))
        if not self.paths:
            raise FileNotFoundError(f"No images found under {self.dir}")
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        return {"image": _load_image(path, self.size), "name": os.path.basename(path)}


class ChessImagesDataset(ImageFolderDataset):
    """ChessImages with optional FEN labels from ``{split}_fen.json``."""

    def __init__(self, root, split="train", size=256):
        img_dir = os.path.join(root, f"{split}_images")
        super().__init__(root=img_dir if os.path.isdir(img_dir) else root, split=split, size=size)
        fen_path = os.path.join(root, f"{split}_fen.json")
        self.fens = json.load(open(fen_path)) if os.path.exists(fen_path) else {}

    def __getitem__(self, i):
        item = super().__getitem__(i)
        item["fen"] = self.fens.get(os.path.splitext(item["name"])[0], "")
        return item


class CardsDataset(ImageFolderDataset):
    """Cards 2x2 grids (128x128)."""

    def __init__(self, root, split="train", size=128):
        super().__init__(root=root, split=split, size=size)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Smoke-test a dataset loader.")
    p.add_argument("--dataset", required=True, choices=["chess", "cards"])
    p.add_argument("--root", required=True)
    p.add_argument("--split", default="train")
    args = p.parse_args()

    ds = ChessImagesDataset(args.root, args.split) if args.dataset == "chess" \
        else CardsDataset(args.root, args.split)
    print(f"{args.dataset}: {len(ds)} images; sample 0 shape={ds[0]['image'].shape}")
    if args.dataset == "chess":
        print("  fen:", ds[0]["fen"])
