---
pretty_name: "VSM Benchmarks: Cards & ChessImages"
license: cc-by-4.0
task_categories:
  - unconditional-image-generation
  - image-classification
tags:
  - diffusion-models
  - hallucination
  - generative-models
  - benchmark
  - chess
  - playing-cards
size_categories:
  - 100K<n<1M
---

# 🎲 VSM Benchmarks — Cards & ChessImages

Two image benchmarks with **very large, structured semantic spaces** and **fast,
training-free validators**, released with the paper
**"Score-Control for Hallucination Reduction in Diffusion Models" (VSM)**.

📄 **Paper:** https://arxiv.org/abs/2606.00377 &nbsp;·&nbsp;
💻 **Code:** https://github.com/bhosalems/VSM

Both datasets are designed for **systematic studies of hallucination** in
generative (especially diffusion) models. Each sample has a well-defined notion
of *validity*, so a generated image can be checked — without any trained
critic — for whether it lies on the data manifold (a legal board / a real card)
or is a hallucination (an illegal configuration). This makes the hallucination
rate a directly measurable quantity rather than a perceptual judgement.

| Dataset | Domain | #Images | Resolution | Semantic space | Validity check |
|---|---|---|---|---|---|
| **Cards** | 2×2 grids of playing cards | 91,390 | 128×128 | ~10⁵ | per-quadrant template match |
| **ChessImages** | Chessboards rendered from FEN | 196,062 | 256×256 | ~10⁴⁴ legal states | FEN parse + `python-chess` legality |

## 📦 Repository layout

Image folders are shipped as `.tar` archives (a handful of large files uploads
far faster than hundreds of thousands of tiny PNGs); labels and metadata are
plain files.

```
mbhosale/VSM/
├── ChessImages/
│   ├── train_images.tar        # 176,938 × 256×256 PNG  -> train_images/
│   ├── test_images.tar         #  19,124 × 256×256 PNG  -> test_images/
│   ├── train_fen.json          # {"<image_id>": "<FEN>"}  (176,938 entries)
│   └── test_fen.json           # {"<image_id>": "<FEN>"}  ( 19,124 entries)
├── Cards/
│   ├── card_imgs.tar           # 91,390 × 128×128 PNG  -> card_imgs/
│   └── templates.tar           # 40 card-face templates (64×64) for the validator
└── validators/                 # training-free hallucination checkers (see below)
    ├── chess_validator.py
    ├── cards_validator.py
    └── templates/chess/        # 26 chess-piece templates used by chess_validator.py
```

## 🃏 Cards

A Cards sample is a **2×2 grid of four standard playing cards** at 128×128 (each
quadrant is 64×64). The card faces are drawn from **40 base cards** — ranks A
through 10 across the four suits (♣ ♦ ♥ ♠); no face cards (J/Q/K). With four
independently drawn quadrants the semantic space is on the order of 10⁵.

- `card_imgs.tar` — the full set of 91,390 grids.
- `templates.tar` — the 40 single-card reference templates (resized to 64×64),
  used by the training-free validator. Extracts to `resized_templates_png/`.

**Hallucination check.** Each of the four quadrants is template-matched
(`cv2.TM_CCOEFF_NORMED`) against the 40 templates. If the *worst* quadrant match
falls below a threshold, the image cannot be explained by four valid cards and is
counted as a hallucination (wrong/garbled symbols, color/count errors, noise).

## ♟️ ChessImages

A ChessImages sample is a **256×256 rendering of a chessboard** generated from a
FEN string. The space of legal piece placements is astronomically large (~10⁴⁴),
making this a stringent test of whether a generator stays on-manifold.

- `train_images.tar` (176,938) and `test_images.tar` (19,124) — rendered boards;
  filenames are `<image_id>.png`.
- `train_fen.json` / `test_fen.json` — map each `<image_id>` to its full FEN.

**Hallucination check.** The board image is parsed back to a FEN
piece-placement string by template-matching each of the 64 squares, then its
legality is tested with `python-chess` (`chess.Board.is_valid()`). Boards that
violate the rules (two white kings, >8 pawns, pawns on the back rank, …) are
counted as hallucinations. The validator can additionally report exact / fuzzy
FEN-reconstruction accuracy against the ground-truth FENs (conditional setting).

## 🚀 Usage

### Option A — via the VSM repo (recommended)

The [VSM repo](https://github.com/bhosalems/VSM) ships loaders and a downloader
that fetch and unpack these archives for you:

```bash
python -m datasets.download --dataset chess --out ./data/ChessImages
python -m datasets.download --dataset cards --out ./data/Cards
```

### Option B — directly with huggingface_hub

```python
from huggingface_hub import snapshot_download
import tarfile, glob, os

root = snapshot_download(repo_id="mbhosale/VSM", repo_type="dataset",
                         allow_patterns=["ChessImages/*"])
for tar in glob.glob(os.path.join(root, "ChessImages", "**", "*.tar"), recursive=True):
    with tarfile.open(tar) as t:
        t.extractall(os.path.dirname(tar))
# -> ChessImages/train_images/*.png, test_images/*.png, train_fen.json, test_fen.json
```

## ✅ Verifiers (training-free hallucination validators)

The two validators used in the paper are bundled here under `validators/`, so you
can score generated images without cloning the full code repo. They need
`opencv-python`, `numpy`, `tqdm`, and — for chess — `python-chess`.

**ChessImages** — parse each rendered board back to a FEN and test legality with
`python-chess`; illegal boards count as hallucinations. The piece templates ship
alongside the script, so no extra flags are required:

```bash
python validators/chess_validator.py --gen-dir /path/to/generated/boards
# optional: also report FEN-reconstruction accuracy against the ground truth
python validators/chess_validator.py --gen-dir /path/gen \
    --gt-json ChessImages/test_fen.json --conditional
```

**Cards** — template-match each of the 4 quadrants against the card templates; if
the worst-matching quadrant falls below the threshold, the image is a
hallucination:

```bash
tar -xf Cards/templates.tar                      # -> resized_templates_png/
python validators/cards_validator.py --gen-dir /path/to/generated/cards \
    --template-dir resized_templates_png
```

Each validator reports the hallucination rate (H%) per folder (repeat `--gen-dir`
for multiple seeds). Full fidelity metrics (FID / CLIP-FID / FLD) live in the
[VSM repo](https://github.com/bhosalems/VSM) under `evaluation/`.

## ⚖️ Intended use & license

These benchmarks are released **for research on generative-model reliability and
hallucination**. They are synthetic renderings (rule-based card grids and chess
positions) and contain no personal data. Released under **CC BY 4.0** — please
verify this matches your intended terms before redistribution, and cite the paper
below.

## 📑 Citation

```bibtex
@article{bhosale2026vsm,
  title   = {Score-Control for Hallucination Reduction in Diffusion Models},
  author  = {Bhosale, Mahesh and Devulapally, Naresh Kumar and Wasi, Abdul and
             Pham, Chau and Lokhande, Vishnu Suresh and Doermann, David},
  journal = {arXiv preprint arXiv:2606.00377},
  year    = {2026}
}
```

The ChessImages dataset uses [python-chess](https://python-chess.readthedocs.io)
and FEN strings sampled from [VALUED](https://arxiv.org/abs/2311.12610).
