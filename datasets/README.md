# Datasets

Helpers for **using** the two benchmark datasets proposed in the paper. These do
not generate the datasets - they download the released versions and load them for
training/evaluation.

| Dataset | Size | Resolution | Validator |
|---|---|---|---|
| **Cards** | 94,000 | 128×128 | `evaluation/cards_validator.py` |
| **ChessImages** | 190,000 | 256×256 | `evaluation/chess_validator.py` |

Both feature extremely large semantic spaces (Cards ~10⁵, ChessImages ~10⁴⁴ valid
states) with fast, training-free validators - designed for systematic
hallucination studies.

## Download

Both datasets live under a single HuggingFace repo,
[mbhosale/VSM](https://huggingface.co/datasets/mbhosale/VSM). Images are shipped
as `.tar` archives; the downloader fetches the requested subfolder, extracts the
tars, and lays the files out exactly as the loaders below expect:

```
mbhosale/VSM
├── Cards/         # card_imgs.tar  templates.tar      -> train/  templates/
└── ChessImages/   # train_images.tar test_images.tar  -> train_images/ test_images/
                   # + train_fen.json test_fen.json
```

```bash
python -m datasets.download --dataset chess --out ./data/ChessImages
python -m datasets.download --dataset cards --out ./data/Cards
# add --no-extract to keep the raw .tar archives
```

## Load

```python
from datasets.dataset import ChessImagesDataset, CardsDataset

chess = ChessImagesDataset("./data/ChessImages", split="train", size=256)
cards = CardsDataset("./data/Cards", split="train", size=128)
print(chess[0]["image"].shape, chess[0]["fen"])   # CHW in [-1,1], FEN string
```

Images are returned as CHW float arrays in `[-1, 1]`.

### Training with the reference codebases

- **latent-diffusion** - use the native loader `ldm.data.chess.ChessDataset`,
  configured by the YAMLs in `latent-diffusion/configs/latent-diffusion/chess-*.yaml`.
- **DDPM-IP** - pass the image folder directly to `--data_dir` (see `DDPM-IP/scripts`).
