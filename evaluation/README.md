# Evaluation

Training-free hallucination detectors and fidelity/diversity metrics used in the
paper. Run everything from the repo root.

All validators take one or more `--gen-dir` folders of generated images (repeat
the flag for multiple seeds) and report a per-folder and aggregated (mean ± std)
**hallucination rate (H%)** - the fraction of generated samples judged invalid.

## Hallucination detectors

| Dataset | Module | How it decides "hallucinated" | Templates |
|---|---|---|---|
| ChessImages | `chess_validator.py` | Parse image → FEN (template match), reject boards failing `python-chess` legality | bundled in `templates/chess/` |
| Cards | `cards_validator.py` | Template-match each of the 4 quadrants; reject if worst match < threshold | ship with the Cards dataset (`Cards/templates`) |
| Shapes | `shapes_validator.py` | Rule-based white-pixel counts per region; reject blank / >max shapes | none |

```bash
# ChessImages (validity only)
python -m evaluation.chess_validator --gen-dir /path/to/gen/images

# ChessImages with FEN reconstruction accuracy against ground truth
python -m evaluation.chess_validator --gen-dir /path/gen --gt-json ChessImages/train_fen.json --conditional

# Cards (templates come with the dataset)
python -m evaluation.cards_validator --gen-dir /path/gen --template-dir /path/to/Cards/templates

# Shapes
python -m evaluation.shapes_validator --gen-dir /path/gen
```

Each detector is calibrated so that real (ground-truth) samples are flagged at a
rate of ~0%, i.e. detected hallucinations reflect the generator, not detector bias.

## Fidelity / diversity metrics

`compute_metrics.py` wraps the bundled `fld` library to report **FID** (Inception),
**CLIP-FID** (CLIP), **FLD**, and improved **precision/recall**.

```bash
python -m evaluation.compute_metrics \
    --train-dir /path/train --test-dir /path/test --gen-dir /path/gen \
    --metrics fid clip_fid fld precision recall
```

Requires `pip install -e ./fld` (see top-level README).
