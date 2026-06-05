"""Training-free hallucination detector for the ChessImages dataset.

A generated 256x256 chessboard is parsed back to a FEN piece-placement string
via template matching, and its legality is checked with ``python-chess``
(``chess.Board.is_valid()``). A board that fails legality (e.g. two white kings,
>8 pawns, pawns on the back rank, ...) is counted as *hallucinated*.

This mirrors the "ChessImages" detection module described in the paper
(Section 5.1 / Appendix E). Only ``<PiecePlacement>`` is recovered from the
image, so rules that need the rest of a FEN (castling / en-passant / side-to-move
checks) are not used by the legality test.

Usage
-----
    python -m evaluation.chess_validator \
        --gen-dir /path/to/generated/images \
        [--gt-json train_fen.json --conditional] \
        [--template-dir evaluation/templates/chess] \
        [--threshold 0.50]

``--gen-dir`` may be passed multiple times (e.g. one folder per seed); the
hallucination rate is reported per folder and aggregated (mean +/- std).
"""
import argparse
import json
import os
from difflib import SequenceMatcher

import chess
import cv2
import numpy as np
from tqdm import tqdm

# Bundled 32x32 piece templates ship alongside this file.
DEFAULT_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates", "chess")

# Map python-chess status flags to a human-readable violation name. Used to
# bucket hallucinated boards by the rule they break (see paper Appendix E).
STATUS_FOLDERS = {
    chess.STATUS_NO_WHITE_KING: "NO_WHITE_KING",
    chess.STATUS_NO_BLACK_KING: "NO_BLACK_KING",
    chess.STATUS_TOO_MANY_KINGS: "TOO_MANY_KINGS",
    chess.STATUS_TOO_MANY_WHITE_PAWNS: "TOO_MANY_WHITE_PAWNS",
    chess.STATUS_TOO_MANY_BLACK_PAWNS: "TOO_MANY_BLACK_PAWNS",
    chess.STATUS_PAWNS_ON_BACKRANK: "PAWNS_ON_BACKRANK",
    chess.STATUS_TOO_MANY_WHITE_PIECES: "TOO_MANY_WHITE_PIECES",
    chess.STATUS_TOO_MANY_BLACK_PIECES: "TOO_MANY_BLACK_PIECES",
    chess.STATUS_BAD_CASTLING_RIGHTS: "BAD_CASTLING_RIGHTS",
    chess.STATUS_INVALID_EP_SQUARE: "INVALID_EP_SQUARE",
    chess.STATUS_OPPOSITE_CHECK: "OPPOSITE_CHECK",
    chess.STATUS_EMPTY: "EMPTY",
    chess.STATUS_RACE_CHECK: "RACE_CHECK",
    chess.STATUS_RACE_OVER: "RACE_OVER",
    chess.STATUS_RACE_MATERIAL: "RACE_MATERIAL",
    chess.STATUS_TOO_MANY_CHECKERS: "TOO_MANY_CHECKERS",
    chess.STATUS_IMPOSSIBLE_CHECK: "IMPOSSIBLE_CHECK",
}

PIECE_MAP = {"king": "K", "queen": "Q", "rook": "R", "bishop": "B", "knight": "N", "pawn": "P"}


def load_templates(template_dir, target_size=(32, 32)):
    """Load piece templates keyed by (square color -> FEN char -> [images]).

    Template files are named ``{piece}_{pc}{sc}.png`` where ``pc`` is the piece
    color (w/b) and ``sc`` the square color (w/b), e.g. ``king_ww.png``.
    """
    templates = {"b": {}, "w": {}}

    def process_image(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        if img.shape != target_size:
            img = cv2.resize(img, target_size)
        return img

    loaded = 0
    for fname in os.listdir(template_dir):
        if not fname.endswith(".png"):
            continue
        try:
            piece, colors = fname[:-4].split("_")
            if len(colors) != 2:
                continue
            piece_color, square_color = colors[0].lower(), colors[1].lower()
            if piece.lower() not in PIECE_MAP or piece_color not in "wb" or square_color not in "wb":
                continue
            fen_char = PIECE_MAP[piece.lower()]
            fen_char = fen_char.upper() if piece_color == "w" else fen_char.lower()
            template = process_image(os.path.join(template_dir, fname))
            if template is not None:
                templates[square_color].setdefault(fen_char, []).append(template)
                loaded += 1
        except ValueError:
            continue

    if loaded != 24:
        raise ValueError(f"Loaded {loaded}/24 templates from {template_dir}. Check filenames/format.")
    return templates


def image_to_fen(image_path, templates, confidence_threshold=0.50, img_size=256):
    """Reconstruct the FEN piece-placement string from a rendered board image."""
    board_img = cv2.imread(image_path)
    if board_img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    board_img = cv2.resize(board_img, (img_size, img_size))
    gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)

    square_size = img_size // 8
    fen_rows = []
    for rank in reversed(range(8)):
        fen_row, empty_count = [], 0
        for file in range(8):
            y, x = (7 - rank) * square_size, file * square_size
            square = gray[y:y + square_size, x:x + square_size]
            square_color = "b" if (file + rank) % 2 == 0 else "w"

            best_match = ("", -1)
            for fen_char, char_templates in templates[square_color].items():
                for template in char_templates:
                    result = cv2.matchTemplate(square, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    if max_val > best_match[1]:
                        best_match = (fen_char, max_val)

            if best_match[1] >= confidence_threshold:
                if empty_count:
                    fen_row.append(str(empty_count))
                    empty_count = 0
                fen_row.append(best_match[0])
            else:
                empty_count += 1
        if empty_count:
            fen_row.append(str(empty_count))
        fen_rows.append("".join(fen_row))
    return "/".join(fen_rows)


def is_fuzzy_match(extracted, gt, threshold=0.95):
    return SequenceMatcher(None, extracted, gt).ratio() >= threshold


def validate_dirs(images_dirs, template_dir=DEFAULT_TEMPLATE_DIR, gt_json=None,
                  conditional=False, fuzzy_threshold=0.8, confidence_threshold=0.50,
                  save_buckets=False):
    """Validate every image folder and return a metrics dict.

    Set ``conditional=True`` together with ``gt_json`` (a ``{name: FEN}`` map) to
    additionally report exact / fuzzy FEN reconstruction accuracy against the
    ground truth. Otherwise only validity / hallucination rate is reported.
    """
    templates = load_templates(template_dir)
    ground_truth = json.load(open(gt_json)) if gt_json else {}

    metrics = {"valid_acc": [], "hallucination": []}
    if conditional:
        metrics["exact_acc"] = []
        metrics[f"{fuzzy_threshold * 100:g}%_match_acc"] = []

    for images_dir in images_dirs:
        if not os.path.isdir(images_dir):
            print(f"[warn] not a directory, skipping: {images_dir}")
            continue
        files = [f for f in os.listdir(images_dir) if f.lower().endswith(".png")]
        if conditional:
            files = [f for f in files if os.path.splitext(f)[0] in ground_truth]

        matches = valid_count = fuzzy_count = total = 0
        for fname in tqdm(files, desc=os.path.basename(images_dir.rstrip("/")) or "images"):
            path = os.path.join(images_dir, fname)
            try:
                fen = image_to_fen(path, templates, confidence_threshold)
                if conditional:
                    gt_fen = ground_truth[os.path.splitext(fname)[0]].split()[0]
                    matches += int(fen == gt_fen)
                    fuzzy_count += int(is_fuzzy_match(fen, gt_fen, fuzzy_threshold))

                board = chess.Board(fen)
                if board.is_valid():
                    valid_count += 1
                elif save_buckets:
                    status_mask = board.status()
                    for flag, folder in STATUS_FOLDERS.items():
                        if status_mask & flag:
                            rule_dir = os.path.join(images_dir, "hallucinated", folder)
                            os.makedirs(rule_dir, exist_ok=True)
                            cv2.imwrite(os.path.join(rule_dir, fname), cv2.imread(path))
                total += 1
            except Exception as e:  # noqa: BLE001 - keep going on a bad file
                print(f"[warn] error on {fname}: {e}")

        if total == 0:
            continue
        metrics["valid_acc"].append(100 * valid_count / total)
        metrics["hallucination"].append(100 - 100 * valid_count / total)
        if conditional:
            metrics["exact_acc"].append(100 * matches / total)
            metrics[f"{fuzzy_threshold * 100:g}%_match_acc"].append(100 * fuzzy_count / total)

    print("\n=== ChessImages validation ===")
    for k, v in metrics.items():
        print(f"  {k}: {np.mean(v):.2f} +/- {np.std(v):.2f}  (per-folder: {[round(x, 2) for x in v]})")
    return metrics


def main():
    p = argparse.ArgumentParser(description="ChessImages hallucination detector (FEN legality).")
    p.add_argument("--gen-dir", action="append", required=True,
                   help="Folder of generated PNG boards. Repeat for multiple seeds.")
    p.add_argument("--template-dir", default=DEFAULT_TEMPLATE_DIR,
                   help="Piece templates (defaults to bundled evaluation/templates/chess).")
    p.add_argument("--gt-json", default=None, help="Optional {name: FEN} ground-truth map.")
    p.add_argument("--conditional", action="store_true",
                   help="Also report exact/fuzzy FEN reconstruction accuracy (requires --gt-json).")
    p.add_argument("--fuzzy-threshold", type=float, default=0.8)
    p.add_argument("--threshold", type=float, default=0.50, help="Template-match confidence threshold.")
    p.add_argument("--save-buckets", action="store_true",
                   help="Save hallucinated boards into per-rule subfolders under each gen-dir.")
    args = p.parse_args()

    validate_dirs(args.gen_dir, args.template_dir, args.gt_json, args.conditional,
                  args.fuzzy_threshold, args.threshold, args.save_buckets)


if __name__ == "__main__":
    main()
