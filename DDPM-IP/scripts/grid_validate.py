import os
import glob
import shutil
import numpy as np
import cv2
from tqdm import tqdm

# ----------------------- CONFIGURATION -----------------------

BASE_DIR = "/home/csgrad/devulapa/phd/neurips_25/halmin/DDPM-IP/logs/cards_var_on_smooth_off/ema_0.9999_080000"
TEMPLATE_DIR_COLOR = "/home/csgrad/devulapa/phd/neurips_25/halmin/DDPM-IP/datasets/cards_data_gen/resized_templates_png"
TEMPLATE_DIR_GRAY = "/home/csgrad/devulapa/phd/neurips_25/halmin/DDPM-IP/datasets/cards_data_gen/resized_templates_png"

FILTER_NOISE = False
NOISE_THRESHOLD = 30
REMOVE_NOISED = False
USE_COLOR_TEMPLATE = False
SAVE_THRESHOLDS = [0.95]

OUTPUT_ROOT = os.path.join(BASE_DIR, "hall_results")

# -------------------------------------------------------------

# Clear hallucinated_cleaned folder if it exists
if os.path.exists(OUTPUT_ROOT):
    print(f"Removing existing folder: {OUTPUT_ROOT}")
    shutil.rmtree(OUTPUT_ROOT)
os.makedirs(OUTPUT_ROOT)

def load_templates(directory, use_color=False):
    paths = sorted(glob.glob(os.path.join(directory, "*.png")))
    if use_color:
        return [cv2.imread(p, cv2.IMREAD_COLOR) for p in paths]
    else:
        return [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]

def is_noised(tile, threshold=NOISE_THRESHOLD):
    return np.std(tile) < threshold

templates = load_templates(
    TEMPLATE_DIR_COLOR if USE_COLOR_TEMPLATE else TEMPLATE_DIR_GRAY,
    use_color=USE_COLOR_TEMPLATE
)

inference_folders = sorted(glob.glob(os.path.join(BASE_DIR, "inference_*/images")))
global_counts = {f"matched_{int(thresh * 100)}": 0 for thresh in SAVE_THRESHOLDS}
global_counts["unmatched"] = 0
global_counts["noised"] = 0
total_processed = 0
hallucination_counts = []

for folder in inference_folders:
    grid_paths = sorted(glob.glob(os.path.join(folder, "*.png")))
    folder_name = os.path.basename(os.path.dirname(folder))
    print(f"\nProcessing {folder_name} with {len(grid_paths)} images...")

    inf_out_root = os.path.join(OUTPUT_ROOT, folder_name)
    os.makedirs(inf_out_root, exist_ok=True)
    for thresh in SAVE_THRESHOLDS:
        os.makedirs(os.path.join(inf_out_root, f"matched_{int(thresh * 100)}"), exist_ok=True)
    unmatched_dir = os.path.join(inf_out_root, "unmatched")
    os.makedirs(unmatched_dir, exist_ok=True)
    if not REMOVE_NOISED:
        noised_dir = os.path.join(inf_out_root, f"noised_std{NOISE_THRESHOLD}")
        os.makedirs(noised_dir, exist_ok=True)

    local_counts = {f"matched_{int(thresh * 100)}": 0 for thresh in SAVE_THRESHOLDS}
    local_counts["unmatched"] = 0
    local_counts["noised"] = 0

    pbar = tqdm(total=len(grid_paths), desc=folder_name, unit="img")

    for grid_path in grid_paths:
        total_processed += 1
        img_color = cv2.imread(grid_path, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape
        tile_h, tile_w = h // 2, w // 2

        tiles = []
        clean = True
        for y in (0, tile_h):
            for x in (0, tile_w):
                tile = img_color[y:y + tile_h, x:x + tile_w] if USE_COLOR_TEMPLATE else img_gray[y:y + tile_h, x:x + tile_w]
                if FILTER_NOISE and is_noised(tile):
                    clean = False
                tiles.append(tile)

        file_name = os.path.basename(grid_path)

        if not clean:
            global_counts["noised"] += 1
            local_counts["noised"] += 1
            if not REMOVE_NOISED:
                shutil.copy(grid_path, os.path.join(noised_dir, file_name))
            pbar.update(1)
            continue

        tile_scores = []
        for tile in tiles:
            best_score = -1
            for tmpl in templates:
                if tile.shape != tmpl.shape:
                    continue
                score = cv2.matchTemplate(tile, tmpl, cv2.TM_CCOEFF_NORMED).max()
                if score > best_score:
                    best_score = score
            tile_scores.append(best_score)

        min_score = min(tile_scores)
        saved = False
        for thresh in sorted(SAVE_THRESHOLDS, reverse=True):
            if min_score >= thresh:
                out_dir = os.path.join(inf_out_root, f"matched_{int(thresh * 100)}")
                shutil.copy(grid_path, os.path.join(out_dir, file_name))
                global_counts[f"matched_{int(thresh * 100)}"] += 1
                local_counts[f"matched_{int(thresh * 100)}"] += 1
                saved = True
                break

        if not saved:
            shutil.copy(grid_path, os.path.join(unmatched_dir, file_name))
            global_counts["unmatched"] += 1
            local_counts["unmatched"] += 1

        pbar.update(1)

    pbar.close()

    hallucination_counts.append(local_counts["unmatched"])

    print(f"\n{folder_name} Summary")
    print(f"  Noised images             : {local_counts['noised']}")
    print(f"  Unmatched (Hallucinated)  : {local_counts['unmatched']}")
    for thresh in sorted(SAVE_THRESHOLDS, reverse=True):
        print(f"  Matched @ {int(thresh * 100)}%         : {local_counts[f'matched_{int(thresh * 100)}']}")

# Global summary
print("\n\nAll Inference Folders Processed")
print("Overall Hallucination Detection Summary")
print(f"Total images processed (including noised): {total_processed}")
print(f"  Noised/skipped               : {global_counts['noised']}")
print(f"  Unmatched (hallucinated)     : {global_counts['unmatched']}")

for thresh in sorted(SAVE_THRESHOLDS, reverse=True):
    key = f"matched_{int(thresh * 100)}"
    print(f"  Matched @ {int(thresh * 100)}%         : {global_counts[key]}")

matched_total = total_processed - global_counts["unmatched"] - global_counts["noised"]
print(f"\nTotal Clean + Matched        : {matched_total}")
print(f"Total Hallucinated (Unmatched): {global_counts['unmatched']} ({global_counts['unmatched'] / total_processed * 100:.2f}%)")

# Mean and std dev of hallucinations across folders
hallucination_counts = np.array(hallucination_counts)
print("\n\nHallucination Counts per Folder:", hallucination_counts)
mean_hall = hallucination_counts.mean()
std_hall = hallucination_counts.std()
print(f"\nMean hallucinations: {mean_hall:.2f}")
print(f"Std deviation      : {std_hall:.2f}")