import os
import torch
from rdkit import Chem
from molscribe import MolScribe
import statistics

CKPT_PATH = "/home/csgrad/alone/diff_hallucinations/latent-diffusion/swin_base_char_aux_1m.pth"
BASE_FOLDER = "/home/csgrad/mbhosale/tmp"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MolScribe model
model = MolScribe(CKPT_PATH, device=DEVICE)

# Function to check validity of a single image
def is_valid_image(img_path):
    out = model.predict_image_file(img_path,
                                    return_atoms_bonds=True,
                                    return_confidence=True)
    smiles = out.get('smiles', '')
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

# Get immediate subfolders (6 seed folders)
subfolders = [d for d in os.listdir(BASE_FOLDER)
              if os.path.isdir(os.path.join(BASE_FOLDER, d))]

# Collect hallucination counts and proportions
halluc_counts = []
halluc_props = []

for sub in subfolders:
    folder_path = os.path.join(BASE_FOLDER, sub)
    total = 0
    invalid = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                total += 1
                if not is_valid_image(os.path.join(root, file)):
                    print(os.path.join(root, file))
                    invalid += 1
    halluc_counts.append(invalid)
    halluc_props.append(invalid / total if total else 0)
    print(f"Folder '{sub}': {invalid}/{total} invalid ⇒ prop = {invalid/total:.3f}")

# Compute mean and SD across folders
mean_count = statistics.mean(halluc_counts)
std_count = statistics.stdev(halluc_counts)
mean_prop = statistics.mean(halluc_props)
std_prop = statistics.stdev(halluc_props)

print(f"\nMean number of hallucinations: {mean_count:.2f} ± {std_count:.2f}")
print(f"Mean hallucination proportion: {mean_prop:.3f} ± {std_prop:.3f}")
