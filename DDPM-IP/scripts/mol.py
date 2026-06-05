import torch
from rdkit import Chem
from molscribe import MolScribe
ckpt_path = "/home/csgrad/alone/diff_hallucinations/latent-diffusion/swin_base_char_aux_1m.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = MolScribe(ckpt_path, device=device)
output = model.predict_image_file('/home/csgrad/mbhosale/tmp/fold1/gdb_111164.png', return_atoms_bonds=True, return_confidence=True)

print(output)
smiles = output['smiles']
mol = Chem.MolFromSmiles(smiles)

if mol:
    print("Valid SMILE:", smiles)
else:
    print("Invalid SMILE!")

# import os
# import torch
# from rdkit import Chem
# from molscribe import MolScribe

# ckpt_path = "/home/csgrad/alone/diff_hallucinations/latent-diffusion/swin_base_char_aux_1m.pth"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Using device:", device)

# model = MolScribe(ckpt_path, device=device)

# image_dir = "/home/csgrad/alone/Data/Molecules-2D-new/check"

# for filename in os.listdir(image_dir):
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#         image_path = os.path.join(image_dir, filename)
#         print(f"Processing: {filename}")
#         try:
#             output = model.predict_image_file(image_path, return_atoms_bonds=True, return_confidence=True)
#             smiles = output.get('smiles', '')
#             mol = Chem.MolFromSmiles(smiles)
#             if mol:
#                 print(f"  ✅ Valid SMILES: {smiles}")
#             else:
#                 print(f"  ❌ Invalid SMILES!")
#         except Exception as e:
#             print(f"  ⚠️ Error processing {filename}: {e}")

