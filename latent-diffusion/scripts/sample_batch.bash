#!/usr/bin/env bash
#
# Run sample_diffusion.py for 6 different seeds, 100 samples each.

# Path to the checkpoint
SCRIPT_DIR=/home/csgrad/mbhosale/phd/ddpm_hallucination/halmin/latent-diffusion
CKPT="/home/csgrad/mbhosale/phd/ddpm_hallucination/halmin/latent-diffusion/logs/2025-07-24T17-12-00_hands-ldm-vq-f4-ocmdpm/tmp/epoch=000090.ckpt"

# Common arguments
N_SAMPLES=100
# N_SAMPLES=2500
DATA_DIR="" # only matters for Chess
SPLIT="train" # only matters for Chess
DATASET="Hands"

# List of seeds to loop over
SEEDS=(12346 12347 12348 12349 12350 12351)
# SEEDS=(47) #TODO commth this and uncomment above line to sample the images in the setting discussed earlier.

cd "$SCRIPT_DIR"
for seed in "${SEEDS[@]}"; do
  echo "=========================================="
  echo " Running sample_diffusion.py with seed $seed "
  echo "=========================================="
  python -m scripts.sample_diffusion \
    --n_samples  $N_SAMPLES \
    --resume     "$CKPT" \
    --data_dir   "$DATA_DIR" \
    --split      "$SPLIT" \
    --dataset    "$DATASET" \
    --seed       "$seed"
  echo
done
