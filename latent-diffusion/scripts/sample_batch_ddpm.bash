#!/usr/bin/env bash
#
# Run sample_diffusion.py for 6 different seeds, 100 samples each.

# Path to the checkpoint
SCRIPT_DIR=/home/csgrad/mbhosale/phd/ddpm_hallucination/halmin/latent-diffusion
CKPT="/home/csgrad/mbhosale/phd/ddpm_hallucination/halmin/latent-diffusion/logs/2026-02-09T20-11-06_chess_finetune_from_ckpt/checkpoints/epoch=000012.ckpt"
# CKPT="/home/csgrad/mbhosale/phd/ddpm_hallucination/halmin/latent-diffusion/logs/2026-01-19T20-53-49_hands-ldm-vq-f4-c-l_smooth_finetune/checkpoints/last.ckpt"
# Common arguments
N_SAMPLES=100

# Matters only for conditional sampling where you are not using the fixed prompt and have per image prompts.
DATA_DIR="/data_local1/mbhosale/Diffhaul/Chess/"
SPLIT="train" #"train"
DATASET="Chess"

# List of seeds to loop over
SEEDS=(12346 12347 12348 12349 12350 12351)

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
    --seed       "$seed" \
    --vanilla_sample \
    --batch_size 100 \
  echo
done
