#!/usr/bin/env bash
set -euo pipefail

# ── 1) List all of your checkpoint files here ────────────────────────────────
MODEL_PATHS=(
  "/home/csgrad/mbhosale/phd/ddpm_hallucination/halmin/DDPM-IP/log/Hands_sigma_False/ema_0.9999_110000.pt"
  "/home/csgrad/mbhosale/phd/ddpm_hallucination/halmin/DDPM-IP/log/Hands_sigma_False/ema_0.9999_130000.pt"
  "/home/csgrad/mbhosale/phd/ddpm_hallucination/halmin/DDPM-IP/log/Hands_sigma_False/ema_0.9999_150000.pt"
  # … add more checkpoints as needed …
)

# ── 2) Common args (everything except --model_path and --out_dir) ────────────
COMMON_ARGS=(
  --image_size        128
  --timestep_respacing 1000
  --use_fp16          True
  --num_channels      256
  --num_head_channels 64
  --num_res_blocks    3
  --attention_resolutions "32,16,8"
  --resblock_updown   True
  --use_new_attention_order True
  --learn_sigma       False
  --dropout           0.1
  --diffusion_steps   1000
  --noise_schedule    cosine
  --use_scale_shift_norm True
  --rescale_learned_sigmas False
  --batch_size        20
  --num_samples       100
  --data_dir          /data_local1/mbhosale/Diffhaul/Hands/train/
  --rho               0.0
  --sample
)

# ── 3) Loop over each checkpoint ─────────────────────────────────────────────
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  # derive a unique output directory per checkpoint, e.g. score_difference/ema_0.9999_160000
  BASENAME=$(basename "${MODEL_PATH}" .pt)
  OUT_DIR="/home/csgrad/mbhosale/phd/ddpm_hallucination/halmin/DDPM-IP/log/Hands_sigma_False/score_difference/"

  echo "=== Running checkpoint ${BASENAME} ===\n" >> "${OUT_DIR}/run_v3.log"

  python score_difference.py \
    "${COMMON_ARGS[@]}" \
    --model_path "${MODEL_PATH}" \
    --out_dir "${OUT_DIR}" \
    >> "${OUT_DIR}/run_v3.log"  2>&1
  echo
done

echo "All done!" >> "${OUT_DIR}/run_v3.log"
