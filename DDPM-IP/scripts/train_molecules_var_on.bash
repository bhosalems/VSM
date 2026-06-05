#!/bin/bash
export OPENAI_LOGDIR=/data_local1/mbhosale/Diffhaul/DDPM-IP/log/Molecules_UC_Variance_on/  # (bull7) /home/csgrad/mbhosale/phd/ddpm_hallucination/halmin/DDPM-IP/log/Molecules_UC_Variance_on/ #
export PYTHONPATH="$(cd "$(dirname "$0")/.."; pwd):${PYTHONPATH:-}"

mpirun -n 3 -env PYTHONPATH "$PYTHONPATH" python scripts/image_train.py  \
  --input_pertub 0.0 \
  --data_dir /data_local1/mbhosale/Diffhaul/Molecules-2D-new/train/ \
  --image_size 128 \
  --use_fp16 True \
  --num_channels 128 \
  --num_head_channels 32 \
  --num_res_blocks 3 \
  --attention_resolutions 16,8 \
  --resblock_updown True \
  --use_new_attention_order True \
  --learn_sigma True \
  --dropout 0.1 \
  --diffusion_steps 1000 \
  --noise_schedule cosine \
  --use_scale_shift_norm True \
  --rescale_learned_sigmas True \
  --schedule_sampler loss-second-moment \
  --lr 1e-4 \
  --batch_size 64 \
  --save_interval 10000 \
  --rho 0.0 \

#screen 3345434.gpu_3