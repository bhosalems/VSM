#!/bin/bash
export OPENAI_LOGDIR=/home/csgrad/mbhosale/phd/ddpm_hallucination/halmin/DDPM-IP/log/MNIST_var_off
export PYTHONPATH="$(cd "$(dirname "$0")/.."; pwd):${PYTHONPATH:-}"

mpirun -n 1 -env PYTHONPATH "$PYTHONPATH" python scripts/image_train.py  \
  --input_pertub 0.0 \
  --data_dir /data_local1/mbhosale/Diffhaul/mnist/train \
  --image_size 32 \
  --use_fp16 True \
  --num_channels 128 \
  --num_head_channels 32 \
  --num_res_blocks 3 \
  --attention_resolutions 16,8 \
  --resblock_updown True \
  --use_new_attention_order True \
  --learn_sigma False \
  --dropout 0.1 \
  --diffusion_steps 1000 \
  --noise_schedule cosine \
  --use_scale_shift_norm True \
  --rescale_learned_sigmas False \
  --schedule_sampler loss-second-moment \
  --lr 1e-4 \
  --batch_size 800 \
  --save_interval 10000 \
  --rho 0.0 \

  #screen 1982998.gpu1