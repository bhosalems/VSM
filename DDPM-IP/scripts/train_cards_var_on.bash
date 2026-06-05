#!/usr/bin/env python
export OPENAI_LOGDIR=/home/csgrad/devulapa/phd/neurips_25/halmin/DDPM-IP/logs/cards_var_off/
export PYTHON_FILE=/home/csgrad/devulapa/phd/neurips_25/halmin/DDPM-IP/scripts/image_train.py

python $PYTHON_FILE \
  --input_pertub 0.0 \
  --data_dir /home/csgrad/devulapa/phd/neurips_25/halmin/DDPM-IP/datasets/cards_data_gen/card_imgs/ \
  --image_size 128 \
  --use_fp16 True \
  --num_channels 256 \
  --num_head_channels 64 \
  --num_res_blocks 3 \
  --attention_resolutions 32,16,8 \
  --resblock_updown True \
  --use_new_attention_order True \
  --learn_sigma False \
  --dropout 0.1 \
  --diffusion_steps 1000 \
  --noise_schedule cosine \
  --use_scale_shift_norm True \
  --rescale_learned_sigmas False \
  --schedule_sampler loss-second-moment \
  --lr 1e-5 \
  --batch_size 24 \
  --save_interval 10000 \
  --gpu_id 5 \
  --rho 0.0 \
#   --resume_checkpoint /home/csgrad/devulapa/phd/neurips_25/halmin/DDPM-IP/logs/small_mnist_var_off/ema_0.9999_100000.pt \