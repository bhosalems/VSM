#!/bin/bash
export OPENAI_LOGDIR=/home/csgrad/mbhosale/phd/ddpm_hallucination/halmin/DDPM-IP/log/Cat_Dog_UC_Variance_ON_V4
export PYTHONPATH="$(cd "$(dirname "$0")/.."; pwd):${PYTHONPATH:-}"
# export CUDA_VISIBLE_DEVICES=1,2,3

# mpirun -n 2 -env PYTHONPATH "$PYTHONPATH" python scripts/image_train.py --data_dir /data_local1/mbhosale/Diffhaul/PetImages/ \
# 	--input_pertub 0.0  --image_size 256 --use_fp16 True --num_channels 256 --num_head_channels 64 \
# 	--num_res_blocks 3 --attention_resolutions 32,16,8 --resblock_updown True --use_new_attention_order \
# 	True --learn_sigma True --dropout 0.1 --diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True \
# 	--rescale_learned_sigmas True --schedule_sampler loss-second-moment --lr 1e-5 --batch_size 6 --rho 0.0 \

mpirun -n 2 -env PYTHONPATH "$PYTHONPATH" python scripts/image_train.py --data_dir /home/csgrad/mbhosale/phd/ddpm_hallucination/halmin/latent-diffusion/datasets/train/ \
	--input_pertub 0.0  --image_size 256 --use_fp16 True --num_channels 256 --num_head_channels 64 \
	--num_res_blocks 3 --attention_resolutions 32,16,8 --resblock_updown True --use_new_attention_order \
	True --learn_sigma True --dropout 0.1 --diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True \
	--rescale_learned_sigmas True --schedule_sampler loss-second-moment --lr 1e-5 --batch_size 2 --rho 0.0 \