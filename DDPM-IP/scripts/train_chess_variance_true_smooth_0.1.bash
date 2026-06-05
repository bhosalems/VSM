#!/bin/bash
export OPENAI_LOGDIR=/home/csgrad/mbhosale/phd/ddpm_hallucination/halmin/DDPM-IP/log/Chess_UC_smooth_rho0.1_75P
export PYTHONPATH="$(cd "$(dirname "$0")/.."; pwd):${PYTHONPATH:-}"

mpirun -n 2 -env PYTHONPATH "$PYTHONPATH" python scripts/image_train.py --data_dir /data_local1/mbhosale/Diffhaul/Chess/subset_75/ \
	--input_pertub 0.0  --image_size 256 --use_fp16 True --num_channels 256 --num_head_channels 64 \
	--num_res_blocks 3 --attention_resolutions 32,16,8 --resblock_updown True --use_new_attention_order \
	True --learn_sigma True --dropout 0.1 --diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True \
	--rescale_learned_sigmas True --schedule_sampler loss-second-moment --lr 1e-5 --batch_size 6 --rho 0.1 
	
	#--resume_checkpoint /home/csgrad/mbhosale/phd/ddpm_hallucination/halmin/DDPM-IP/log/Chess_UC_Variance_ON_V3/ema_0.9999_130000.pt \