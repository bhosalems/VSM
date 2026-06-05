#!/bin/bash
#SBATCH --job-name=debug_puython
#SBATCH --output=debug.log
#SBATCH --error=debug.err
#SBATCH --partition=a2il
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --nodelist=deepbull7


export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# source /home/csgrad/mbhosale/anaconda3/envs/ADM
source /home/csgrad/mbhosale/anaconda3/etc/profile.d/conda.sh
conda activate ADM

# Run your Python script
python /home/csgrad/mbhosale/phd/ddpm_hallucination/DDPM-IP/scripts/image_sample.py --image_size 128 --timestep_respacing 100 --model_path "/home/csgrad/mbhosale/phd/ddpm_hallucination/DDPM-IP/log/Hands_train1/ema_0.9999_200000.pt" --use_fp16 True --num_channels 256 --num_head_channels 64 --num_res_blocks 3 --attention_resolutions 32,16,8 --resblock_updown True --use_new_attention_order True --learn_sigma True --dropout 0.1 --diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True --rescale_learned_sigmas True --batch_size 1 --num_samples 10 --out_dir "/home/csgrad/mbhosale/phd/ddpm_hallucination/DDPM-IP/log/Hands_train1/inference7" --sample

conda deactivate