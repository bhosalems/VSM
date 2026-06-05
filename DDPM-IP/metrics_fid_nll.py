#!/usr/bin/env python
"""
Approximate the bits/dimension for an image model and compute CleanFID.
"""

import argparse
import os
import numpy as np
import torch
import torch.distributed as dist
# from torchvision.datasets import ImageFolder
# from torchvision.transforms import Compose, Resize, ToTensor, Lambda
# from torch.utils.data import DataLoader

from cleanfid import fid as cleanfid  # CleanFID library
# from guided_diffusion import dist_util, logger
from guided_diffusion import logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def compute_cleanfid(real_folder, fake_folder, batch_size, mode="clean", feature_extractor='inception_v3'):  #"clip_vit_b_32"'):
    """
    Compute FID using CleanFID between two folders of images.
    """
    score = cleanfid.compute_fid(
        fake_folder,
        real_folder,
        mode=mode,
        batch_size=batch_size,
        num_workers=4,
        model_name=feature_extractor,
    )
    return score

def run_bpd_evaluation(model, diffusion, data, num_samples, clip_denoised):
    all_bpd = []
    all_metrics = {"vb": [], "mse": [], "xstart_mse": []}
    num_complete = 0
    world_size = dist.get_world_size()

    while num_complete < num_samples:
        batch, model_kwargs = next(data)
        x = batch.to(dist_util.dev())
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

        metrics = diffusion.calc_bpd_loop(
            model, x, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )

        # aggregate per-term metrics
        for name in all_metrics:
            term = metrics[name].mean(dim=0) / world_size
            dist.all_reduce(term)
            all_metrics[name].append(term.detach().cpu().numpy())

        # total bits-per-dim
        total_bpd = metrics["total_bpd"].mean() / world_size
        dist.all_reduce(total_bpd)
        all_bpd.append(total_bpd.item())

        num_complete += world_size * x.shape[0]
        logger.log(f"done {num_complete} samples: bpd={np.mean(all_bpd):.4f}")

    # # save breakdown of terms
    # if dist.get_rank() == 0:
    #     out_dir = logger.get_dir()
    #     for name, terms in all_metrics.items():
    #         arr = np.stack(terms, axis=0).mean(axis=0)
    #         path = os.path.join(out_dir, f"{name}_terms.npz")
    #         logger.log(f"saving {name} terms to {path}")
    #         np.savez(path, arr)

    dist.barrier()
    return np.mean(all_bpd)

def create_argparser():
    defaults = dict(
        # data paths & metrics folders
        data_dir= "/a2il/data/mbhosale/Diffhaul/Hands/train/", #/data_local1/mbhosale/Diffhaul/mnist/train/", #"/data_local1/mbhosale/Diffhaul/Hands/train/", #"/data_local1/mbhosale/Diffhaul/Chess/train_images/", #"/data_local1/mbhosale/Diffhaul/Hands/train/",
        real_folder= "/a2il/data/mbhosale/Diffhaul/Hands/train/", #"/data_local1/mbhosale/Diffhaul/Hands/train/", #"/data_local1/mbhosale/Diffhaul/Chess/train_images/",#,
        fake_folder="/home/csgrad/mbhosale/phd/ddpm_hallucination/halmin/latent-diffusion/logs/2025-09-04T11-59-06_hands-ldm-vq-f4-uc-l_smooth/inference/00012075", #"/home/csgrad/mbhosale/phd/ddpm_hallucination/halmin/DDPM-IP/log/Chess_UC_Variance_OFF/ema_0.9999_160000",
        num_samples=100,   # Number of samples to calculate the BPD
        batch_size=50,
        bpd=False, # this will give NLL
        clip_denoised=True,
        fid_mode="clean",
        feature_extractor='clip_vit_b_32', #'inception_v3',  #"clip_vit_b_32"'


        # diffusion/model settings (only required for NLL)
        image_size=256,
        timestep_respacing="250",
        model_path="/home/csgrad/mbhosale/phd/ddpm_hallucination/halmin/DDPM-IP/log/Chess_UC_smooth_rho0.1_V3/ema_0.9999_190000.pt",
        use_fp16=True,
        num_channels=256,
        num_head_channels=64,
        num_res_blocks=3,
        attention_resolutions="32,16,8",
        resblock_updown=True,
        use_new_attention_order=True,
        learn_sigma=True,
        dropout=0.1,
        diffusion_steps=1000,
        noise_schedule="cosine",
        use_scale_shift_norm=True,
        rescale_learned_sigmas=True,
        rho=0.1,
        
        # Some deafult parameters for the model which you likely have not chnaged
        num_heads=4,
        num_heads_upsample=-1,
        channel_mult="",
        class_cond=False,
        use_checkpoint=False,
        input_pertub=0.0,
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
    )
    # defaults.update(model_and_diffusion_defaults())
    print(defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def main():
    args = create_argparser().parse_args()
    # dist_util.setup_dist()
    logger.configure()

    if args.bpd:
        logger.log("Creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(rho=args.rho,
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        ckpt = dist_util.load_state_dict(args.model_path, map_location="cuda:0")
        model.load_state_dict(ckpt)
        model.to(dist_util.dev())
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()

        logger.log("Creating data loader for BPD evaluation...")
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            deterministic=True,
        )

        logger.log("Running BPD evaluation...")
        avg_bpd = run_bpd_evaluation(
            model, diffusion, data, args.num_samples, args.clip_denoised
        )
        if dist.get_rank() == 0:
            logger.log(f"Average bits/dim: {avg_bpd:.4f}")
        
    fid_score_list = []
    for fake_dir_i in os.listdir(args.fake_folder):
        fake_dir = os.path.join(args.fake_folder, fake_dir_i)
        if dist.get_rank() == 0:
            # Compute CleanFID if configured
            if fake_dir and args.fake_folder:
                logger.log("Computing CleanFID...")
                fid_score = compute_cleanfid(
                    args.real_folder, fake_dir,
                    batch_size=args.batch_size, mode=args.fid_mode,
                    feature_extractor=args.feature_extractor
                )
                logger.log(f"CleanFID ({args.fid_mode}): {fid_score:.4f}")
                print(f"CleanFID: {fid_score:.4f}")
                fid_score_list.append(fid_score)
            else:
                logger.log("Skipping CleanFID (real_folder or fake_folder not set)")

    dist.barrier()
    logger.log("Evaluation complete")
    if args.bpd:
        print(f"Mean BPD: {np.mean(avg_bpd):.4f}")
    print(f"Mean CleanFID: {np.mean(fid_score_list):.4f}")
    print(f"Standard Deviation CleanFID: {np.std(fid_score_list):.4f}")

if __name__ == "__main__":
    main()