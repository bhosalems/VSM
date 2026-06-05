import argparse
import os
from tqdm import tqdm
import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.image_datasets import load_data
from sklearn.decomposition import PCA
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import random
from matplotlib import pyplot as plt

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        out_dir="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--plot_pca_variance", action="store_true")
    parser.add_argument("--pca_df_file", type=str, help="Path to the PCA results CSV file")
    parser.add_argument("--pca", action="store_true", help="If we should do the PCA")
    parser.add_argument("--thr_pcntl", type=int, default=85, help="The percentile to use for the hallucination filter")
    parser.add_argument("--rho", type=float, default=0.01, help="Smoothness rho")
    parser.add_argument(
        "--calc_halmtr",
        default=False,
        action='store_true',
        help="Want to filter hallucinated images?",
    )
    parser.add_argument(
        "--t_range",
        default="0,999",
        type=str, 
        help="Range of timesteps to consider for the filter"
    )
    parser.add_argument("--hal_from_forward", action="store_true", help="Use forward diffusion for hallucination filter")
    parser.add_argument("--data_dir", type=str, default="/a2il/data/mbhosale/Diffhaul/Hands/train/", help="Data directory of the text split")
    parser.add_argument("--seed_list", default="12346,12347,12348,12349,12350,12351")
    return parser

def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def measure_progressive_score_difference(args, seed):
    set_seed(seed)
    out_dir = args.out_dir
    ckpt_tag = os.path.splitext(os.path.basename(args.model_path))[0]
    run_dir      = os.path.join(out_dir, ckpt_tag, f"inference{seed}")
    imgs_out_dir = os.path.join(run_dir, "images")

    os.makedirs(imgs_out_dir, exist_ok=True)
    dist_util.setup_dist()
    logger.log(f"==== running seed {seed} progressive====")
    model, diffusion = create_model_and_diffusion(
        rho=args.rho, **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    ckpt = dist_util.load_state_dict(args.model_path, map_location="cuda")
    model.load_state_dict(ckpt)
    model.to(dist_util.dev()).eval()
    if args.use_fp16: model.convert_to_fp16()

    data = load_data(
        data_dir  = args.data_dir,
        batch_size= args.batch_size,
        image_size= args.image_size,
        class_cond = args.class_cond,
        random_flip=False,
        deterministic=True
    )

    device     = dist_util.dev()
    alphas_bar = th.from_numpy(diffusion.alphas_cumprod).to(device)
    t0, t1     = [int(x) for x in args.t_range.split(",")]
    all_sample_mean_errors = []
    all_sample_norm_errors = []
    save_index   = 0   
    
    for i, batch in tqdm(enumerate(data, start=1)):
        if i * args.batch_size > args.num_samples:
            break
        x0 = batch[0].to(device)                     # ground‐truth clean images
        eps_T = th.randn_like(x0)
        t_batch = th.from_numpy(np.array([t1] * args.batch_size)).long().to(device=dist_util.dev())
        x_t   = diffusion.q_sample(x0, t_batch, noise=eps_T)
        norm_err_per_step = []
        mean_err_per_step = []
        
        for t in range(t1, max(t0-1, 0), -1):
            abar_t    = alphas_bar[t]                   # [B]
            sqrt_abar = abar_t.sqrt().view(-1,1,1,1)          # [B,1,1,1]
            sqrt_one  = (1-abar_t).sqrt().view(-1,1,1,1)      # [B,1,1,1]
            scale = -1.0 / th.sqrt(1.0 - abar_t)

            with th.no_grad():
                model_out = diffusion.p_sample_loop_progressive_from_forwardv2(
                        model, 
                        (args.batch_size, 3, args.image_size, args.image_size),
                        t,
                        clip_denoised=args.clip_denoised,
                        model_kwargs={},
                        noise_img=x_t,
                        device=device,
                        return_eps=True)
                eps_hat = model_out['eps']
                score_hat = eps_hat * scale 
            eps_true = (x_t - sqrt_abar * x0) / sqrt_one
            score_true = eps_true * scale
            norm_err_t = (score_hat - score_true).view(x0.size(0), -1).norm(dim=1)
            mean_err_t = th.sqrt((score_hat - score_true).view(x0.size(0), -1)**2).mean(axis=1) # rmse
            norm_err_per_step.append(norm_err_t.cpu().numpy())  # list of [B] arrays
            mean_err_per_step.append(mean_err_t.cpu().numpy())
            sample = model_out['sample']
            x_t = sample
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous()
        
        if dist.get_rank() == 0:
            for local_idx, img_np in enumerate(sample.cpu().numpy()):
                img = Image.fromarray(img_np)                        # (H,W,3) uint8
                img.save(os.path.join(imgs_out_dir,
                                    f"sample_{save_index + local_idx}.png"))
            save_index += sample.shape[0]
        
        logger.log(f"created {(i) * args.batch_size} samples")
        mean_errs = np.stack(mean_err_per_step, axis=1)  # shape [B, #steps]        
        all_sample_mean_errors.append(mean_errs.mean(axis=1))
        norm_errs = np.stack(norm_err_per_step, axis=1)  # shape [B, #steps]
        all_sample_norm_errors.append(norm_errs.mean(axis=1))


    all_sample_mean_errors = np.concatenate(all_sample_mean_errors, axis=0)[: args.num_samples]
    mean_prog_err = float(all_sample_mean_errors.mean())
    all_sample_norm_errors = np.concatenate(all_sample_norm_errors, axis=0)[: args.num_samples]
    norm_prog_err = float(all_sample_norm_errors.mean())
    logger.log(f"seed {seed} → progressive mean error {mean_prog_err:.6f}")
    logger.log(f"seed {seed} → progressive mean error {norm_prog_err:.6f}")
    return mean_prog_err, norm_prog_err


def measure_score_difference_for_seedv1(args, seed):
    """Run one full pass at a given seed, return the mean score-difference."""
    set_seed(seed)
    dist_util.setup_dist()
    logger.log(f"==== running seed {seed} ====")

    # build model
    model, diffusion = create_model_and_diffusion(
        rho=args.rho,
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location='cuda'))
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # data loader
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        random_flip=False,
        deterministic=True
    )

    # prepare
    device     = dist_util.dev()
    alphas_bar = th.from_numpy(diffusion.alphas_cumprod).to(device)
    t0, t1     = [int(x) for x in args.t_range.split(",")]
    all_diffs  = []

    # loop
    for i, batch in enumerate(data, start=1):
        if i * args.batch_size > args.num_samples:
            break
        img = batch[0].to(device)
        per_t_diffs = []
        for t in range(t1, t0, -1):
            t_batch = th.full((args.batch_size,), t, dtype=th.long, device=device)
            with th.no_grad():
                gt_noise    = th.randn_like(img)
                noised_img  = diffusion.q_sample(img, t_batch, noise=gt_noise)
                alpha_t     = alphas_bar[t_batch]

                model_out = model(noised_img, t_batch)
                # split variance if present
                if model_out.shape[1] == img.shape[1] * 2:
                    model_out, _ = th.split(model_out, img.shape[1], dim=1)

                scale        = -1.0 / th.sqrt(1.0 - alpha_t)
                score_approx = model_out * scale.view(-1,1,1,1)
                gt_score     = gt_noise * scale.view(-1,1,1,1)

                # per-sample L1 (or L2) difference
                diffs = (score_approx - gt_score).abs() \
                            .view(args.batch_size, -1) \
                            .mean(dim=1)      # [batch]
                per_t_diffs.append(diffs.cpu().numpy())

        # average over timesteps, then collect
        all_diffs.append(np.stack(per_t_diffs, axis=1).mean(axis=1))  # [batch]

    # flatten to [num_samples]
    all_diffs = np.concatenate(all_diffs, axis=0)[: args.num_samples]
    mean_diff = float(all_diffs.mean())
    logger.log(f"seed {seed} → mean score-difference = {mean_diff:.6f}")
    return mean_diff

def main():
    args = create_argparser().parse_args()

    # allow seeds as comma-list
    seed_list = [int(s) for s in args.seed_list.split(",")]
    per_seed_mean_errs = []
    per_seed_norm_errs = []
    
    for seed in tqdm(seed_list):
        # m = measure_score_difference_for_seedv1(args, seed)
        m, n = measure_progressive_score_difference(args, seed)
        per_seed_mean_errs.append(m)
        per_seed_norm_errs.append(n)

    overall_mean = np.mean(per_seed_mean_errs)
    overall_std  = np.std(per_seed_mean_errs, ddof=1)
    print("=== RESULTS over seeds", seed_list, "===")
    print(f"Per-seed means: {per_seed_mean_errs}")
    print(f"Norm overall mean    : {overall_mean:.6f}")
    print(f"Norm overall std dev : {overall_std:.6f}")
    
    print("per seed norm errs: ", per_seed_norm_errs)
    overall_mean = np.mean(per_seed_norm_errs)
    overall_std  = np.std(per_seed_norm_errs, ddof=1)
    print(f"Norm overall mean    : {overall_mean:.6f}")
    print(f"Norm overall std dev : {overall_std:.6f}")

if __name__ == "__main__":
    # add `--seed_list` to your parser defaults
    # e.g. parser.add_argument("--seed_list", default="1240,1241,1242")
    main()