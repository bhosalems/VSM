import torch
import torch.nn.functional as F
from torch.autograd.functional import jvp
import argparse
from BetaVAE.model import BetaVAE_B
import os
from BetaVAE.dataset import return_data
import sys
import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    str2bool,
)
from guided_diffusion import dist_util, logger

@torch.no_grad()
def estimate_score_x0_from_eps_model(
    x0,                          # [B,C,H,W] in [0,1] (or whatever your diffusion expects)
    t,                           # int or tensor [B]
    eps_model,                   # callable: eps_model(xt, t) -> eps_hat
    alphas_bar,                  # 1D tensor [T] of alpha_bar
    K=16,                        # noise samples
):
    device = x0.device
    if isinstance(t, int):
        t = torch.full((x0.shape[0],), t, device=device, dtype=torch.long)
    
    if not isinstance(alphas_bar, torch.Tensor):
        alphas_bar = torch.from_numpy(alphas_bar).to(device)

    a = torch.sqrt(alphas_bar[t]).view(-1, 1, 1, 1)          
    sigma = torch.sqrt(1.0 - alphas_bar[t]).view(-1, 1, 1, 1) 

    scores = []
    for _ in range(K):
        eps = torch.randn_like(x0)
        xt = a * x0 + sigma * eps
        
        xt_input = xt.to(next(eps_model.parameters()).dtype)
        model_output = eps_model(xt_input, t)               # [B,C*2,H,W] if learn_sigma else [B,C,H,W]
        model_output = model_output.to(x0.dtype)             # Convert back to x0's dtype
        if model_output.shape[1] == x0.shape[1] * 2:
            eps_hat = model_output[:, :x0.shape[1]]         # First half is eps prediction
        else:
            eps_hat = model_output
        
        score_xt = -eps_hat / (sigma + 1e-12)              
        scores.append(score_xt)

    return torch.stack(scores, dim=0).mean(dim=0)            # [B,C,H,W] PAAS style proxy.

def latent_score_from_decoder(
    z,                     # [B,d], requires_grad maybe
    decoder_fn,            # decoder_fn(z)->x0 [B,C,H,W]
    g_x0,                  # [B,C,H,W]
    compute_geometry=True,
    geom_eps=1e-4,
):
    B, d = z.shape
    x_shape = g_x0.shape[1:]  # (C,H,W)
    n = g_x0[0].numel()
    create_graph = compute_geometry
    J_cols = []
    for i in range(d):
        v = torch.zeros_like(z)
        v[:, i] = 1.0
        _, Jv = jvp(decoder_fn, (z,), (v,), create_graph=create_graph)
        J_cols.append(Jv.reshape(B, n))  # [B,n]
    Jmat = torch.stack(J_cols, dim=-1)
    gx = g_x0.reshape(B, n)  # [B,n]
    
    if gx.dtype != Jmat.dtype:
        Jmat = Jmat.to(gx.dtype)
    term1 = torch.einsum("bn,bnd->bd", gx, Jmat)

    if not compute_geometry:
        term2 = torch.zeros_like(term1)
        return term1, term2, term1
    
    G = torch.einsum("bnd,bne->bde", Jmat, Jmat)
    I = torch.eye(d, device=z.device, dtype=z.dtype).unsqueeze(0)  # [1,d,d]
    G = G + geom_eps * I
    sign, logabsdet = torch.linalg.slogdet(G)  # sign should be +1 after eps
    geom_scalar = 0.5 * logabsdet.sum()        # sum over batch for one backward call
    term2 = torch.autograd.grad(geom_scalar, z, create_graph=False, retain_graph=False)[0]
    return term1, term2, term1 + term2


def latent_scores(x, encoder, decoder, eps_model, alphas_bar, t=100, K=16):
    distributions = encoder(x)
    mu = distributions[:, :distributions.shape[1]//2]
    logvar = distributions[:, distributions.shape[1]//2:]
    z = mu.detach()
    z.requires_grad_(True)

    def decoder_fn(z_in):
        x0 = torch.sigmoid(decoder(z_in))  
        if x0.shape[1] == 1:
            x0 = x0.repeat(1, 3, 1, 1)
        return x0
    x0 = decoder_fn(z)
    g_x0 = estimate_score_x0_from_eps_model(x0, t, eps_model, alphas_bar, K=K)
    term1, term2, total = latent_score_from_decoder(z, decoder_fn, g_x0, compute_geometry=True)

    return {
        "z": z.detach(),
        "x0": x0.detach(),
        "g_x0": g_x0.detach(),
        "latent_score_pullback": term1.detach(), 
        "latent_score_geometry": term2.detach(),  
        "latent_score_total": total.detach(),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='B', type=str, help='model type (H or B)')
    parser.add_argument("--vae_checkpoint", type=str, help="Path to the Beta-VAE checkpoint")
    parser.add_argument('--dataset', default='singleshapes', type=str, help='dataset name')
    parser.add_argument('--dset_dir', default='/a2il/data/mbhosale/Diffhaul/', type=str, 
                        help='dataset directory')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')
    parser.add_argument('--image_size', default=64, type=int, help='image size')
    parser.add_argument('--z_dim', default=10, type=int, help='latent dimension')
    parser.add_argument('--cuda', action='store_true', help='use cuda if available')
    parser.add_argument('--diffusion_checkpoint', type=str, nargs='+', required=True, help='path(s) to diffusion model checkpoint(s)')
    parser.add_argument('--rho', type=float, nargs='+', default=[0.1, 0.0], help='rho parameter(s) for diffusion (one per checkpoint or single value for all)')
    parser.add_argument('--timestep_respacing', type=str, default='1000', help='timestep respacing for diffusion')
    parser.add_argument('--learn_sigma', type=str2bool, nargs='+', default=[True, False], help='whether each model learns the denoising variance (one per checkpoint or single value for all)')
    parser.add_argument('--input_pertub', type=float, default=0.0, help='amount of input perturbation noise (0.0 disables)')
    parser.add_argument('--use_fp16', type=str2bool, default=True, help='enable fp16/mixed precision')
    parser.add_argument('--num_channels', type=int, default=256, help='base channel count of the U-Net')
    parser.add_argument('--num_head_channels', type=int, default=64, help='channels per attention head')
    parser.add_argument('--num_res_blocks', type=int, default=3, help='number of residual blocks per resolution')
    parser.add_argument('--attention_resolutions', type=str, default='32,16,8', help='comma-separated spatial resolutions to apply attention at (e.g., "32,16,8")')
    parser.add_argument('--resblock_updown', type=str2bool, default=True, help='use residual blocks for up/down sampling')
    parser.add_argument('--use_new_attention_order', type=str2bool, default=True, help='use new attention order implementation')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability')
    parser.add_argument('--diffusion_steps', type=int, default=1000, help='number of diffusion timesteps')
    parser.add_argument('--noise_schedule', type=str, default='cosine', choices=['linear', 'cosine'], help='beta/noise schedule')
    parser.add_argument('--use_scale_shift_norm', type=str2bool, default=True, help='use scale-shift normalization in resblocks')
    parser.add_argument('--rescale_learned_sigmas', type=str2bool, default=True, help='rescale learned sigmas (if applicable)')
    parser.add_argument('--schedule_sampler', type=str, default='loss-second-moment', help='schedule sampler type (e.g., "uniform", "loss-second-moment")')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads')
    parser.add_argument('--num_heads_upsample', type=int, default=-1, help='number of attention heads for upsampling layers')
    parser.add_argument('--channel_mult', type=str, default='', help='channel multiplier for each resolution')
    parser.add_argument('--class_cond', type=str2bool, default=False, help='whether the model is class-conditional')
    parser.add_argument('--use_checkpoint', type=str2bool, default=False, help='use gradient checkpointing to save memory')
    parser.add_argument('--use_kl', type=str2bool, default=False, help='use KL divergence loss instead of MSE')
    parser.add_argument('--predict_xstart', type=str2bool, default=False, help='predict x0 instead of noise')
    parser.add_argument('--rescale_timesteps', type=str2bool, default=False, help='rescale timesteps to [0,1]')
    parser.add_argument('--plot_pullback', action='store_true', help='plot pullback component of latent score')
    parser.add_argument('--plot_geometry', action='store_true', help='plot geometry component of latent score')
    parser.add_argument('--limit', type=float, default=10.0, help='traversal limit (e.g., 3.0 means [-3, 3])')
    parser.add_argument('--inter', type=float, default=0.25, help='traversal interval/step size between points')
    parser.add_argument('--image_idx', type=int, default=7, help='index of image from dataset to use for latent score computation (default: 7, matching inference.py)')
    parser.add_argument('--delay', type=int, default=10, help='delay between frames in GIFs (in 1/100 sec units, e.g., 100=1 sec/frame)')
    parser.add_argument('--use_log_scale', action='store_true', help='use log scale for y-axis (default: False)')
    parser.add_argument('--common_ylim', action='store_true', help='use common y-axis limits across all subplots to highlight dimensions with largest scores (default: False)')
    
    args = parser.parse_args()
    print("Arguments:", args)
    
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    net = BetaVAE_B(z_dim=args.z_dim, nc=1).to(device)
    if not os.path.isfile(args.vae_checkpoint):
        print(f"ERROR: Checkpoint not found at {args.vae_checkpoint}")
        exit(1)
    checkpoint = torch.load(args.vae_checkpoint, map_location=device)
    net.load_state_dict(checkpoint['model_states']['net'])
    net.eval()
    data_loader = return_data(args)
    
    # Load all diffusion models
    diffusion_checkpoints = args.diffusion_checkpoint if isinstance(args.diffusion_checkpoint, list) else [args.diffusion_checkpoint]
    
    # Handle rho and learn_sigma as lists (one per checkpoint or broadcast single value)
    rho_values = args.rho if isinstance(args.rho, list) else [args.rho]
    if len(rho_values) == 1 and len(diffusion_checkpoints) > 1:
        rho_values = rho_values * len(diffusion_checkpoints)
    
    learn_sigma_values = args.learn_sigma if isinstance(args.learn_sigma, list) else [args.learn_sigma]
    if len(learn_sigma_values) == 1 and len(diffusion_checkpoints) > 1:
        learn_sigma_values = learn_sigma_values * len(diffusion_checkpoints)
    
    if len(rho_values) != len(diffusion_checkpoints):
        raise ValueError(f"Number of rho values ({len(rho_values)}) must match number of checkpoints ({len(diffusion_checkpoints)})")
    if len(learn_sigma_values) != len(diffusion_checkpoints):
        raise ValueError(f"Number of learn_sigma values ({len(learn_sigma_values)}) must match number of checkpoints ({len(diffusion_checkpoints)})")
    
    models = []
    diffusions = []
    checkpoint_names = []
    
    for model_idx, model_path in enumerate(diffusion_checkpoints):
        # Create model-specific args
        model_args = args_to_dict(args, model_and_diffusion_defaults().keys())
        model_args['rho'] = rho_values[model_idx]
        model_args['learn_sigma'] = learn_sigma_values[model_idx]
        
        model, diffusion = create_model_and_diffusion(**model_args)
        
        # Load checkpoint and handle different formats
        checkpoint = dist_util.load_state_dict(model_path, map_location='cuda')
        
        # Check if checkpoint is a full training checkpoint with optimizer state
        if 'state' in checkpoint and 'param_groups' in checkpoint:
            # This is an optimizer checkpoint, skip it or raise error
            logger.log(f"ERROR: Checkpoint appears to be an optimizer state dict, not a model state dict")
            logger.log(f"Keys found: {list(checkpoint.keys())}")
            raise ValueError(f"Invalid checkpoint format for {model_path}. Expected model weights, got optimizer state.")
        elif 'model' in checkpoint or 'model_state_dict' in checkpoint:
            # Checkpoint contains model weights under a key
            state_dict = checkpoint.get('model', checkpoint.get('model_state_dict'))
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume the checkpoint itself is the state dict
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        logger.log(f"loading checkpoint {model_idx + 1}: {model_path} (rho={rho_values[model_idx]}, learn_sigma={learn_sigma_values[model_idx]})")
        model.to(dist_util.dev())
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()
        models.append(model)
        diffusions.append(diffusion)
        # Extract a short name for the checkpoint
        checkpoint_names.append(os.path.basename(os.path.dirname(model_path)) + '/' + os.path.basename(model_path).split('.')[0])
    
    logger.log(f"timesteps: {args.timestep_respacing}")
    
    # Get a fixed image from dataset (matching inference.py pattern)
    sample_image = data_loader.dataset.__getitem__(args.image_idx)
    sample_image = sample_image.unsqueeze(0).to(device)  # Add batch dimension
    
    print(f"Using image at index {args.image_idx} for latent score computation")
    
    # Get initial latent representation
    with torch.no_grad():
        distributions = net.encoder(sample_image)
        mu = distributions[:, :distributions.shape[1]//2]
        z_base = mu.detach()
    
    # Print actual latent values for debugging
    print(f"\nOriginal latent values (z_base):")
    print(f"  Shape: {z_base.shape}")
    print(f"  Values: {z_base[0].cpu().numpy()}")
    print(f"  Mean: {z_base[0].mean().item():.4f}, Std: {z_base[0].std().item():.4f}")
    print(f"  Min: {z_base[0].min().item():.4f}, Max: {z_base[0].max().item():.4f}")
    
    # Setup traversal parameters
    num_dims = z_base.shape[1]  # Should be 10
    limit = args.limit
    inter = args.inter
    
    print(f"\nTraversal range: [{-limit:.2f}, {limit:.2f}] with step {inter}")
    print(f"This means each dimension will be explored from (z_original - {limit}) to (z_original + {limit})")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    print("Computing latent scores across dimension traversals...")
    
    # Collect all scores for consistent scaling
    all_scores = []
    all_dim_scores = {}
    
    for dim_idx in range(num_dims):
        print(f"Processing dimension {dim_idx + 1}/{num_dims}...")
        
        traversal_values = torch.arange(-limit, limit + 0.1, inter)
        all_dim_scores[dim_idx] = []
        
        # Iterate over all diffusion models
        for model_idx, (model, diffusion, chkpt_name) in enumerate(zip(models, diffusions, checkpoint_names)):
            scores_total = []
            scores_pullback = [] if args.plot_pullback else None
            scores_geometry = [] if args.plot_geometry else None
            
            for val in traversal_values:
                # Create modified latent vector (using val as offset from original latent)
                z_modified = z_base.clone()
                z_modified[0, dim_idx] = z_base[0, dim_idx] + val
                z_modified.requires_grad_(True)
                
                # Compute score at this point
                def decoder_fn(z_in):
                    x0 = torch.sigmoid(net.decoder(z_in))
                    if x0.shape[1] == 1:
                        x0 = x0.repeat(1, 3, 1, 1)
                    return x0
                
                x0 = decoder_fn(z_modified)
                g_x0 = estimate_score_x0_from_eps_model(x0, t=250, eps_model=model, 
                                                         alphas_bar=diffusion.alphas_cumprod, K=16)
                term1, term2, total = latent_score_from_decoder(z_modified, decoder_fn, g_x0, 
                                                                compute_geometry=True)
                
                # Store scores for this dimension
                score_val = total[0, dim_idx].item()
                scores_total.append(score_val)
                # Collect scores for determining global range if common y-axis is enabled
                if args.common_ylim:
                    if args.use_log_scale:
                        all_scores.append(abs(score_val))  # Collect absolute values for log scale
                    else:
                        all_scores.append(score_val)  # Collect raw values for linear scale
                if args.plot_pullback:
                    scores_pullback.append(term1[0, dim_idx].item())
                if args.plot_geometry:
                    scores_geometry.append(term2[0, dim_idx].item())
            
            # Store for plotting
            all_dim_scores[dim_idx].append({
                'model_idx': model_idx,
                'chkpt_name': chkpt_name,
                'scores_total': scores_total,
                'scores_pullback': scores_pullback,
                'scores_geometry': scores_geometry
            })
    
    # Determine global y-axis limits for common scale across all subplots (if enabled)
    if args.common_ylim:
        if args.use_log_scale:
            all_scores = [s for s in all_scores if s > 0]
            if all_scores:
                global_min = min(all_scores)
                global_max = max(all_scores)
                y_min = global_min * 0.5
                y_max = global_max * 2.0
            else:
                y_min, y_max = 1e-6, 1.0
            print(f"Global score range (log scale): [{y_min:.2e}, {y_max:.2e}]")
        else:
            # Common y-axis limits for linear scale to highlight active dimensions
            if all_scores:
                global_min = min(all_scores)
                global_max = max(all_scores)
                # Add some padding for better visualization
                padding = (global_max - global_min) * 0.1
                y_min = global_min - padding
                y_max = global_max + padding
            else:
                y_min, y_max = -1.0, 1.0
            print(f"Global score range (linear scale): [{y_min:.2e}, {y_max:.2e}]")
    
    # Plot with appropriate scale
    for dim_idx in range(num_dims):
        ax = axes[dim_idx]
        
        # Define colors for different models
        colors = ['b', 'r', 'g', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # Plot stored scores
        for score_data in all_dim_scores[dim_idx]:
            model_idx = score_data['model_idx']
            chkpt_name = score_data['chkpt_name']
            scores_total = score_data['scores_total']
            scores_pullback = score_data['scores_pullback']
            scores_geometry = score_data['scores_geometry']
            
            # Select color for this model
            color = colors[model_idx % len(colors)]
            
            if args.use_log_scale:
                # Convert to absolute values for log scale
                scores_total_plot = [abs(s) if s != 0 else y_min for s in scores_total]
                ylabel = '|Latent Score| (log)'
            else:
                scores_total_plot = scores_total
                ylabel = 'Latent Score'
            
            # Plot components if requested
            if args.plot_pullback and scores_pullback:
                scores_pb_plot = [abs(s) if args.use_log_scale and s != 0 else s for s in scores_pullback]
                ax.plot(traversal_values, scores_pb_plot, color=color, linestyle='--', 
                       label=f'{chkpt_name} (Pullback)', linewidth=2.0, alpha=0.5)
            if args.plot_geometry and scores_geometry:
                scores_geom_plot = [abs(s) if args.use_log_scale and s != 0 else s for s in scores_geometry]
                ax.plot(traversal_values, scores_geom_plot, color=color, linestyle=':', 
                       label=f'{chkpt_name} (Geometry)', linewidth=2.0, alpha=0.5)
            
            # Always plot total score
            label = chkpt_name if len(models) > 1 or args.plot_pullback or args.plot_geometry else None
            ax.plot(traversal_values, scores_total_plot, color=color, linestyle='-', 
                   label=label, linewidth=2.5, alpha=0.65)
        
        # Set scale and limits
        if args.use_log_scale:
            ax.set_yscale('log')
            if args.common_ylim:
                ax.set_ylim(y_min, y_max)
        else:
            ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)
            # Apply common y-axis limits only if enabled
            if args.common_ylim:
                ax.set_ylim(y_min, y_max)
        
        ax.axvline(x=0, color='k', linestyle=':', alpha=0.3)
        ax.set_xlabel(f'z_{dim_idx + 1} value', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f'Dimension {dim_idx + 1}', fontsize=12, fontweight='bold')
        if len(models) > 1 or args.plot_pullback or args.plot_geometry:
            ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3, which='both' if args.use_log_scale else 'major')
    
    plt.tight_layout()
    
    # Create output directory based on VAE checkpoint path
    checkpoint_path = args.vae_checkpoint
    # Extract checkpoint name (e.g., 100000)
    checkpoint_name = os.path.basename(checkpoint_path)
    # Extract parent name (e.g., singleshapes_B_gamma100_z10_v6)
    parent_name = os.path.basename(os.path.dirname(checkpoint_path))
    
    # Find BetaVAE directory and replace 'checkpoints' with 'outputs'
    # checkpoint_path is like: /path/to/BetaVAE/checkpoints/singleshapes_B_gamma100_z10_v6/100000
    # We want: /path/to/BetaVAE/outputs/singleshapes_B_gamma100_z10_v6/100000
    checkpoint_dir = os.path.dirname(checkpoint_path)  # .../checkpoints/singleshapes_B_gamma100_z10_v6
    checkpoints_dir = os.path.dirname(checkpoint_dir)  # .../checkpoints
    betavae_dir = os.path.dirname(checkpoints_dir)  # .../BetaVAE
    output_dir = os.path.join(betavae_dir, 'outputs', parent_name, checkpoint_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the original image used for computation
    from torchvision.utils import save_image
    original_img_path = os.path.join(output_dir, 'original_image.png')
    save_image(sample_image.cpu(), original_img_path, normalize=True, pad_value=1)
    print(f"\nOriginal image saved as '{original_img_path}'")
    
    # Save plot
    output_path = os.path.join(output_dir, 'latent_scores_traversal.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{output_path}'")
    plt.close()
    
    # Generate GIF of latent traversals
    print("\nGenerating latent traversal GIF...")
    decoder = net.decoder
    
    gifs = []
    for dim_idx in range(num_dims):
        print(f"  Generating traversal for dimension {dim_idx + 1}/{num_dims}...")
        for val in traversal_values:
            z_modified = z_base.clone()
            z_modified[0, dim_idx] = z_base[0, dim_idx] + val
            
            with torch.no_grad():
                sample = F.sigmoid(decoder(z_modified)).data
            gifs.append(sample)
    
    # Organize gifs: [num_dims, num_steps, C, H, W]
    gifs = torch.cat(gifs)
    gifs = gifs.view(num_dims, len(traversal_values), gifs.shape[1], gifs.shape[2], gifs.shape[3])
    
    # Save individual frames and create GIF
    from torchvision.utils import save_image
    for j, val in enumerate(traversal_values):
        frame_path = os.path.join(output_dir, 'traversal_{:03d}.jpg'.format(j))
        save_image(tensor=gifs[:, j].cpu(),
                   fp=frame_path,
                   nrow=num_dims, pad_value=1, normalize=True)
    
    # Create GIF from frames
    frame_glob = os.path.join(output_dir, 'traversal_*.jpg')
    gif_path = os.path.join(output_dir, 'latent_traversal.gif')
    
    files = sorted(glob.glob(frame_glob))
    if files:
        frames = [Image.open(f) for f in files]
        duration_ms = args.delay * 10  # Convert to milliseconds
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False
        )
        print(f"GIF saved as '{gif_path}'")
    else:
        print("Warning: No frames generated for GIF")
    
    print("Finished!!")