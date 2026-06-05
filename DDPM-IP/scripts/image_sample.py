import argparse
import os

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

# import debugpy

# debugpy.listen(("0.0.0.0", 568))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client() 

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
    # Optionally enforce deterministic behavior (can slow down performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
def process_x0s(x0_preds, reverse=True):
    processed_x0s = []
    for x0_pred in x0_preds:
        x0_pred = ((x0_pred + 1) * 127.5).clamp(0, 255).to(th.uint8)
        x0_pred = x0_pred.permute(0, 2, 3, 1)
        x0_pred = x0_pred.contiguous()
        processed_x0s.append(x0_pred.cpu().numpy())
    if reverse: # DDPM-IP returns pred x0s in the order that we want reverse of. Please check suppl of https://arxiv.org/pdf/2406.09358
        processed_x0s = reversed(processed_x0s)
    processed_x0s = np.stack(processed_x0s, axis=1)
    return processed_x0s
 
def calc_hal(x0_preds, thr_pcntl, t_range, logdir, version=1):
    n_images = x0_preds.shape[0]
    n_timesteps = x0_preds.shape[1]
    assert t_range[0] <= t_range[1] <= n_timesteps, "Invalid Time range"
    sample_names = [f"sample_{i}" for i in range(n_images)]
    if version == 2:
        hal_metric = [] 
        x0_preds = x0_preds[:, t_range[0]:t_range[1], ...]
        x_bar = x0_preds.mean(axis=1)
        for j in range(n_images):
            diff_sum = np.zeros_like(x_bar[0])
            for i in range(t_range[1] - t_range[0]):
                diff_sum += (x0_preds[j, i, ...] - x_bar[j, ...])**2
            hal_metric.append(np.mean(diff_sum / (t_range[1] - t_range[0])))
    elif version == 1:
       time_variance = np.var(x0_preds[:, t_range[0]:t_range[1]+1, ], axis=1)
       hal_metric = np.mean(time_variance, axis=(1,2,3))
    hal_df = pd.DataFrame({"Sample Name": sample_names, "HAL Metric": hal_metric})
    hal_df.to_csv(os.path.join(logdir, "halmtr.csv"))
    return hal_metric

def main():
    args = create_argparser().parse_args()
    set_seed(args.seed)
    out_dir = args.out_dir
    imgs_out_dir = os.path.join(out_dir, "images")
    os.makedirs(out_dir, exist_ok=True)
    dist_util.setup_dist()
    logger.configure()
    t_range = [int(x) for x in args.t_range.split(",")]
    if args.sample:
        logger.log("creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(rho=args.rho,
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        model_path = args.model_path
        model.load_state_dict(
            dist_util.load_state_dict(model_path, map_location='cuda') #'map_location="cpu")
        )
        logger.log(f"loading checkpoint: {model_path}")
        logger.log(f"timesteps: {args.timestep_respacing}")
        model.to(dist_util.dev())
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()

        logger.log("sampling...")
        all_images = []
        all_labels = []
        all_x0_preds = None
        if args.hal_from_forward:
            # Inversion from the real sample with forward diffusion
            data = load_data(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                image_size=args.image_size,
                class_cond=args.class_cond,
            )
            i = 1
            for batch in data:
                if i * args.batch_size > args.num_samples:
                    break
                i+=1
                img = batch[0].to(dist_util.dev())
                model_kwargs = {}
                if args.class_cond:
                    classes = th.randint(
                        low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                    )
                    model_kwargs["y"] = classes
                indices = th.from_numpy(np.array([t_range[1]] * args.batch_size)).long().to(device=dist_util.dev())
                noised_img = diffusion.q_sample(img, indices)
                x0_preds = []
                for t in range(t_range[1], -1, -1):
                    out = diffusion.p_sample_loop_progressive_from_forward(
                        model, 
                        (args.batch_size, 3, args.image_size, args.image_size),
                        t,
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs,
                        noise_img=noised_img,
                        device=dist_util.dev())
                    sample, x0_pred = out['sample'], out['pred_xstart']
                    x0_preds.append(x0_pred)
                    noised_img = sample
                x0_preds = process_x0s(x0_preds)
                if all_x0_preds is not None:
                    all_x0_preds = np.concatenate((all_x0_preds, x0_preds), axis=0)
                else:
                    all_x0_preds = x0_preds
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()
                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
                if args.class_cond:
                    gathered_labels = [
                        th.zeros_like(classes) for _ in range(dist.get_world_size())
                    ]
                    dist.all_gather(gathered_labels, classes)
                    all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
                
                # if args.class_cond:
                #     label_arr = np.concatenate(all_labels, axis=0)
                #     label_arr = label_arr[: args.num_samples]
                # if dist.get_rank() == 0:
                #     shape_str = "x".join([str(x) for x in arr.shape])
                #     out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
                #     logger.log(f"saving to {out_path}")
                #     if args.class_cond:
                #         np.savez(out_path, arr, label_arr)
                #     else:
                #         np.savez(out_path, arr)
                #     os.makedirs(imgs_out_dir, exist_ok=True)
                    # for i, img_array in enumerate(arr):
                    #     img = Image.fromarray((img_array).astype(np.uint8))  # Convert to 8-bit pixel values
                    #     img.save(os.path.join(imgs_out_dir, f"sample_{i}.png"))
                logger.log(f"created {len(all_images) * args.batch_size} samples")       
        else:
                save_index   = 0                                
                if dist.get_rank() == 0:
                    os.makedirs(imgs_out_dir, exist_ok=True)
                    
                # Usual generation/inference path
                while len(all_images) * args.batch_size < args.num_samples:
                    model_kwargs = {}
                    if args.class_cond:
                        classes = th.randint(
                            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
                        )
                        model_kwargs["y"] = classes
                    sample_fn = (
                        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                    )
                    sample, x0_preds = sample_fn( # FIXME check if the final sample is included in x0_preds
                        model,
                        (args.batch_size, 3, args.image_size, args.image_size),
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs, return_x0_preds=True
                    )
                    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                    sample = sample.permute(0, 2, 3, 1).contiguous()
                    # x0_preds = process_x0s(x0_preds)

                    gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                    all_images.extend([g.cpu().numpy() for g in gathered_samples])
                    # if all_x0_preds is not None:
                        # all_x0_preds = np.concatenate((all_x0_preds, x0_preds), axis=0)
                    # else:
                        # all_x0_preds = x0_preds
                    
                    if dist.get_rank() == 0:
                        for local_idx, img_np in enumerate(sample.cpu().numpy()):
                            img = Image.fromarray(img_np)                        # (H,W,3) uint8
                            img.save(os.path.join(imgs_out_dir,
                                                f"sample_{save_index + local_idx}.png"))
                        save_index += sample.shape[0]
                    # if args.class_cond:
                    #     gathered_labels = [
                    #         th.zeros_like(classes) for _ in range(dist.get_world_size())
                    #     ]
                    #     dist.all_gather(gathered_labels, classes)
                    #     all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
                    logger.log(f"created {len(all_images) * args.batch_size} samples")

        # arr = np.concatenate(all_images, axis=0)
        # all_xpreds = np.concatenate(all_x0_preds, axis=0)
        # arr = arr[: args.num_samples]
        # all_x0_preds = all_x0_preds[: args.num_samples] # only get required numbers of samples/

        # Perform PCA on image data
        if args.pca:
            pca_image_space(arr, all_x0_preds, out_dir, args.num_samples)
            
        if args.calc_halmtr:
            print("Filtering images based on the hallucination filter..") # https://github.com/locuslab/diffusion-model-hallucination
            # all_x0_preds = np.random.randint(1, 11, size=(1, 3, 3, 3, 3))
            l = calc_hal(all_x0_preds, args.thr_pcntl, t_range, out_dir)

        dist.barrier()
        logger.log("sampling complete")
        
import plotly.graph_objects as go

def pca_image_space(arr, all_x0_preds, out_dir, num_samples):
    """
    Perform PCA on image space to identify outliers (hallucinations),
    save an interactive plot, and save principal components to a CSV file.
    """
    logger.log("Performing PCA on image space...")
    
    # Validate input shapes
    num_images, num_timesteps, c, h, w = all_x0_preds.shape
    assert arr.shape[0] == num_images, "Mismatch in number of images"
    
    # Flatten image data: (num_images * num_timesteps, flattened_image_vector)
    img_flattened = all_x0_preds.reshape(num_images * num_timesteps, -1)

    # Generate sample names and color labels
    sample_names = [
        f"sample_{i}_time_{j}.png" for i in range(num_images) for j in range(num_timesteps)
    ]
    sample_colors = [
        f"Sample {i}" for i in range(num_images) for j in range(num_timesteps)
    ] 

    # Apply PCA to reduce to 2 components
    logger.log("Applying PCA...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(img_flattened)

    # Save PCA results to a CSV file
    logger.log("Saving PCA results to CSV...")
    
    pca_data = {
        "Sample Name": sample_names,
        "Sample Color": sample_colors,
        "PCA Component 1": pca_result[:, 0],
        "PCA Component 2": pca_result[:, 1],
    }
    pca_df = pd.DataFrame(pca_data)
    csv_path = os.path.join(out_dir, "pca_results.csv")
    pca_df.to_csv(csv_path, index=False)
    logger.log(f"PCA results saved to {csv_path}")

    # Create a plotly figure
    logger.log("Creating and saving interactive PCA plot with trajectories...")
    fig = go.Figure()

    # Add scatter traces for each sample with lines connecting points
    for i in range(num_images):
        # Filter data for the current sample
        sample_indices = [j for j in range(len(sample_colors)) if sample_colors[j] == f"Sample {i}"]
        fig.add_trace(
            go.Scatter(
                x=pca_result[sample_indices, 0],
                y=pca_result[sample_indices, 1],
                mode="markers+lines",  # Add lines between points
                name=f"Sample {i}",  # Legend entry
                text=[sample_names[j] for j in sample_indices],  # Hover data
                marker=dict(size=10, opacity=0.6),  # Marker properties
                line=dict(width=2),  # Line properties
            )
        )

    # Update layout
    fig.update_layout(
        title="PCA of Image Space with Trajectories",
        xaxis_title="PCA Component 1",
        yaxis_title="PCA Component 2",
        legend_title="Samples",
    )

    # Save the interactive plot as an HTML file
    interactive_plot_path = os.path.join(out_dir, 'pca_image_space_interactive.html')
    fig.write_html(interactive_plot_path)
    logger.log(f"Interactive PCA plot with trajectories saved to {interactive_plot_path}")

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
        default="20,35",
        type=str, 
        help="Range of timesteps to consider for the filter"
    )
    parser.add_argument("--hal_from_forward", action="store_true", help="Use forward diffusion for hallucination filter")
    parser.add_argument("--data_dir", type=str, default="/a2il/data/mbhosale/Diffhaul/Hands/test/", help="Data directory of the text split")
    parser.add_argument("--seed", type=int, default=42)
    return parser

def plot_pca_variance(df_file, out_dir, num_samples, total_samples, selected_samples, number_of_step_groups=10):
    """
    Create bar plots of variance for PCA components.

    Args:
        df_file (str): Path to the input CSV file.
        out_dir (str): Directory where the output plots will be saved.
        num_samples (int): Number of samples to show in the plot.
        total_samples (int): Total number of available samples.
        selected_samples (list): List of manually selected sample indices.
        number_of_step_groups (int): Number of step groups to divide the range into.
    """
    pca_df = pd.read_csv(df_file)

    # Calculate the size of each step group
    steps_per_group = 100 // number_of_step_groups

    # Create a new column for step groups dynamically
    pca_df['Step Group'] = (
        pca_df['Sample Name'].str.extract(r'_time_(\d+).png')[0].astype(int) // steps_per_group
    )

    # Format selected_samples into "Sample X" format
    selected_samples = [f"Sample {i}" for i in selected_samples]

    # Total samples available
    all_samples = set(f"Sample {i}" for i in range(1, total_samples + 1))

    # Identify random samples to fill the remaining slots
    remaining_samples = list(all_samples - set(selected_samples))
    num_random_samples = max(0, num_samples - len(selected_samples))
    if num_random_samples > 0:
        random_samples = random.sample(remaining_samples, num_random_samples)
    else:
        random_samples = []

    # Combine selected and random samples
    combined_samples = selected_samples + random_samples

    # Filter the DataFrame for the combined samples
    filtered_df = pca_df[pca_df["Sample Color"].isin(combined_samples)]

    # Calculate variance grouped by Sample Color and Step Group
    variance_df = (
        filtered_df.groupby(["Step Group", "Sample Color"])[["PCA Component 1", "PCA Component 2"]]
        .var()
        .reset_index()
    )

    # Rename columns for clarity
    variance_df.columns = ["Step Group", "Sample Color", "Variance Component 1", "Variance Component 2"]

    # Create subplots for each step group
    step_groups = variance_df["Step Group"].unique()
    num_subplots = len(step_groups)

    # ------ Plotly Visualization ------
    fig = make_subplots(
        rows=num_subplots,
        cols=1,
        subplot_titles=[f"Step Group {int(step_group) * steps_per_group}–{int((step_group + 1) * steps_per_group - 1)}" for step_group in step_groups],
        vertical_spacing=0.1
    )

    for i, step_group in enumerate(step_groups):
        group_data = variance_df[variance_df["Step Group"] == step_group]

        # Data for bar plots
        bar_labels = []
        bar_values_c1 = []
        bar_values_c2 = []
        bar_colors = []

        for sample in combined_samples:
            sample_data = group_data[group_data["Sample Color"] == sample]
            if not sample_data.empty:
                bar_labels.append(sample)
                bar_values_c1.append(sample_data["Variance Component 1"].values[0])
                bar_values_c2.append(sample_data["Variance Component 2"].values[0])
                bar_colors.append("red" if sample in selected_samples else "black")

        # Add bar plots for Variance Component 1 and Component 2
        x = list(range(len(bar_labels)))  # X positions for the bars
        bar_width = 0.4
        fig.add_trace(
            go.Bar(
                x=[p - bar_width / 2 for p in x],
                y=bar_values_c1,
                name="Variance C1",
                marker=dict(color="lightblue"),
                showlegend=(i == 0)  # Only show legend once
            ),
            row=i + 1,
            col=1
        )
        fig.add_trace(
            go.Bar(
                x=[p + bar_width / 2 for p in x],
                y=bar_values_c2,
                name="Variance C2",
                marker=dict(color="lightgreen"),
                showlegend=(i == 0)  # Only show legend once
            ),
            row=i + 1,
            col=1
        )

        # Update x-axis with sample labels and colors
        fig.update_xaxes(
            tickvals=x,
            ticktext=[
                f'<span style="color:{color}">{label}</span>'
                for label, color in zip(bar_labels, bar_colors)
            ],
            tickangle=45,
            row=i + 1,
            col=1
        )

    fig.update_layout(
        height=400 * num_subplots,
        title_text="Variance of PCA Components by Step Groups and Selected Samples",
        barmode="group",
        showlegend=True
    )

    # Save the Plotly plot as an HTML file
    plotly_output_path = os.path.join(out_dir, f'pca_variance_{number_of_step_groups}_step_groups.html')
    fig.write_html(plotly_output_path)
    print(f"Interactive bar plot saved to {plotly_output_path}")

    # ------ Matplotlib Visualization ------
    fig, axes = plt.subplots(2, 5, figsize=(25, 10), sharey=False)
    fig.suptitle("Variance of PCA Components by Step Groups and Selected Samples", fontsize=20)
    axes = axes.flatten()

    for ax, step_group in zip(axes, step_groups):
        group_data = variance_df[variance_df["Step Group"] == step_group]

        # Data for bar plots
        bar_labels = []
        bar_values_c1 = []
        bar_values_c2 = []
        bar_colors = []

        for sample in combined_samples:
            sample_data = group_data[group_data["Sample Color"] == sample]
            if not sample_data.empty:
                bar_labels.append(sample)
                bar_values_c1.append(sample_data["Variance Component 1"].values[0])
                bar_values_c2.append(sample_data["Variance Component 2"].values[0])
                bar_colors.append("red" if sample in selected_samples else "black")

        # Create the bar plot
        x = range(len(bar_labels))
        bar_width = 0.35
        ax.bar(x, bar_values_c1, width=bar_width, label="Variance C1", color="lightblue")
        ax.bar([p + bar_width for p in x], bar_values_c2, width=bar_width, label="Variance C2", color="lightgreen")

        # Set x-ticks explicitly to align with the bars
        ax.set_xticks([p + bar_width / 2 for p in x])
        ax.set_xticklabels(bar_labels, rotation=45)

        # Set x-tick colors explicitly
        for tick_label, label in zip(ax.get_xticklabels(), bar_labels):
            tick_label.set_color("red" if label in selected_samples else "black")

        # Set subplot title and labels
        ax.set_title(f"Step Group {step_group * steps_per_group}–{(step_group + 1) * steps_per_group - 1}", fontsize=12)
        ax.set_xlabel("Samples")
        ax.legend()

    # Hide unused subplots if fewer than 10 step groups
    for ax in axes[len(step_groups):]:
        ax.set_visible(False)

    # Add shared y-axis label
    fig.text(0.04, 0.5, "Variance", va="center", rotation="vertical", fontsize=16)

    # Save the Matplotlib plot as a single PNG file
    matplotlib_output_path = os.path.join(out_dir, f"pca_variance_{number_of_step_groups}_step_groups.png")
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # Adjust layout to fit the title and axis labels
    plt.savefig(matplotlib_output_path)
    plt.close()

    print(f"Saved Matplotlib bar plot with 10 subplots to: {matplotlib_output_path}")
    
if __name__ == "__main__":
    main()
