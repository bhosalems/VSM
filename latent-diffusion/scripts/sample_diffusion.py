import argparse, os, sys, glob, datetime, yaml
sys.path.insert(0, "/home/csgrad/mbhosale/phd/ddpm_hallucination/halmin/latent-diffusion")
# sys.path.append(os.getcwd())
import torch
import time
import numpy as np
from tqdm import trange
import json 

from omegaconf import OmegaConf
from PIL import Image

from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import random
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import pandas as pd


rescale = lambda x: (x + 1.) / 2.

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, condtion=None, return_intermediates=True,
               verbose=True,
               make_prog_row=False):


    if not make_prog_row:
        return model.p_sample_loop(condtion, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model,batch_size, condition=None, vanilla=False, custom_steps=None, eta=1.0, return_inter=False):


    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape, condtion=condition,
                                         make_prog_row=False)
        else:
            sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                    eta=eta)
        t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    if return_inter:
        log["intermediates"] = intermediates
    print(f'Throughput for this batch: {log["throughput"]}')
    return log

def process_x0s(x0_preds, model=None, decode=False):
    processed_x0s = []
    for x0_pred in x0_preds:
        if decode:
            x0_pred = model.decode_first_stage(x0_pred)
            x0_pred = ((x0_pred + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        x0_pred = x0_pred.permute(0, 2, 3, 1)
        x0_pred = x0_pred.contiguous()
        processed_x0s.append(x0_pred.cpu().numpy())
    processed_x0s = np.stack(processed_x0s, axis=1)
    return processed_x0s

def pca_image_space(arr, all_x0_preds, out_dir, num_samples):
    """
    Perform PCA on image space to identify outliers (hallucinations),
    save an interactive plot, and save principal components to a CSV file.
    """
    print("Performing PCA ...")
    
    # Validate input shapes
    num_images, num_timesteps, c, h, w = all_x0_preds.shape
    assert arr.shape[0] == num_images, "Mismatch in number of images"
    
    img_flattened = all_x0_preds.reshape(num_images * num_timesteps, -1)

    # Generate sample names and color labels
    sample_names = [
        f"sample_{i}_time_{j}.png" for i in range(num_images) for j in range(num_timesteps)
    ]
    sample_colors = [
        f"Sample {i}" for i in range(num_images) for j in range(num_timesteps)
    ] 

    # Apply PCA to reduce to 2 components
    print("Applying PCA...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(img_flattened)

    # Save PCA results to a CSV file
    print("Saving PCA results to CSV...")
    
    pca_data = {
        "Sample Name": sample_names,
        "Sample Color": sample_colors,
        "PCA Component 1": pca_result[:, 0],
        "PCA Component 2": pca_result[:, 1],
    }
    pca_df = pd.DataFrame(pca_data)
    csv_path = os.path.join(out_dir, "pca_results.csv")
    pca_df.to_csv(csv_path, index=False)
    print(f"PCA results saved to {csv_path}")

    # Create a plotly figure
    print("Creating and saving interactive PCA plot with trajectories...")
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
        title="PCA with Trajectories",
        xaxis_title="PCA Component 1",
        yaxis_title="PCA Component 2",
        legend_title="Samples",
    )

    # Save the interactive plot as an HTML file
    interactive_plot_path = os.path.join(out_dir, 'pca_image_space_interactive.html')
    fig.write_html(interactive_plot_path)
    print(f"Interactive PCA plot with trajectories saved to {interactive_plot_path}")

def make_hal_plots(df_file, out_dir, halmtr_file, hallucinated_samples, n_hallucinated_samples,
                               number_of_steps=100, number_of_step_groups=10, thr_pcntl=85):
    """
    Create bar plots of variance for each PCA component in separate subplots.

    Args:
        df_file (str): Path to the input CSV file.
        out_dir (str): Directory where the output plots will be saved.
        hallucinated_samples (list): Hallucinated samples.
        n_hallucinated_samples (list): Non-hallucinated samples.
        number_of_steps (int): Total DDIM steps.
        number_of_step_groups (int): Number of step groups (bins).
    """
    # Read the CSV
    pca_df = pd.read_csv(df_file)

    # Calculate how many steps per group
    steps_per_group = number_of_steps // number_of_step_groups

    # Create a 'Step Group' column from the filename pattern
    pca_df['Step Group'] = (
        pca_df['Sample Name'].str.extract(r'_time_(\d+).png')[0].astype(int)
        // steps_per_group
    )

    # Convert the sample indices into "Sample X" strings
    hallucinated_samples = [f"Sample {i}" for i in hallucinated_samples]
    n_hallucinated_samples = [f"Sample {i}" for i in n_hallucinated_samples]
    combined_samples = hallucinated_samples + n_hallucinated_samples

    # Filter only the relevant samples
    filtered_df = pca_df[pca_df["Sample Color"].isin(combined_samples)]

    # Calculate variance grouped by Step Group & Sample Color
    variance_df = (
        filtered_df.groupby(["Step Group", "Sample Color"])[["PCA Component 1", "PCA Component 2"]]
        .var()
        .reset_index()
    )
    # Rename for clarity
    variance_df.columns = ["Step Group", "Sample Color", "Variance C1", "Variance C2"]

    # Unique step groups
    step_groups = variance_df["Step Group"].unique()
    num_subplots = len(step_groups)

    # -------------- PLOTLY --------------
    # We make 2 columns: Column 1 for C1, Column 2 for C2
    fig = make_subplots(
        rows=num_subplots,
        cols=2,
        subplot_titles=[
            f"Group {sg}: C1" for sg in step_groups
        ] + [
            f"Group {sg}: C2" for sg in step_groups
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )

    # We will store legends only on the first row
    show_legend_c1 = True
    show_legend_c2 = True

    for i, step_group in enumerate(step_groups):
        group_data = variance_df[variance_df["Step Group"] == step_group]

        # Gather data for bars
        bar_labels = []
        bar_vals_c1 = []
        bar_vals_c2 = []
        bar_colors = []

        for sample in combined_samples:
            sample_data = group_data[group_data["Sample Color"] == sample]
            if not sample_data.empty:
                bar_labels.append(sample)
                bar_vals_c1.append(sample_data["Variance C1"].values[0])
                bar_vals_c2.append(sample_data["Variance C2"].values[0])
                # color "red" if hallucinated, else "black"
                bar_colors.append("red" if sample in hallucinated_samples else "black")

        # x positions
        x_positions = list(range(len(bar_labels)))

        # ---- Column 1: Variance C1 ----
        fig.add_trace(
            go.Bar(
                x=x_positions,
                y=bar_vals_c1,
                name="Variance C1" if show_legend_c1 else "",
                marker=dict(color="lightblue"),
                text=bar_labels,
                showlegend=show_legend_c1
            ),
            row=i+1,
            col=1
        )
        show_legend_c1 = False  # Only show legend once

        # Customize X-axis ticks for C1
        fig.update_xaxes(
            tickvals=x_positions,
            ticktext=[
                f'<span style="color:{c}">{lbl}</span>'
                for lbl, c in zip(bar_labels, bar_colors)
            ],
            tickangle=45,
            row=i+1,
            col=1
        )
        fig.update_yaxes(title_text="Variance", row=i+1, col=1)

        # ---- Column 2: Variance C2 ----
        fig.add_trace(
            go.Bar(
                x=x_positions,
                y=bar_vals_c2,
                name="Variance C2" if show_legend_c2 else "",
                marker=dict(color="lightgreen"),
                text=bar_labels,
                showlegend=show_legend_c2
            ),
            row=i+1,
            col=2
        )
        show_legend_c2 = False

        # Customize X-axis ticks for C2
        fig.update_xaxes(
            tickvals=x_positions,
            ticktext=[
                f'<span style="color:{c}">{lbl}</span>'
                for lbl, c in zip(bar_labels, bar_colors)
            ],
            tickangle=45,
            row=i+1,
            col=2
        )
        fig.update_yaxes(title_text="Variance", row=i+1, col=2)

        # Step group subtitle
        start_step = step_group * steps_per_group
        end_step = (step_group + 1) * steps_per_group - 1
        fig.layout.annotations[i].text = f"Step Group {start_step}-{end_step}, C1"
        fig.layout.annotations[i + num_subplots].text = f"Step Group {start_step}-{end_step}, C2"

    fig.update_layout(
        height=300 * num_subplots,
        width=1200,
        barmode="group",
        title_text="Variance of PCA Components",
        showlegend=True
    )

    plotly_out_path = os.path.join(out_dir, f"pca_variance_{number_of_step_groups}_step_groups.html")
    fig.write_html(plotly_out_path)
    print(f"Saved Plotly plot: {plotly_out_path}")

    # -------------- MATPLOTLIB --------------
    # We'll create (num_subplots x 2) subplots
    fig, axes = plt.subplots(nrows=num_subplots, ncols=2,
                             figsize=(14, 4*num_subplots), sharey=False)
    fig.suptitle("Variance of PCA Components", fontsize=16)

    # If there's only 1 row (e.g., num_subplots=1), axes is not a 2D array
    if num_subplots == 1:
        axes = np.array([axes])  # make it 2D for consistent indexing

    for i, step_group in enumerate(step_groups):
        group_data = variance_df[variance_df["Step Group"] == step_group]

        bar_labels = []
        bar_vals_c1 = []
        bar_vals_c2 = []
        bar_colors = []

        for sample in combined_samples:
            sample_data = group_data[group_data["Sample Color"] == sample]
            if not sample_data.empty:
                bar_labels.append(sample)
                bar_vals_c1.append(sample_data["Variance C1"].values[0])
                bar_vals_c2.append(sample_data["Variance C2"].values[0])
                bar_colors.append("red" if sample in hallucinated_samples else "black")

        x_positions = np.arange(len(bar_labels))

        # LEFT column: PCA Component 1
        ax_left = axes[i, 0]
        ax_left.bar(x_positions, bar_vals_c1, color="lightblue")
        ax_left.set_xticks(x_positions)
        ax_left.set_xticklabels(bar_labels, rotation=45, ha="right")
        for tick_label, c in zip(ax_left.get_xticklabels(), bar_colors):
            tick_label.set_color(c)
        ax_left.set_ylabel("Variance")
        start_step = step_group * steps_per_group
        end_step = (step_group + 1) * steps_per_group - 1
        ax_left.set_title(f"Step Group {start_step}-{end_step}, C1")

        # RIGHT column: PCA Component 2
        ax_right = axes[i, 1]
        ax_right.bar(x_positions, bar_vals_c2, color="lightgreen")
        ax_right.set_xticks(x_positions)
        ax_right.set_xticklabels(bar_labels, rotation=45, ha="right")
        for tick_label, c in zip(ax_right.get_xticklabels(), bar_colors):
            tick_label.set_color(c)
        ax_right.set_ylabel("Variance")
        ax_right.set_title(f"Step Group {start_step}-{end_step}, C2")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path_matplotlib = os.path.join(out_dir, f"pca_variance_{number_of_step_groups}_step_groups.png")
    plt.savefig(out_path_matplotlib, dpi=150)
    plt.close()
    print(f"Saved Matplotlib plot: {out_path_matplotlib}")
    
    #Hallucination metrics plots
    haldf = pd.read_csv(halmtr_file)
    plt.figure(figsize=(8,6))
    plt.bar(haldf["Sample Name"][:50], haldf["HAL Metric"][:50], color="skyblue", edgecolor="black")
    plt.title(f"Histogram of Hallucination-Metric", fontsize=14) # we choose only 50 such images for now
    plt.xlabel("Sample Name")
    plt.ylabel("Hal Metric")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join("/".join(out_dir.split("/")[:-1]), "halmtr.png"), dpi=150)
    plt.close()
    
    threshold = haldf["HAL Metric"].quantile(thr_pcntl / 100.0)
    return haldf[haldf["HAL Metric"] > threshold].copy()["Sample Name"]

def calc_hal(x0_preds, thr_pcntl, t_range, logdir):
    hal_metric = []
    n_images = x0_preds.shape[0]
    n_timesteps = x0_preds.shape[1]
    assert t_range[0] <= t_range[1] <= n_timesteps, "Invalid Time range" 
    x0_preds = x0_preds[:, t_range[0]:t_range[1], ...]
    x_bar = x0_preds.mean(axis=1)
    sample_names = [f"sample_{i}" for i in range(n_images)]
    for j in range(n_images):
        diff_sum = np.zeros_like(x_bar[0])
        for i in range(t_range[1] - t_range[0]):
            diff_sum += (x0_preds[j, i, ...] - x_bar[j, ...])**2
        hal_metric.append(np.mean(diff_sum / (t_range[1] - t_range[0])))        
    hal_df = pd.DataFrame({"Sample Name": sample_names, "HAL Metric": hal_metric})
    hal_df.to_csv(os.path.join(logdir, "halmtr.csv"))
    return hal_metric
    
def run(model, logdir, pca_dir, batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=50000, 
        pca=False, scale=0.75, H=256, W=256, decode=True, 
        calc_halmtr=False, thr_pcntl=85, t_range=[20, 35], dataset='Hands', data_dir="", split="test",
        save_trajectory=False, prompt_tuning=False):
    if vanilla:
        # print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
        return_inter = False
    else:
        print(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')
        return_inter = True

    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir,'*.png')))-1
    # path = logdir
    all_images = []
    all_x0_preds = None
    if vanilla:
        print(f"Running unconditional sampling for {n_samples} samples")
        if model.cond_stage_model is not None:
            if dataset =="Hands":
                if not prompt_tuning:
                    condition = "Close-up high quality image of a human hand."
                else:
                    assert NotImplementedError("Prompt tuning not implemented for unconditional sampling.")
                condition = model.get_learned_conditioning(batch_size * [condition])
            elif dataset == "Mnist":
                if not prompt_tuning:
                    digit = lambda: random.choice(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"])
                    condition = [digit() for i in range(batch_size)]  
                else:
                    assert NotImplementedError("Prompt tuning not implemented for unconditional sampling.")
                condition = model.get_learned_conditioning(condition)
        else:
            condition = None
        for _ in trange(n_samples // batch_size, desc="Sampling Batches with DDPM"):
            logs = make_convolutional_sample(model, condition=condition,  batch_size=batch_size,
                                             vanilla=vanilla, custom_steps=custom_steps,
                                             eta=eta, return_inter=return_inter)
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            all_images.extend([custom_to_np(logs["sample"])])
            # x0_preds = process_x0s(logs["intermediates"], model, decode=decode)
            # if all_x0_preds is not None:
            #     all_x0_preds = np.concatenate((all_x0_preds, x0_preds), axis=0)
            # else:
            #     all_x0_preds = x0_preds
            
            if n_saved >= n_samples:
                print(f'Finish after generating {n_saved} samples')
                break
        # shape_str = "x".join([str(x) for x in all_img.shape])
        # nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        # np.savez(nppath, all_img)
    elif model.cond_stage_model is not None and not vanilla:
        sampler = DDIMSampler(model)
        save_prompts = False
        if dataset == "Hands":
            if not prompt_tuning:
                prompt_i = "Close-up high quality image of a human hand."
            else:
                tuning_prompts = lambda: random.choice([
                        "High-resolution photo of a human hand, palm fully open with five fingers (thumb, index, middle, ring, pinky) spread naturally, plain white background.",
                        "Close-up shot of an open human palm showing all five fingers in correct thumb-to-pinky order, flat facing the camera, on white.",
                        "Photograph of a human hand with palm wide open, five straight fingers (thumb - index - middle - ring - little finger), against a white backdrop.",
                        "Studio image of an open palm displaying five fingers in proper sequence—thumb at left, pinky at right—on a clean white background.",
                        "Realistic photo of a single human palm, five fingers fully extended in thumb-to-pinky order, flat and facing forward, white background.",
                        "High-quality image of a human hand, palm completely open, five fingers aligned anatomically (thumb, index, middle, ring, pinky), white backdrop.",
                        "Close-up of an open palm with five straight fingers, thumb on the left and pinky on the right, on solid white.",
                        "Photorealistic shot of a fully opened palm showing five fingers in correct order, flat against a white background.",
                        "Sharp photo of a human hand, palm fully extended with thumb, index, middle, ring, and little finger visible in order, white background.",
                        "Clean studio portrait of an open palm—five fingers (thumb through pinky) splayed evenly—on a white backdrop.",
                        "High-resolution image of an open palm with five anatomically ordered fingers, thumb first then index, middle, ring, and pinky, against white.",
                        "Close-up studio photo of a human palm fully open, showing five straight fingers in thumb-to-pinky sequence, white background.",
                        "Real-life shot of an open hand with palm facing camera, five fingers (thumb - index - middle - ring - little) in order, white backdrop.",
                        "Crisp image of an open palm with five fingers aligned anatomically, thumb on the left edge, pinky on the right, plain white background.",
                        "Photograph of a human palm flat and facing forward, five fingers visible in correct anatomical order, white background.",
                        "Studio-style image of an open hand—five fingers from thumb to pinky—fully extended and flat against white.",
                        "Close-up of a human palm with five distinct fingers, starting from thumb then index, middle, ring, little, on a white backdrop.",
                        "Detailed photo of an open palm showing five fingers in sequence, thumb at outer edge, pinky at other, on solid white.",
                        "High-detail shot of a human palm fully opened, five straight fingers in anatomical order, flat and white background.",
                        "Clear photo of a human hand, palm fully open with thumb, index, middle, ring, and pinky fingers visible in order on a white background."])
            save_prompts = True
        elif dataset == "Chess":
            fen = pd.DataFrame(json.load(open(os.path.join(data_dir, split+"_fen.json"))).items(), columns=['img', 'fen'])
        elif dataset == "Mnist":
            if not prompt_tuning:
                digit = lambda: random.choice(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"])
            else:
                digit = lambda: random.choice(
                        [# Zero
                        "MNIST-style handwritten 'zero': thin white strokes, centered on a clean black background, no extra marks.",
                        "MNIST-style handwritten 'zero': minimal white loop, centered on black, uniform thickness, no noise.",
                        # One
                        "MNIST-style handwritten 'one': single thin white vertical stroke, centered on black, no stray pixels.",
                        "MNIST-style handwritten 'one': clean white digit one, straight line, centered on black, isolated.",
                        # Two
                        "MNIST-style handwritten 'two': crisp white strokes, centered on black, no overlapping or smudges.",
                        "MNIST-style handwritten 'two': clear white digit two, centered on black, uniform lines, no noise.",
                        # Three
                        "MNIST-style handwritten 'three': two smooth thin white strokes, centered on black, no extra artifacts.",
                        "MNIST-style handwritten 'three': neat white digit three, centered on black, distinct curves, clean.",
                        # Four
                        "MNIST-style handwritten 'four': intersecting thin white strokes, centered on black, no stray marks.",
                        "MNIST-style handwritten 'four': crisp white digit four, centered on black, clear junctions.",
                        # Five
                        "MNIST-style handwritten 'five': clear thin white strokes, centered on black, no overlapping lines.",
                        "MNIST-style handwritten 'five': sharp white digit five, centered on black, isolated strokes.",
                        # Six
                        "MNIST-style handwritten 'six': continuous thin white stroke, centered on black, no breaks.",
                        "MNIST-style handwritten 'six': clean white digit six, rounded form, centered on black, no noise.",
                        # Seven
                        "MNIST-style handwritten 'seven': two thin white strokes, centered on black, no extra marks.",
                        "MNIST-style handwritten 'seven': neat white digit seven, centered on black, uniform thickness.",
                        # Eight
                        "MNIST-style handwritten 'eight': two distinct thin white loops, centered on black, no distortions.",
                        "MNIST-style handwritten 'eight': symmetric white digit eight, centered on black, clear separation.",
                        # Nine
                        "MNIST-style handwritten 'nine': thin white strokes, centered on black, isolated and clean.",
                        "MNIST-style handwritten 'nine': crisp white digit nine, centered on black, no extra pixels."])
            save_prompts = True
        else:
            assert NotImplementedError(f"Dataset {dataset} not implemented for sampling.")    
        base_count = len(os.listdir(logdir))
        os.makedirs(logdir, exist_ok=True)
        if save_prompts:
            prompt_dir = os.path.join("/".join(logdir.split("/")[:-1]), "prompts")
            os.makedirs(os.path.join(prompt_dir), exist_ok=True)
        if save_trajectory:
            traj_dir = os.path.join("/".join(logdir.split("/")[:-1]), "trajectories")
            os.makedirs(traj_dir, exist_ok=True)
            base_count1 = len(os.listdir(traj_dir))
        fnames = []
        
        with torch.no_grad():
            with model.ema_scope():
                uc = None
                if scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [""])
                for n in trange(n_samples // batch_size, desc="Sampling"):
                    if dataset == "Hands":
                        if not prompt_tuning:
                            prompt = batch_size * [prompt_i]
                        else:
                            prompt = [tuning_prompts() for i in range(batch_size)]
                    elif dataset == "Chess":
                        prompt = list(fen.iloc[n*batch_size:(n+1)*batch_size]["fen"].apply(lambda x: x.split()[0]))
                        fnames = list(fen.iloc[n*batch_size:(n+1)*batch_size]["img"])
                    elif dataset == "Mnist":
                        prompt = [digit() for i in range(batch_size)]  
                    c = model.get_learned_conditioning(prompt)
                    shape = [3, H//4, W//4]
                    samples_ddim, intermediates = sampler.sample(S=custom_steps,
                                                    conditioning=c,
                                                    batch_size=batch_size,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc,
                                                    eta=eta,
                                                    log_every_t=1)
                    # intermediates['pred_x0'] = intermediates['pred_x0'][1:]
                    # intermediates['x_inter'] = intermediates['x_inter'][1:] 
                    sampled_images = model.decode_first_stage(samples_ddim)
                    sampled_images = torch.clamp((sampled_images+1.0)/2.0, min=0.0, max=1.0)

                    k=0
                    for x_sample in sampled_images:
                        if len(fnames) !=0:
                            fname = fnames.pop(0)
                            text_name = fnames + ".txt"
                            fname = fname + ".png"
                        else:
                            fname = f"{base_count:04}.png"
                            text_name = f"{base_count:04}.txt"
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(logdir, fname))
                        if save_prompts:
                            with open(os.path.join(prompt_dir, text_name), "w", encoding="utf-8") as f:
                                f.write(prompt[k])
                        base_count += 1
                        k+=1
                    all_images.extend(sampled_images.cpu().numpy())
                    
                    # x0_preds = process_x0s(intermediates['pred_x0'], model=model, decode=decode)
                    # if all_x0_preds is not None:
                    #     all_x0_preds = np.concatenate((all_x0_preds, x0_preds), axis=0)
                    # else:
                    #     all_x0_preds = x0_preds
                    # if save_trajectory:
                    #     for x0_pred in x0_preds: # saves intermediate x0 predictions all the timesteps for all the images being generated
                    #         torch.save(x0_pred, (os.path.join(traj_dir, f"{base_count1:04}.pt")))
                    #         base_count1 += 1

        # all_img = np.concatenate(all_images, axis=0) # here the sizes are weird, but we dont care as much at the moment.
        # all_img = all_img[:n_samples]
        # all_x0_preds = all_x0_preds[:n_samples]
        # perform PCA on image data
        # os.makedirs(pca_dir, exist_ok=True)
        # if pca:
            # pca_image_space(all_img, all_x0_preds, pca_dir, n_samples)
            
        if calc_halmtr:
            print("Filtering images based on the hallucination filter..") # https://github.com/locuslab/diffusion-model-hallucination
            l = calc_hal(all_x0_preds, thr_pcntl, t_range, "/".join(logdir.split("/")[:-1]))
        n_saved = n_samples 
        
    print(f"Your samples are ready and waiting four you here: \n{logdir} \nEnjoy.")
    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50000
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=20
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        nargs="?",
        help="Unconditional guidance scale",
        default=1.25
    )
    parser.add_argument(
        "--pca_df_file",
        type=str,
        default=None,
        help="Path to PCA results CSV file",
    )
    parser.add_argument(
        "--calc_halmtr",
        default=False,
        action='store_true',
        help="Want to filter hallucinated images?",
    )
    parser.add_argument(
        "--thr_pcntl",
        type=float,
        default=85,
        help="Threshold for pixel variance over timesteps"
    )
    parser.add_argument(
        "--t_range",
        default=[20, 35],
        type=list, 
        help="Range of timesteps to consider for the filter"
    )
    parser.add_argument(
        "--pca",
        default=False,
        action='store_true',
        help="Perform PCA to plot the trajectories"
        )
    parser.add_argument(
        "--halmtr_file",
        type=str,
        default="",
        help="Hallucination metric file"
        )
    parser.add_argument(
        "--hal_plots",
        default=False,
        action='store_true',
        help="Plots hallucination related plots - varince of PCA, Trajectories, Hallucination metrics"
        )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Test data directory for the text conditional for conditional sampling, for example")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Hands",
        help="Dataset for sampling.")
    parser.add_argument(
        "--split",
        type=str,
        default='test',
        help="Split of the dataset tos ample from"
        )
    parser.add_argument(
        "--save_trajectory",
        default=False,
        action='store_true',
        help="If all the intermittent pred_x0s should be saved"
        )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed to be set for the inference"
    )
    parser.add_argument(
        "--prompt_tuning",
        default=False,
        action='store_true',
        help="If prompt tuning is used for the Hands dataset"
    )
    return parser


def load_model_from_config(config, sd, device):
    if not config.params.cond_stage_config == "__is_unconditional__" :
        config.params.cond_stage_config.params["device"] = device
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def load_model(config, ckpt, device, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location=device)
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"], device)

    return model, global_step

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
        
if __name__ == "__main__":
    
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None
    set_seed(opt.seed)
    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume.split('/')[:-2])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    if opt.resume.endswith(".ckpt"):
        config_dir = os.path.join("/".join(opt.resume.split("/")[:-2]), "configs")
        base_configs = sorted(os.listdir(config_dir))
        base_configs = [os.path.join(config_dir, cfg) for cfg in base_configs]
    else:
        base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    device = "cuda:0"
    eval_mode = True

    if opt.logdir != "none":
        # locallog = logdir.split(os.sep)[-1]
        # if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{opt.logdir}'")
        logdir = os.path.join(opt.logdir)
    
    if opt.n_samples > 0: # if you want to sample
        print(config)
        model, global_step = load_model(config, ckpt, device, eval_mode)
        print(f"global step: {global_step}")
        print(75 * "=")
        print("logging to:")
        print(logdir)
        logdir = os.path.join(logdir, "inference", f"{global_step:08}", str(opt.seed))
        pca_dir = os.path.join(logdir, "pca")
        imglogdir = os.path.join(logdir, "images")
        os.makedirs(imglogdir, exist_ok=True)
        # write config out
        sampling_file = os.path.join(logdir, "sampling_config.yaml")
        sampling_conf = vars(opt)
        with open(sampling_file, 'w') as f:
            yaml.dump(sampling_conf, f, default_flow_style=False)
        print(sampling_conf)
        run(model, imglogdir, pca_dir, eta=opt.eta,
            vanilla=opt.vanilla_sample,  n_samples=opt.n_samples, custom_steps=opt.custom_steps,
            batch_size=opt.batch_size, scale=opt.guidance_scale, pca=opt.pca, 
            decode=True, calc_halmtr=opt.calc_halmtr, thr_pcntl=opt.thr_pcntl, t_range=opt.t_range,
            split=opt.split, data_dir=opt.data_dir, dataset=opt.dataset, save_trajectory=opt.save_trajectory, prompt_tuning=opt.prompt_tuning)
    elif opt.hal_plots: # please use /home/csgrad/mbhosale/phd/ddpm_hallucination/DDPM-IP/scripts/hal_plot.py instead, this code for plotting is outdated.
        pca_dir = "/".join(opt.pca_df_file.split("/")[:-1]) # reuse the pca dir
        assert opt.pca_df_file is not None, "Please provide a path to the PCA results CSV file"
        l = make_hal_plots(opt.pca_df_file, pca_dir, halmtr_file=opt.halmtr_file, hallucinated_samples=[6, 14, 26, 41, 54, 57, 84, 85, 86, 28, 35, 64, 65, 92], 
                           n_hallucinated_samples=[1, 4, 12, 36, 40, 61, 82, 87, 90, 95, 98, 5, 22, 38], number_of_step_groups=5, number_of_steps=opt.custom_steps, thr_pcntl=opt.thr_pcntl)
        print(f"Hallucinated samples at {opt.thr_pcntl}th Percentile are "+str(l))
    print("done.")
