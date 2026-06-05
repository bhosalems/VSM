import argparse
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots 
import pandas as pd
import os
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "--pca_df_file",
        type=str,
        default=None,
        help="Path to PCA results CSV file",
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
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--hal_samples",
        default=[],
        type=str,
        help="List of hallicinated sample indices"
    )
    parser.add_argument(
        "--n_hal_samples",
        default=[],
        type=str, 
        help="List of non-hallucinated sample indices"
    )
    return parser

def pca_plots(df_file, out_dir, hallucinated_samples, n_hallucinated_samples,
                               number_of_steps=100, number_of_step_groups=10):
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

def halmtr_plots(halmtr_file, hallucinated_samples, n_hallucinated_samples=[], thr_pcntl=85, 
                 out_dir='/home/csgrad/mbhosale/phd/ddpm_hallucination/DDPM-IP/log/'):
    #Hallucination metrics plots
    metric_data = pd.read_csv(halmtr_file)
    hall_values = []
    if n_hallucinated_samples:
        non_hall_values = n_hallucinated_samples
        get_all=False
    else:
        non_hall_values = []
        get_all = True
        
    for sid, val in metric_data.iterrows():
        if sid in hallucinated_samples:
            hall_values.append(val['HAL Metric'])
        elif get_all:
            n_hallucinated_samples.append(int(val['Sample Name'].split("_")[-1])) 
            non_hall_values.append(val['HAL Metric'])
            
    # Convert to numpy arrays if you like (not strictly necessary):
    hall_values = np.array(hall_values)
    non_hall_values = np.array(non_hall_values)
    threshold = metric_data["HAL Metric"].quantile(thr_pcntl / 100.0)
    
    plt.figure(figsize=(8,6))
    plt.hist(
       non_hall_values, bins=50, color='blue', alpha=0.5, label='Non-Hallucinated'  
    )
    plt.hist(
        hall_values, bins=50, color='orange', alpha=0.5, label='Hallucinated'
    )
    plt.axvline(
        x=threshold, color='red', linestyle='--',
        linewidth=2, label=f'{thr_pcntl} % cutoff = {threshold:.4f}'
    )
    plt.xlabel("Hallucination Metric")
    plt.ylabel("Frequency")
    plt.title("Histogram of Metric Values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_halmtr.png"), dpi=150)
    plt.close()
    
    plt.figure(figsize=(8,6))
    plt.bar(metric_data["Sample Name"][:50], metric_data["HAL Metric"][:50], color="skyblue", edgecolor="black")
    plt.title(f"Histogram of Hallucination-Metric", fontsize=14) # we choose only 50 such images for now
    plt.xlabel("Sample Name")
    plt.ylabel("Hal Metric")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "all_halmtr.png"), dpi=150)
    plt.close()
    filtered_samples = metric_data[metric_data["HAL Metric"] > threshold].copy()["Sample Name"]

    # calcualte TPR and Retention = 100-FPR
    predicted_set = set([int(f.split("_")[-1]) for f in set(filtered_samples)])
    truth_set = set(hallucinated_samples)
    TP = len(predicted_set.intersection(truth_set))
    FN = len(truth_set - predicted_set)
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    print("True Positive Rate (TPR): {:.2f}%: \n".format(TPR * 100))
    
    false_positives = predicted_set.intersection(set(n_hallucinated_samples))
    FP = len(false_positives)
    total_negatives = len(n_hallucinated_samples)
    FPR = FP / total_negatives if total_negatives > 0 else 0
    retention_rate = (1 - FPR) * 100

    print("False Positive Rate (FPR): {:.2f}% ".format(FPR * 100))
    print("Retention Rate: {:.2f}% \n".format(retention_rate))
    
    return predicted_set

if __name__ == '__main__':
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    hal_samples = []
    n_hal_samples = []
    if opt.hal_samples:
        hal_samples = [int(x) for x in opt.hal_samples.split(",")]
    elif opt.n_hal_samples:
        n_hal_samples = [int(x) for x in opt.n_hal_samples.split(",")]
    pca_dir = None
    if opt.pca_df_file is not None and opt.pca:
        pca_dir = "/".join(opt.pca_df_file.split("/")[:-1]) # reuse the pca dir
        pca_plots(opt.pca_df_file, pca_dir, hallucinated_samples=hal_samples, 
                        n_hallucinated_samples=n_hal_samples, number_of_step_groups=5, number_of_steps=opt.custom_steps)
    if opt.halmtr_file is not None:
        l = halmtr_plots(halmtr_file=opt.halmtr_file, hallucinated_samples=hal_samples, n_hallucinated_samples=n_hal_samples, 
                         thr_pcntl=opt.thr_pcntl, out_dir=opt.logdir)
        print(f"Filtered Hallucinated samples at {opt.thr_pcntl}th Percentile are \n"+str(l))
    print("done.")