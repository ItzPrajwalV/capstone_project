#!/usr/bin/env python3
"""
generate_visualizations.py
==========================
Generates high-resolution publication-quality plots and comparison figures
for both Experiment 1 and Experiment 2.

Outputs (in experiments/results/figures/):
  1. fig_throughput_fps.png
  2. fig_latency_breakdown.png
  3. fig_compute_gflops.png
  4. fig_memory_usage.png
  5. fig_model_complexity.png
  6. fig_detection_metrics.png
  7. fig_action_confusion_matrix.png
  8. fig_experiment2_timeline.png
"""

import os
import sys

# Ensure UTF-8 console output on Windows
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["figure.titlesize"] = 14
plt.rcParams["figure.titleweight"] = "bold"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "experiments", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

EXP1_JSON = os.path.join(RESULTS_DIR, "experiment1_results.json")
EXP1_PER_VID = os.path.join(RESULTS_DIR, "experiment1_per_video.csv")
EXP1_COMPLEXITY = os.path.join(RESULTS_DIR, "experiment1_model_complexity.csv")
EXP2_JSON = os.path.join(RESULTS_DIR, "experiment2_results.json")
EXP2_PER_SEG = os.path.join(RESULTS_DIR, "experiment2_per_segment.csv")

OURS_COLOR = "#1b9e77"      # Teal green
YOLO_COLOR = "#d95f02"      # Bright orange
YOLO_KEY_COLOR = "#7570b3"  # Purple
ACCENT_BLUE = "#386cb0"


def plot_model_complexity():
    """Plots Parameters (M) vs Weights Size (MB) vs FLOPs."""
    if not os.path.exists(EXP1_COMPLEXITY):
        return
    import pandas as pd
    df = pd.read_csv(EXP1_COMPLEXITY)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    width = 0.35

    bars1 = ax1.bar(x - width/2, df["Params (M)"], width, label="Parameters (M)", color="#2b5c8f", edgecolor="black")
    ax1.set_ylabel("Parameters (Millions)", color="#2b5c8f")
    ax1.tick_params(axis="y", labelcolor="#2b5c8f")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Model"], rotation=15, ha="right", fontweight="bold")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, df["GFLOPs / Inference"], width, label="GFLOPs / Inference", color="#e76f51", edgecolor="black")
    ax2.set_ylabel("GFLOPs / Inference (MACs x 2)", color="#e76f51")
    ax2.tick_params(axis="y", labelcolor="#e76f51")

    # Add data labels
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}M", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f"{yval:.2f}G", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.title("Model Complexity: Parameter Count & Computational Demand (FLOPs)")
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig_model_complexity.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [✓] Generated {out_path}")


def plot_throughput_and_speedup():
    """Plots Throughput (FPS) comparing Our Compressed Model vs YOLOv8."""
    if not os.path.exists(EXP1_PER_VID):
        return
    import pandas as pd
    df = pd.read_csv(EXP1_PER_VID)

    # Group by Class
    cls_summary = df.groupby("Class").agg({
        "Ours_FPS": "mean",
        "YOLO_FPS": "mean",
        "Speedup_x": "mean"
    }).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(cls_summary))
    width = 0.35

    b1 = ax1.bar(x - width/2, cls_summary["Ours_FPS"], width, label="Our Compressed HAR Model (CPU)", color=OURS_COLOR, edgecolor="black")
    b2 = ax1.bar(x + width/2, cls_summary["YOLO_FPS"], width, label="YOLOv8-FULL (GPU / CPU)", color=YOLO_COLOR, edgecolor="black")

    ax1.axhline(30.0, color="gray", linestyle="--", linewidth=1.5, label="Real-Time Target (30 FPS)")
    ax1.set_ylabel("Inference Throughput (Frames Per Second)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cls_summary["Class"], rotation=20, ha="right", fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.set_title("End-to-End Processing Throughput by Action Class (FPS)")

    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig_throughput_fps.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [✓] Generated {out_path}")


def plot_latency_breakdown():
    """Plots Per-Frame Latency Breakdown (Preprocessing, Inference, Postprocessing)."""
    if not os.path.exists(EXP1_PER_VID):
        return
    import pandas as pd
    df = pd.read_csv(EXP1_PER_VID)

    ours_pre = df["Ours_Pre_ms"].mean()
    ours_inf = df["Ours_Inf_ms"].mean()
    ours_post = df["Ours_Post_ms"].mean()

    yolo_pre = df["YOLO_Pre_ms"].mean()
    yolo_inf = df["YOLO_Inf_ms"].mean()
    yolo_post = df["YOLO_Post_ms"].mean()

    models = ["Our Compressed Model", "YOLOv8-FULL Baseline"]
    pre = [ours_pre, yolo_pre]
    inf = [ours_inf, yolo_inf]
    post = [ours_post, yolo_post]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    width = 0.45

    p1 = ax.bar(x, pre, width, label="Preprocessing (Bitstream / Decode & Resizing)", color="#4575b4", edgecolor="black")
    p2 = ax.bar(x, inf, width, bottom=pre, label="Model Inference Forward Pass", color="#d73027", edgecolor="black")
    p3 = ax.bar(x, post, width, bottom=np.array(pre) + np.array(inf), label="Postprocessing & Tracking (MV / NMS)", color="#fee090", edgecolor="black")

    ax.set_ylabel("Latency per Frame (ms / frame)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_title("Per-Frame Processing Latency Breakdown (ms)")

    # Annotate total
    totals = [ours_pre + ours_inf + ours_post, yolo_pre + yolo_inf + yolo_post]
    for i, t in enumerate(totals):
        ax.text(x[i], t + 1, f"{t:.2f} ms ({1000/t:.1f} FPS)", ha="center", va="bottom", fontweight="bold", fontsize=11)

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig_latency_breakdown.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [✓] Generated {out_path}")


def plot_compute_gflops():
    """Plots Total Detector GFLOPs per Video (Log Scale)."""
    if not os.path.exists(EXP1_PER_VID):
        return
    import pandas as pd
    df = pd.read_csv(EXP1_PER_VID)

    fig, ax = plt.subplots(figsize=(9, 5))
    models = ["Our Compressed HAR Model", "YOLOv8-FULL (Every Frame)"]
    gflops = [df["Ours_GFLOPs"].mean(), df["YOLO_GFLOPs"].mean()]
    calls = [df["Ours_Detector_Calls"].mean(), df["YOLO_Detector_Calls"].mean()]
    colors = [OURS_COLOR, YOLO_COLOR]

    bars = ax.bar(models, gflops, color=colors, width=0.45, edgecolor="black")
    ax.set_yscale("log")
    ax.set_ylabel("Total Detector GFLOPs per Video (Log Scale)")
    ax.set_title("Computational Workload: Total Detector Compute per 10s Video")

    reduction = gflops[1] / gflops[0] if gflops[0] > 0 else 0
    ax.text(0, gflops[0] * 1.5, f"{gflops[0]:.1f} GFLOPs\n({calls[0]:.0f} calls)", ha="center", va="bottom", fontweight="bold", color="darkgreen")
    ax.text(1, gflops[1] * 1.5, f"{gflops[1]:.1f} GFLOPs\n({calls[1]:.0f} calls)", ha="center", va="bottom", fontweight="bold", color="darkred")

    ax.text(0.5, np.sqrt(gflops[0] * gflops[1]), f"{reduction:.1f}x COMPUTE REDUCTION",
            ha="center", va="center", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8, edgecolor="black"))

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig_compute_gflops.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [✓] Generated {out_path}")


def plot_memory_usage():
    """Plots Memory Utilization (RAM and VRAM)."""
    if not os.path.exists(EXP1_PER_VID):
        return
    import pandas as pd
    df = pd.read_csv(EXP1_PER_VID)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    categories = ["Ours: Extra RAM (CPU)", "YOLO: GPU VRAM (CUDA)"]
    values = [df["Ours_RAM_MB"].mean(), df["YOLO_VRAM_MB"].mean()]

    bars = ax.bar(categories, values, color=["#3182bd", "#de2d26"], width=0.4, edgecolor="black")
    ax.set_ylabel("Memory Allocated (MB)")
    ax.set_title("Peak Process Memory Consumption (RAM vs GPU VRAM)")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 10, f"{h:.1f} MB", ha="center", va="bottom", fontweight="bold")

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig_memory_usage.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [✓] Generated {out_path}")


def plot_action_confusion_matrix():
    """Generates Confusion Matrix for Action Recognition."""
    if not os.path.exists(EXP1_JSON):
        return
    with open(EXP1_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    videos = data.get("videos", {})
    if not videos:
        return

    classes = [
        "clapping", "meet and split", "sitting", "standing still",
        "walking", "walking while reading book", "walking while using phone"
    ]
    cls_map = {c.lower(): i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)

    for v_name, v_info in videos.items():
        gt_cls = v_info.get("class_name", "").lower()
        ours_data = v_info.get("ours", {})
        act_res = ours_data.get("action_result")
        if act_res and gt_cls in cls_map:
            pred_cls = act_res["predicted_class_name"].lower()
            if pred_cls in cls_map:
                cm[cls_map[gt_cls], cls_map[pred_cls]] += 1

    # Normalize by row
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                xticklabels=[c.title() for c in classes],
                yticklabels=[c.title() for c in classes], ax=ax)
    ax.set_xlabel("Predicted Action Class", fontweight="bold")
    ax.set_ylabel("Ground Truth Action Class", fontweight="bold")
    ax.set_title("TemporalCNN Action Recognition Normalized Confusion Matrix")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig_action_confusion_matrix.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [✓] Generated {out_path}")


def plot_experiment2_timeline():
    """Plots Continuous Long-Duration Video Timeline (70 Segments)."""
    if not os.path.exists(EXP2_PER_SEG):
        return
    import pandas as pd
    df = pd.read_csv(EXP2_PER_SEG)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    x = df["Segment_Idx"]
    ax1.plot(x, df["Confidence"], marker="o", color="#2ca02c", linewidth=2, label="Prediction Confidence")
    ax1.set_ylabel("Confidence Score")
    ax1.set_title("Experiment 2: Continuous 70-Video Stream Timeline Evaluation")
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Color background segments by class
    unique_classes = df["Class"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    cls_to_color = dict(zip(unique_classes, colors))

    for idx, row in df.iterrows():
        c = cls_to_color[row["Class"]]
        ax1.axvspan(row["Segment_Idx"] - 0.5, row["Segment_Idx"] + 0.5, color=c, alpha=0.15)
        ax2.axvspan(row["Segment_Idx"] - 0.5, row["Segment_Idx"] + 0.5, color=c, alpha=0.15)

    # Plot Detection Agreement F1
    ax2.bar(x, df["Detection_Agreement_F1"], color="#1f77b4", width=0.6, label="Detection Agreement F1 (vs YOLO)")
    ax2.set_ylabel("Detection F1")
    ax2.set_xlabel("Segment Index (1 to 70 across 7 Classes)")
    ax2.set_ylim(0, 1.1)

    # Add legend for classes
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cls_to_color[c], alpha=0.5, label=c) for c in unique_classes]
    ax1.legend(handles=legend_elements, loc="lower right", ncol=2, fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig_experiment2_timeline.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [✓] Generated {out_path}")


def main():
    print(f"\n📊 Generating Visualizations and Research Plots...")
    plot_model_complexity()
    plot_throughput_and_speedup()
    plot_latency_breakdown()
    plot_compute_gflops()
    plot_memory_usage()
    plot_action_confusion_matrix()
    plot_experiment2_timeline()
    print(f"\n✅ All figures successfully generated in: {FIG_DIR}\n")


if __name__ == "__main__":
    main()
