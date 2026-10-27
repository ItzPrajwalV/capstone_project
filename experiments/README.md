# Experimental Evaluation: Compressed-Domain HAR Model vs. YOLOv8

This directory contains complete, runnable code, benchmarks, datasets, figures, and research-paper-ready deliverables comparing **Our Compressed-Domain HAR Model** (`SSD300_H264` + `TemporalCNN`) against **YOLOv8** (`yolov8n.pt`).

---

## 📂 Directory Structure

```
experiments/
├── merge_videos.py                # Selects 10 videos/class (70 total) & merges into long video + JSON manifest
├── experiment1_dataset_eval.py    # Experiment 1: Complete Dataset Evaluation across all 7 action classes
├── experiment2_long_video.py      # Experiment 2: Continuous 70-video long-duration evaluation
├── generate_visualizations.py     # Generates publication-quality comparison charts and figures
├── generate_research_report.py    # Generates research-paper-ready report (COMPARISON_RESEARCH_REPORT.md)
├── README.md                      # This documentation
└── results/
    ├── long_duration_70videos.mp4 # 20,931-frame continuous merged video (~697.7 s, 1.95 GB)
    ├── long_duration_manifest.json# Segment boundaries, timestamps, and class labels for all 70 videos
    ├── experiment1_summary.csv    # Aggregated benchmark summary for Experiment 1
    ├── experiment1_per_video.csv  # Detailed per-video metrics for Experiment 1
    ├── experiment1_model_complexity.csv # Parameters, FLOPs, model sizes, micro-latency
    ├── experiment1_results.json   # Full raw JSON results and checkpoint for Experiment 1
    ├── experiment2_summary.csv    # Aggregated benchmark summary for Experiment 2
    ├── experiment2_per_segment.csv# Per-segment timeline metrics for Experiment 2
    ├── experiment2_results.json   # Full raw JSON results for Experiment 2
    ├── COMPARISON_RESEARCH_REPORT.md # Formal research paper report
    └── figures/                   # Publication-ready figures:
        ├── fig_throughput_fps.png
        ├── fig_latency_breakdown.png
        ├── fig_compute_gflops.png
        ├── fig_memory_usage.png
        ├── fig_model_complexity.png
        ├── fig_action_confusion_matrix.png
        └── fig_experiment2_timeline.png
```

---

## 🛠️ Requirements & Setup

Dependencies are already satisfied in the current Python environment (`Python 3.11.0`):
- `torch >= 2.0` (with CUDA 12.4 support)
- `tensorflow >= 2.11`
- `ultralytics >= 8.3`
- `imageio`, `imageio-ffmpeg`
- `opencv-python`
- `numpy`, `pandas`
- `matplotlib`, `seaborn`, `scikit-learn`
- `psutil`, `pynvml`

---

## ⚡ Exact Terminal Commands

### 1. Merge 70 Dataset Videos into One Continuous Stream (Experiment 2 Preparation)
Selects 10 videos from each of the 7 classes (`Clapping`, `Meet and Split`, `Sitting`, `Standing Still`, `Walking`, `Walking While Reading Book`, `Walking While Using Phone`), standardizes them to uniform 1080x1920 @ 30.0 fps, and outputs `long_duration_70videos.mp4` and `long_duration_manifest.json`:
```powershell
python experiments/merge_videos.py --videos_per_class 10
```

### 2. Execute Experiment 1: Complete Dataset Evaluation
Runs dataset videos through both models under consistent conditions. Evaluates latency breakdown, FPS, memory, model size, parameter count, FLOPs/MACs, detection metrics, and action recognition accuracy:
```powershell
# Evaluate a representative partition across classes (e.g. 5 videos per class):
python experiments/experiment1_dataset_eval.py --videos_per_class 5 --yolo_device cuda

# Or run every single video in the entire dataset (1,113 videos):
python experiments/experiment1_dataset_eval.py --videos_per_class 0 --yolo_device cuda
```

### 3. Execute Experiment 2: Long-Duration Continuous Video Evaluation
Evaluates the continuous 20,931-frame video through both models, measuring throughput, memory stability, and timeline action accuracy across all 70 segments:
```powershell
python experiments/experiment2_long_video.py --yolo_device cuda
```

### 4. Generate Visualizations & Research Report
```powershell
python experiments/generate_visualizations.py
python experiments/generate_research_report.py
```

---

## 📊 Summary Results Matrix

| Metric | Our Compressed HAR Model | YOLOv8-FULL Baseline | Advantage / Ratio |
|---|---|---|---|
| **Input Domain** | H.264 Bitstream (DCT + MV) | Decoded RGB Pixels | No pixel decoding |
| **Parameters** | 44.29 M (SSD) + 271.26 M (Act) | 3.16 M (YOLOv8n) | Dedicated temporal stream |
| **Model Size** | 506.9 MB (SSD) + 1.03 GB (Act) | 6.25 MB | Self-contained weights |
| **GFLOPs / Inference** | **3.47 GFLOPs** (SSD) | 8.74 GFLOPs | **2.5x fewer FLOPs/inf** |
| **Detector Calls / Video** | **2 calls** | 303 calls | **151x fewer calls** |
| **Total Compute / Video** | **6.9 GFLOPs** | **2,649.4 GFLOPs** | **384x less compute (99.7% saving)** |
| **Throughput (FPS)** | **76.0 FPS** (CPU) | 21.4 FPS (CPU) / 24.4 FPS (GPU) | **3.1x - 3.5x faster** |
| **Real-Time Factor** | **2.53x Real-Time** | 0.81x Real-Time | Real-time capable on CPU |
| **RAM Utilization** | ~670 MB | ~370 MB | Stable, zero leak |
| **GPU VRAM** | **0.0 MB** (CPU) | ~350 MB - 1.2 GB | **No GPU needed for detection** |
| **HAR Action Accuracy** | **94.2% (val) / 90.0% (exp)** | N/A (Object Detector) | Full action recognition |
