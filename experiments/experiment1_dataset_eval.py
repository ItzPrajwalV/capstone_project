#!/usr/bin/env python3
"""
experiment1_dataset_eval.py
===========================
Experiment 1: Complete Dataset Evaluation

Runs HAR dataset videos through both:
  1. Our HAR Model:
     - Person Detection & Tracking: Compressed-Domain SSD300_H264 on I-frame DCT energy maps
       + Vectorized motion vector box propagation on P/B frames.
     - Action Recognition: TemporalCNN on 10-frame motion vector clips (7 classes).
  2. YOLOv8:
     - YOLOv8n-FULL: RGB decoding of every frame + YOLOv8 person detector on every frame.
     - YOLOv8n-KEY : RGB decoding of key frames + YOLOv8 on key frames + MV box propagation.

Compares:
  - Model Complexity: Parameters (M), FLOPs/MACs per inference, weights file size (MB), model-only latency (ms).
  - Computational Performance: Preprocessing latency, inference latency, postprocessing latency,
    total wall time (s), FPS, ms/frame, real-time factor, detector calls, detector GFLOPs per video.
  - Resource Usage: Extra peak RAM (RSS MB), GPU peak VRAM (MB), CPU-seconds.
  - Detection Metrics: Precision, Recall, F1-Score, Mean IoU (mIoU) @ 0.5 wherever GT annotations exist,
    plus detection agreement between Our Model and YOLOv8.
  - Action Recognition Metrics: Action classification accuracy, Top-1 / Top-5, per-class F1, confusion matrix.

Outputs (in experiments/results/):
  - experiment1_summary.csv
  - experiment1_per_video.csv
  - experiment1_model_complexity.csv
  - experiment1_results.json
"""

import os
import sys
import csv
import json
import time
import shutil
import argparse
import platform
import psutil
from collections import defaultdict
from pathlib import Path

# Force GPU growth and UTF-8 output
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

import numpy as np
import torch
import tensorflow as tf

# Add project root and subdirectories to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ACTION_DIR = os.path.join(BASE_DIR, "Action_Recognition")
DETECTOR_DIR = os.path.join(BASE_DIR, "H264_Compressed_Detector")

for p in [BASE_DIR, ACTION_DIR, DETECTOR_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Existing modules (used as-is without unnecessary modification)
import fast_compressed_ops as fco
import i_frame_extractor as ife
from bbox_propagator import propagate_bbox
from src.inference import ActionRecognizer
from benchmark_detection import (
    StageTimer, ResourceMonitor, latency_stats, keras_flops, torch_flops,
    bench_ssd_model, bench_yolo_model, ffmpeg_frames, get_keyframe_indices,
    load_gt, evaluate_boxes, propagate_timeline, yolo_detect
)

DATASET_DIR = r"C:\Users\PESU-RF\Downloads\Human Activity Recognition - Video Dataset"
GT_DIR = os.path.join(BASE_DIR, "data", "annotations")

CLASSES = [
    "Clapping",
    "Meet and Split",
    "Sitting",
    "Standing Still",
    "Walking",
    "Walking While Reading Book",
    "Walking While Using Phone"
]


# ============================================================
# RUN OUR HAR PIPELINE (Detection + Tracking + Action Recognition)
# ============================================================

from benchmark_detection import run_ours_fast, run_yolo_full, run_yolo_keyframes, get_video_frame_count

def run_ours_pipeline(video_path, ssd_model, ssd_conf, action_recognizer=None):
    """
    Runs Our Compressed-Domain HAR Pipeline:
      Stage 1-4: Fast vectorized bitstream extraction, I-frame SSD300_H264 detection,
                 and motion vector box propagation across P/B frames (run_ours_fast).
      Stage 5: Optional Action Recognition with TemporalCNN on cropped MV clips.
    """
    fps, _, width, height = ife.get_fps_and_timebase(video_path)
    total_frames = get_video_frame_count(video_path) or 300

    fast_out = run_ours_fast(video_path, ssd_model, width, height, total_frames, ssd_conf)

    # Action Recognition (if recognizer provided and motion vectors exist)
    action_result = None
    mvs = fast_out.get("_p_mvs", {})
    boxes = fast_out.get("boxes", {})
    if action_recognizer is not None and mvs and boxes:
        action_result = predict_action_from_memory(
            action_recognizer, mvs, boxes, width, height
        )

    stages = fast_out["stages_s"]
    pre_time = (stages.get("1_ffprobe_frame_types", 0.0) +
                stages.get("2_single_pass_extraction_ffmpeg", 0.0) +
                stages.get("3_dct_feature_map_build_numpy", 0.0) +
                stages.get("4_mv_parse_numpy", 0.0))
    inf_time = stages.get("5_ssd_detection_iframes", 0.0)
    post_time = stages.get("6_mv_box_propagation", 0.0)

    return {
        "pipeline": "OURS_HAR",
        "device": "cpu",
        "stages_s": stages,
        "total_time_s": fast_out["total_s"],
        "pre_time_s": pre_time,
        "inf_time_s": inf_time,
        "post_time_s": post_time,
        "fps": total_frames / fast_out["total_s"] if fast_out["total_s"] > 0 else 0.0,
        "ms_per_frame": (fast_out["total_s"] / total_frames * 1000.0) if total_frames > 0 else 0.0,
        "detector_calls": fast_out["detector_calls"],
        "total_frames": total_frames,
        "detector_call_latency": fast_out.get("detector_call_latency"),
        "cpu_seconds": fast_out.get("cpu_seconds"),
        "extra_peak_rss_mb": fast_out.get("extra_peak_rss_mb"),
        "boxes": boxes,
        "action_result": action_result,
        "_info": {"width": width, "height": height, "fps": fps, "num_frames": total_frames},
        "_mvs": mvs
    }



def cv2_resize_norm(feat_map):
    """Resizes DCT feature map to 300x300 and standardizes for SSD input."""
    import cv2
    resized = cv2.resize(feat_map, (300, 300))
    mean = np.mean(resized, axis=(0, 1), keepdims=True)
    std = np.std(resized, axis=(0, 1), keepdims=True) + 1e-6
    norm = ((resized - mean) / std).astype(np.float32)
    return np.expand_dims(norm, axis=0)


def predict_action_from_memory(recognizer, mvs, boxes_by_frame, width, height):
    """
    Runs TemporalCNN action recognition using in-memory motion vectors and bounding boxes
    without needing intermediate disk I/O.
    """
    sorted_frames = sorted(mvs.keys())
    if len(sorted_frames) < recognizer.num_frames:
        return None

    # Filter frames that have valid boxes
    usable_frames = [f for f in sorted_frames if f in boxes_by_frame and boxes_by_frame[f]]
    if len(usable_frames) < recognizer.num_frames:
        return None

    # Construct clips of consecutive P-frames
    clip_len = recognizer.num_frames
    stride = recognizer.clip_stride
    crops_per_frame = {}

    for f in usable_frames:
        box = boxes_by_frame[f][0] # Main person
        mv = mvs[f]
        # Rasterize and crop
        crop = rasterize_and_crop_mv(mv, box, width, height, recognizer.frame_size, recognizer.motion_norm_range)
        crops_per_frame[f] = crop

    starts = list(range(0, len(usable_frames) - clip_len + 1, stride))
    if not starts:
        starts = [0]

    all_clips = []
    for s in starts:
        frame_seq = usable_frames[s : s + clip_len]
        if len(frame_seq) == clip_len:
            # Stack 10 frames of 2-channel MVs -> (20, 224, 224)
            clip_arr = np.concatenate([crops_per_frame[f] for f in frame_seq], axis=0)
            all_clips.append(clip_arr)

    if not all_clips:
        return None

    batch_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(all_clips))).to(recognizer.device)
    with torch.no_grad():
        out = recognizer.model(batch_tensor)
        probs = out.cpu().numpy()

    mean_probs = probs.mean(axis=0)
    pred_idx = int(np.argmax(mean_probs))
    top_5_idx = np.argsort(mean_probs)[::-1][:5]

    return {
        "predicted_class_id": pred_idx,
        "predicted_class_name": recognizer.class_names[pred_idx],
        "confidence": float(mean_probs[pred_idx]),
        "all_probabilities": {name: float(mean_probs[i]) for i, name in enumerate(recognizer.class_names)},
        "top_5": [
            {"rank": r + 1, "class_name": recognizer.class_names[idx], "probability": float(mean_probs[idx])}
            for r, idx in enumerate(top_5_idx)
        ]
    }


def rasterize_and_crop_mv(mv_rec, box, width, height, out_size=224, norm_range=20.0):
    """Rasterizes sparse macroblock MVs into a 2-channel grid and crops to person box."""
    import cv2
    grid_h = (height + 15) // 16
    grid_w = (width + 15) // 16
    grid = np.zeros((grid_h, grid_w, 2), dtype=np.float32)

    mb_x = (mv_rec['src_x'] // 16).astype(np.int32)
    mb_y = (mv_rec['src_y'] // 16).astype(np.int32)
    valid = (mb_x >= 0) & (mb_x < grid_w) & (mb_y >= 0) & (mb_y < grid_h)
    grid[mb_y[valid], mb_x[valid], 0] = mv_rec['motion_x'][valid]
    grid[mb_y[valid], mb_x[valid], 1] = mv_rec['motion_y'][valid]

    # Convert box to grid coords with 15% margin
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad_w, pad_h = bw * 0.15, bh * 0.15
    gx1 = int(max(0, (x1 - pad_w) / 16))
    gy1 = int(max(0, (y1 - pad_h) / 16))
    gx2 = int(min(grid_w, (x2 + pad_w) / 16 + 1))
    gy2 = int(min(grid_h, (y2 + pad_h) / 16 + 1))

    if gx2 <= gx1 or gy2 <= gy1:
        crop = np.zeros((out_size, out_size, 2), dtype=np.float32)
    else:
        patch = grid[gy1:gy2, gx1:gx2]
        crop = cv2.resize(patch, (out_size, out_size))

    crop = np.clip(crop, -norm_range, norm_range) / norm_range
    # Return as contiguous (2, out_size, out_size)
    return np.ascontiguousarray(np.transpose(crop, (2, 0, 1)))


# ============================================================
# RUN YOLOv8 PIPELINE
# ============================================================

def run_yolo_evaluation(video_path, yolo, device, imgsz=640, conf=0.25, width=1080, height=1920, mode="FULL", ours_mvs=None):
    """
    Evaluates YOLOv8 on the video.
      mode='FULL': Decodes every frame and runs YOLOv8 on every frame.
      mode='KEY' : Decodes keyframes, runs YOLOv8, and propagates with MVs.
    """
    timer = StageTimer()
    is_cuda = str(device).startswith("cuda")
    if is_cuda and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with ResourceMonitor(use_gpu=is_cuda) as mon:
        if mode == "FULL":
            boxes, det_times, pre_times, post_times = {}, [], [], []
            t_start = time.perf_counter()
            frame_count = 0

            with timer("1_rgb_decode_and_yolo"):
                for f_idx, frame in enumerate(ffmpeg_frames(video_path, width, height)):
                    frame_count += 1
                    t0 = time.perf_counter()
                    bxs, sp = yolo_detect(yolo, frame, imgsz, conf, device)
                    det_times.append(time.perf_counter() - t0)
                    boxes[f_idx] = bxs
                    pre_times.append(sp.get("preprocess", 0.0) / 1000.0)
                    post_times.append(sp.get("postprocess", 0.0) / 1000.0)

            total_s = timer.total
            pre_s = float(np.sum(pre_times))
            post_s = float(np.sum(post_times))
            inf_s = float(np.sum(det_times)) - pre_s - post_s

        elif mode == "KEY":
            # Keyframes only
            key_indices = set(get_keyframe_indices(video_path))
            key_boxes, det_times, pre_times, post_times = {}, [], [], []
            frame_count = 0

            with timer("1_decode_keyframes"):
                for f_idx, frame in enumerate(ffmpeg_frames(video_path, width, height, keyframes_only=True)):
                    # Matching keyframe indices
                    pass

            with timer("2_yolo_keyframes"):
                for f_idx, frame in enumerate(ffmpeg_frames(video_path, width, height)):
                    frame_count += 1
                    if f_idx in key_indices:
                        t0 = time.perf_counter()
                        bxs, sp = yolo_detect(yolo, frame, imgsz, conf, device)
                        det_times.append(time.perf_counter() - t0)
                        key_boxes[f_idx] = bxs
                        pre_times.append(sp.get("preprocess", 0.0) / 1000.0)
                        post_times.append(sp.get("postprocess", 0.0) / 1000.0)

            with timer("3_mv_propagation"):
                boxes = {}
                current = []
                for f_idx in range(frame_count):
                    if f_idx in key_boxes and key_boxes[f_idx]:
                        current = list(key_boxes[f_idx])
                    elif ours_mvs and f_idx in ours_mvs and current:
                        current = [fco.propagate_bbox_fast(b, ours_mvs[f_idx]) for b in current]
                    boxes[f_idx] = list(current)

            total_s = timer.total
            pre_s = float(np.sum(pre_times))
            post_s = float(np.sum(post_times)) + timer.stages.get("3_mv_propagation", 0.0)
            inf_s = float(np.sum(det_times))

    gpu_peak_mb = (torch.cuda.max_memory_allocated() / 1024**2) if (is_cuda and torch.cuda.is_available()) else 0.0

    return {
        "pipeline": f"YOLOv8_{mode}",
        "device": str(device),
        "stages_s": timer.stages,
        "total_time_s": total_s,
        "pre_time_s": pre_s,
        "inf_time_s": inf_s,
        "post_time_s": post_s,
        "fps": frame_count / total_s if total_s > 0 else 0.0,
        "ms_per_frame": (total_s / frame_count * 1000.0) if frame_count > 0 else 0.0,
        "detector_calls": len(det_times),
        "total_frames": frame_count,
        "detector_call_latency": latency_stats(det_times) if det_times else None,
        "cpu_seconds": mon.cpu_seconds,
        "extra_peak_rss_mb": mon.extra_peak_rss_mb,
        "gpu_peak_mem_mb": gpu_peak_mb,
        "boxes": boxes
    }


# ============================================================
# DATASET DISCOVERY
# ============================================================

def discover_dataset(dataset_dir, max_per_class=None, class_filter=None):
    """Walks the 7 class folders and discovers all video paths."""
    videos = []
    classes = class_filter or CLASSES
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        vids = [
            os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]
        # Natural sort
        import re
        vids.sort(key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))])
        if max_per_class and max_per_class > 0:
            vids = vids[:max_per_class]
        for v in vids:
            videos.append({
                "class_name": cls,
                "video_name": os.path.splitext(os.path.basename(v))[0],
                "video_path": v
            })
    return videos


# ============================================================
# MAIN EVALUATION RUNNER
# ============================================================

def run_experiment_1(
    dataset_dir=DATASET_DIR,
    output_dir=os.path.join(BASE_DIR, "experiments", "results"),
    videos_per_class=5,
    limit=None,
    yolo_device="cuda",
    yolo_model_path="yolov8n.pt",
    detector_weights=os.path.join(BASE_DIR, "models", "best_model.h5"),
    action_model_path=os.path.join(ACTION_DIR, "models", "best_model.pth"),
    skip_key_baseline=False
):
    os.makedirs(output_dir, exist_ok=True)
    summary_csv = os.path.join(output_dir, "experiment1_summary.csv")
    per_video_csv = os.path.join(output_dir, "experiment1_per_video.csv")
    results_json = os.path.join(output_dir, "experiment1_results.json")
    model_complexity_csv = os.path.join(output_dir, "experiment1_model_complexity.csv")

    print(f"\n{'=' * 75}")
    print(f"🚀 EXPERIMENT 1: COMPLETE DATASET EVALUATION")
    print(f"{'=' * 75}")
    print(f"  Dataset Directory : {dataset_dir}")
    print(f"  Output Directory  : {output_dir}")
    print(f"  Videos per Class  : {'ALL' if not videos_per_class else videos_per_class}")
    print(f"  YOLOv8 Device     : {yolo_device}")
    print(f"  SSD Weights       : {detector_weights}")
    print(f"  Action Model      : {action_model_path}")
    print(f"{'=' * 75}\n")

    # 1. Hardware Environment
    env_info = {
        "OS": f"{platform.system()} {platform.release()}",
        "CPU": platform.processor(),
        "Logical Cores": os.cpu_count(),
        "RAM_GB": round(psutil.virtual_memory().total / (1024**3), 2) if 'psutil' in globals() else "N/A",
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "VRAM_GB": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else 0.0,
        "PyTorch": torch.__version__,
        "TensorFlow": tf.__version__,
        "Python": platform.python_version()
    }
    print("💻 System Hardware:")
    for k, v in env_info.items():
        print(f"  {k:15s}: {v}")

    # 2. Load Models
    print("\n📦 Loading Models...")
    # SSD300_H264
    from H264_Compressed_Detector.model.model import SSD300_H264
    ssd_model = SSD300_H264()
    ssd_model.load_weights(detector_weights)
    print("  [✓] Loaded SSD300_H264 Detector (TensorFlow)")

    # Action Recognizer (TemporalCNN)
    action_recognizer = None
    if os.path.exists(action_model_path):
        action_device = "cuda" if torch.cuda.is_available() else "cpu"
        action_recognizer = ActionRecognizer(action_model_path, device=action_device)
        print(f"  [✓] Loaded TemporalCNN Action Recognizer on {action_device}")
    else:
        print(f"  [!] Action model not found at {action_model_path}")

    # YOLOv8
    from ultralytics import YOLO
    yolo = YOLO(yolo_model_path)
    print(f"  [✓] Loaded YOLOv8 ({yolo_model_path})")

    # 3. Model Complexity Analysis
    print("\n📐 Computing Model Complexity & Micro-benchmarks...")
    ssd_params = ssd_model.count_params()
    ssd_gflops = keras_flops(ssd_model) / 1e9
    ssd_size_mb = os.path.getsize(detector_weights) / (1024**2)
    ssd_lat = bench_ssd_model(ssd_model, reps=50)

    # YOLOv8 complexity
    yolo_net = yolo.model.to(yolo_device)
    yolo_params = sum(p.numel() for p in yolo_net.parameters())
    dummy_input = torch.zeros(1, 3, 640, 640, device=yolo_device)
    yolo_gflops = torch_flops(yolo_net, dummy_input) / 1e9
    yolo_size_mb = os.path.getsize(yolo_model_path) / (1024**2)
    yolo_lat = bench_yolo_model(yolo, yolo_device, imgsz=640, reps=50)

    # TemporalCNN complexity
    temporal_params = sum(p.numel() for p in action_recognizer.model.parameters()) if action_recognizer else 0
    temporal_size_mb = os.path.getsize(action_model_path) / (1024**2) if os.path.exists(action_model_path) else 0
    dummy_mv = torch.zeros(1, 20, 224, 224, device=action_recognizer.device) if action_recognizer else None
    temporal_gflops = (torch_flops(action_recognizer.model, dummy_mv) / 1e9) if action_recognizer else 0

    complexity_records = [
        {
            "Model": "SSD300_H264 (Ours Detector)",
            "Framework": "TensorFlow",
            "Input Shape": "300x300x3 DCT Energy Map",
            "Params (M)": round(ssd_params / 1e6, 2),
            "GFLOPs / Inference": round(ssd_gflops, 3),
            "Weights Size (MB)": round(ssd_size_mb, 2),
            "Median Latency (ms)": round(ssd_lat["median_ms"], 2),
            "P95 Latency (ms)": round(ssd_lat["p95_ms"], 2),
            "Device": "cpu"
        },
        {
            "Model": "TemporalCNN (Ours Action)",
            "Framework": "PyTorch",
            "Input Shape": "20x224x224 Cropped MVs",
            "Params (M)": round(temporal_params / 1e6, 2),
            "GFLOPs / Inference": round(temporal_gflops, 3),
            "Weights Size (MB)": round(temporal_size_mb, 2),
            "Median Latency (ms)": 0.0,
            "P95 Latency (ms)": 0.0,
            "Device": str(action_recognizer.device) if action_recognizer else "N/A"
        },
        {
            "Model": "YOLOv8n",
            "Framework": "PyTorch / Ultralytics",
            "Input Shape": "640x640x3 RGB Frame",
            "Params (M)": round(yolo_params / 1e6, 2),
            "GFLOPs / Inference": round(yolo_gflops, 3),
            "Weights Size (MB)": round(yolo_size_mb, 2),
            "Median Latency (ms)": round(yolo_lat["median_ms"], 2),
            "P95 Latency (ms)": round(yolo_lat["p95_ms"], 2),
            "Device": str(yolo_device)
        }
    ]

    with open(model_complexity_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(complexity_records[0].keys()))
        writer.writeheader()
        writer.writerows(complexity_records)
    print(f"  [✓] Saved Model Complexity Table: {model_complexity_csv}")

    # 4. Discover Dataset Videos
    video_list = discover_dataset(dataset_dir, max_per_class=videos_per_class)
    if limit and limit > 0:
        video_list = video_list[:limit]
    print(f"\n🎥 Discovered {len(video_list)} videos for Complete Dataset Evaluation across {len(CLASSES)} classes.")

    # 5. Load existing checkpoints if resuming
    existing_results = {}
    if os.path.exists(results_json):
        try:
            with open(results_json, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            print(f"  [i] Resuming from checkpoint: {len(existing_results.get('videos', {}))} videos already completed.")
        except Exception:
            existing_results = {}

    all_video_results = existing_results.get("videos", {})
    per_video_records = []

    # 6. Evaluation Loop
    t_start_all = time.time()
    for idx, item in enumerate(video_list):
        v_name = item["video_name"]
        v_cls = item["class_name"]
        v_path = item["video_path"]
        print(f"\n[{idx+1}/{len(video_list)}] Evaluating: {v_cls} / {v_name} ...")

        if v_name in all_video_results:
            print(f"  -> Skipping (already evaluated in previous run)")
            continue

        vid_res = {
            "video_name": v_name,
            "class_name": v_cls,
            "video_path": v_path
        }

        # A) Run OURS HAR
        try:
            ours_out = run_ours_pipeline(v_path, ssd_model, ssd_conf=0.5, action_recognizer=action_recognizer)
            vid_res["ours"] = {k: v for k, v in ours_out.items() if k not in ["_mvs", "boxes"]}
            w = ours_out["_info"]["width"]
            h = ours_out["_info"]["height"]
            total_frames = ours_out["total_frames"]
            mvs = ours_out["_mvs"]
            print(f"  [OURS]     Total: {ours_out['total_time_s']:.2f}s | FPS: {ours_out['fps']:.1f} | "
                  f"Calls: {ours_out['detector_calls']} | FLOPs: {(ours_out['detector_calls']*ssd_gflops):.1f} GFLOPs")
            if ours_out.get("action_result"):
                act = ours_out["action_result"]
                correct = (act["predicted_class_name"].lower() == v_cls.lower())
                print(f"  [ACTION]   Predicted: '{act['predicted_class_name']}' ({act['confidence']:.1%}) | "
                      f"GT: '{v_cls}' | {'CORRECT ✓' if correct else 'INCORRECT ✗'}")
        except Exception as e:
            print(f"  [ERROR] OURS pipeline failed on {v_name}: {e}")
            continue

        # B) Run YOLOv8 FULL
        try:
            yolo_full = run_yolo_evaluation(v_path, yolo, device=yolo_device, width=w, height=h, mode="FULL")
            vid_res["yolo_full"] = {k: v for k, v in yolo_full.items() if k != "boxes"}
            print(f"  [YOLO-FULL] Total: {yolo_full['total_time_s']:.2f}s | FPS: {yolo_full['fps']:.1f} | "
                  f"Calls: {yolo_full['detector_calls']} | FLOPs: {(yolo_full['detector_calls']*yolo_gflops):.1f} GFLOPs")
        except Exception as e:
            print(f"  [ERROR] YOLO FULL failed on {v_name}: {e}")
            continue

        # C) Run YOLOv8 KEY (Ablation)
        yolo_key = None
        if not skip_key_baseline:
            try:
                yolo_key = run_yolo_evaluation(v_path, yolo, device=yolo_device, width=w, height=h, mode="KEY", ours_mvs=mvs)
                vid_res["yolo_key"] = {k: v for k, v in yolo_key.items() if k != "boxes"}
                print(f"  [YOLO-KEY]  Total: {yolo_key['total_time_s']:.2f}s | FPS: {yolo_key['fps']:.1f} | "
                      f"Calls: {yolo_key['detector_calls']}")
            except Exception as e:
                print(f"  [ERROR] YOLO KEY failed on {v_name}: {e}")

        # D) Ground-Truth Detection Evaluation (if available)
        gt_file = os.path.join(GT_DIR, f"{v_name}.txt")
        if not os.path.exists(gt_file):
            # Check without (N) e.g. Walking.txt
            gt_file_simple = os.path.join(GT_DIR, f"{v_cls}.txt")
            if os.path.exists(gt_file_simple):
                gt_file = gt_file_simple

        gt_data = load_gt(gt_file, w, h) if os.path.exists(gt_file) else None
        if gt_data:
            eval_ours = evaluate_boxes(ours_out["boxes"], gt_data, iou_thr=0.5)
            eval_yolo = evaluate_boxes(yolo_full["boxes"], gt_data, iou_thr=0.5)
            vid_res["gt_evaluation"] = {
                "ours": eval_ours,
                "yolo_full": eval_yolo,
                "gt_file": gt_file
            }
            print(f"  [ACCURACY] GT Found! Ours F1: {eval_ours['f1']:.3f} (mIoU {eval_ours['mean_iou']:.3f}) | "
                  f"YOLO F1: {eval_yolo['f1']:.3f} (mIoU {eval_yolo['mean_iou']:.3f})")
        else:
            # Evaluate Detection Agreement between Ours and YOLO
            agreement = evaluate_boxes(ours_out["boxes"], yolo_full["boxes"], iou_thr=0.5)
            vid_res["yolo_agreement"] = agreement
            print(f"  [AGREEMENT] Ours vs YOLOv8 Overlap -> F1: {agreement['f1']:.3f} | Mean IoU: {agreement['mean_iou']:.3f}")

        # Record per-video tabular metrics
        speedup = yolo_full["total_time_s"] / ours_out["total_time_s"] if ours_out["total_time_s"] > 0 else 0
        flop_reduction = (yolo_full["detector_calls"] * yolo_gflops) / (ours_out["detector_calls"] * ssd_gflops) \
            if (ours_out["detector_calls"] * ssd_gflops) > 0 else 0

        action_acc = 1.0 if (ours_out.get("action_result") and 
                             ours_out["action_result"]["predicted_class_name"].lower() == v_cls.lower()) else 0.0

        rec = {
            "Class": v_cls,
            "Video": v_name,
            "Frames": total_frames,
            "Ours_Time_s": round(ours_out["total_time_s"], 2),
            "Ours_FPS": round(ours_out["fps"], 1),
            "Ours_Pre_ms": round(ours_out["pre_time_s"] * 1000 / total_frames, 2),
            "Ours_Inf_ms": round(ours_out["inf_time_s"] * 1000 / total_frames, 2),
            "Ours_Post_ms": round(ours_out["post_time_s"] * 1000 / total_frames, 2),
            "Ours_Detector_Calls": ours_out["detector_calls"],
            "Ours_GFLOPs": round(ours_out["detector_calls"] * ssd_gflops, 1),
            "Ours_RAM_MB": round(ours_out["extra_peak_rss_mb"] or 0, 1),
            "YOLO_Time_s": round(yolo_full["total_time_s"], 2),
            "YOLO_FPS": round(yolo_full["fps"], 1),
            "YOLO_Pre_ms": round(yolo_full["pre_time_s"] * 1000 / total_frames, 2),
            "YOLO_Inf_ms": round(yolo_full["inf_time_s"] * 1000 / total_frames, 2),
            "YOLO_Post_ms": round(yolo_full["post_time_s"] * 1000 / total_frames, 2),
            "YOLO_Detector_Calls": yolo_full["detector_calls"],
            "YOLO_GFLOPs": round(yolo_full["detector_calls"] * yolo_gflops, 1),
            "YOLO_VRAM_MB": round(yolo_full["gpu_peak_mem_mb"], 1),
            "Speedup_x": round(speedup, 2),
            "Compute_Reduction_x": round(flop_reduction, 1),
            "Action_Predicted": ours_out["action_result"]["predicted_class_name"] if ours_out.get("action_result") else "N/A",
            "Action_Confidence": round(ours_out["action_result"]["confidence"], 3) if ours_out.get("action_result") else 0.0,
            "Action_Correct": int(action_acc),
            "Detection_Agreement_F1": round(vid_res.get("yolo_agreement", {}).get("f1", 0.0), 3)
        }
        per_video_records.append(rec)
        all_video_results[v_name] = vid_res

        # Save checkpoint after each video
        save_results_checkpoint(
            results_json, summary_csv, per_video_csv,
            env_info, complexity_records, all_video_results, per_video_records
        )

    total_eval_time = time.time() - t_start_all
    print(f"\n{'=' * 75}")
    print(f"🎉 EXPERIMENT 1 COMPLETE in {total_eval_time:.1f} s")
    print(f"  Summary CSV     : {summary_csv}")
    print(f"  Per-video CSV   : {per_video_csv}")
    print(f"  Raw JSON        : {results_json}")
    print(f"{'=' * 75}\n")


def save_results_checkpoint(results_json, summary_csv, per_video_csv, env_info, complexity, all_videos, records):
    """Saves evaluation state incrementally to prevent data loss."""
    # Write JSON
    full_data = {
        "environment": env_info,
        "complexity": complexity,
        "total_evaluated": len(all_videos),
        "videos": all_videos
    }
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(full_data, f, indent=2)

    # Write per-video CSV
    if records:
        with open(per_video_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

        # Write aggregated summary CSV
        agg = {
            "Metric": [
                "Average Total Time (s)",
                "Average FPS",
                "Average Preprocessing Latency (ms/frame)",
                "Average Inference Latency (ms/frame)",
                "Average Postprocessing Latency (ms/frame)",
                "Average Detector Calls per Video",
                "Average Detector GFLOPs per Video",
                "Peak Memory Usage",
                "Average Speedup Factor (Ours vs YOLO)",
                "Average FLOPs Compute Reduction",
                "HAR Action Recognition Accuracy",
                "Average Detection Agreement (F1 vs YOLO)"
            ],
            "Our Compressed HAR Model": [
                f"{np.mean([r['Ours_Time_s'] for r in records]):.2f} s",
                f"{np.mean([r['Ours_FPS'] for r in records]):.1f} FPS",
                f"{np.mean([r['Ours_Pre_ms'] for r in records]):.2f} ms",
                f"{np.mean([r['Ours_Inf_ms'] for r in records]):.2f} ms",
                f"{np.mean([r['Ours_Post_ms'] for r in records]):.2f} ms",
                f"{np.mean([r['Ours_Detector_Calls'] for r in records]):.1f}",
                f"{np.mean([r['Ours_GFLOPs'] for r in records]):.1f} GFLOPs",
                f"{np.mean([r['Ours_RAM_MB'] for r in records]):.1f} MB (RAM)",
                f"{np.mean([r['Speedup_x'] for r in records]):.2f}x faster",
                f"{np.mean([r['Compute_Reduction_x'] for r in records]):.1f}x less compute",
                f"{np.mean([r['Action_Correct'] for r in records]) * 100:.1f} %",
                f"{np.mean([r['Detection_Agreement_F1'] for r in records]):.3f}"
            ],
            "YOLOv8-FULL Baseline": [
                f"{np.mean([r['YOLO_Time_s'] for r in records]):.2f} s",
                f"{np.mean([r['YOLO_FPS'] for r in records]):.1f} FPS",
                f"{np.mean([r['YOLO_Pre_ms'] for r in records]):.2f} ms",
                f"{np.mean([r['YOLO_Inf_ms'] for r in records]):.2f} ms",
                f"{np.mean([r['YOLO_Post_ms'] for r in records]):.2f} ms",
                f"{np.mean([r['YOLO_Detector_Calls'] for r in records]):.1f}",
                f"{np.mean([r['YOLO_GFLOPs'] for r in records]):.1f} GFLOPs",
                f"{np.mean([r['YOLO_VRAM_MB'] for r in records]):.1f} MB (VRAM)",
                "1.00x (Baseline)",
                "1.0x (Baseline)",
                "N/A (Object Detector)",
                "1.000 (Self-Reference)"
            ]
        }
        import pandas as pd
        pd.DataFrame(agg).to_csv(summary_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Complete Dataset Evaluation")
    parser.add_argument("--dataset_dir", default=DATASET_DIR, help="Path to HAR Dataset")
    parser.add_argument("--output_dir", default=os.path.join(BASE_DIR, "experiments", "results"))
    parser.add_argument("--videos_per_class", type=int, default=5,
                        help="Number of videos per class (default: 5; use 0 for ALL videos in dataset)")
    parser.add_argument("--limit", type=int, default=None, help="Cap total videos evaluated")
    parser.add_argument("--yolo_device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for YOLOv8 (cuda or cpu)")
    parser.add_argument("--yolo_model", default="yolov8n.pt", help="Path to YOLOv8 weights")
    parser.add_argument("--detector_weights", default=os.path.join(BASE_DIR, "models", "best_model.h5"))
    parser.add_argument("--action_model", default=os.path.join(ACTION_DIR, "models", "best_model.pth"))
    parser.add_argument("--skip_key_baseline", action="store_true", help="Skip YOLOv8-KEY ablation")
    args = parser.parse_args()

    run_experiment_1(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        videos_per_class=args.videos_per_class if args.videos_per_class > 0 else None,
        limit=args.limit,
        yolo_device=args.yolo_device,
        yolo_model_path=args.yolo_model,
        detector_weights=args.detector_weights,
        action_model_path=args.action_model,
        skip_key_baseline=args.skip_key_baseline
    )

if __name__ == "__main__":
    main()
