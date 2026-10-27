#!/usr/bin/env python3
"""
experiment2_long_video.py
=========================
Experiment 2: Long-Duration Video Evaluation

Evaluates both Our Compressed-Domain HAR Model and YOLOv8 on the 70-video
continuous merged video (20,931 frames, ~700 seconds), preserving class labels
and source-video boundaries from long_duration_manifest.json.

Compares:
  - Continuous computational performance (Total time, FPS, ms/frame, real-time factor).
  - Stage latency breakdown (Bitstream extraction / RGB decode, model inference, MV propagation / NMS).
  - Memory consumption (Extra peak RAM, GPU VRAM peak).
  - Compute demands (Detector calls, total GFLOPs, GFLOPs per second).
  - Boundary tracking stability and detection agreement across the continuous timeline.
  - Action recognition accuracy evaluated on each of the 70 segments.

Outputs (in experiments/results/):
  - experiment2_summary.csv
  - experiment2_per_segment.csv
  - experiment2_results.json
"""

import os
import sys
import csv
import json
import time
import argparse
import platform
import subprocess
from pathlib import Path
from collections import defaultdict

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

# Add paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ACTION_DIR = os.path.join(BASE_DIR, "Action_Recognition")
DETECTOR_DIR = os.path.join(BASE_DIR, "H264_Compressed_Detector")

for p in [BASE_DIR, ACTION_DIR, DETECTOR_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import fast_compressed_ops as fco
from src.inference import ActionRecognizer
from benchmark_detection import (
    StageTimer, ResourceMonitor, latency_stats, keras_flops, torch_flops,
    ffmpeg_frames, get_keyframe_indices, evaluate_boxes, yolo_detect
)
from experiments.experiment1_dataset_eval import (
    cv2_resize_norm, predict_action_from_memory
)

DEFAULT_VIDEO = os.path.join(BASE_DIR, "experiments", "results", "long_duration_70videos.mp4")
DEFAULT_MANIFEST = os.path.join(BASE_DIR, "experiments", "results", "long_duration_manifest.json")


def evaluate_segment_action(action_recognizer, mvs, boxes_by_frame, seg, width, height):
    """Evaluates action recognition on a specific segment's frame range."""
    s_frame = seg["start_frame"]
    e_frame = seg["end_frame"]

    # Filter MVs and boxes to segment boundaries
    seg_mvs = {f: mvs[f] for f in range(s_frame, e_frame + 1) if f in mvs}
    seg_boxes = {f: boxes_by_frame[f] for f in range(s_frame, e_frame + 1) if f in boxes_by_frame}

    if not seg_mvs or not seg_boxes:
        return None

    # Shift frame indices locally for clip construction
    local_mvs = {f - s_frame: seg_mvs[f] for f in seg_mvs}
    local_boxes = {f - s_frame: seg_boxes[f] for f in seg_boxes}

    return predict_action_from_memory(action_recognizer, local_mvs, local_boxes, width, height)


def run_experiment_2(
    video_path=DEFAULT_VIDEO,
    manifest_path=DEFAULT_MANIFEST,
    output_dir=os.path.join(BASE_DIR, "experiments", "results"),
    yolo_device="cuda",
    yolo_model_path="yolov8n.pt",
    detector_weights=os.path.join(BASE_DIR, "models", "best_model.h5"),
    action_model_path=os.path.join(ACTION_DIR, "models", "best_model.pth"),
    skip_yolo_full=False
):
    os.makedirs(output_dir, exist_ok=True)
    summary_csv = os.path.join(output_dir, "experiment2_summary.csv")
    per_segment_csv = os.path.join(output_dir, "experiment2_per_segment.csv")
    results_json = os.path.join(output_dir, "experiment2_results.json")

    print(f"\n{'=' * 75}")
    print(f"🎬 EXPERIMENT 2: LONG-DURATION VIDEO EVALUATION (70 VIDEOS)")
    print(f"{'=' * 75}")
    print(f"  Long-Duration Video : {video_path}")
    print(f"  Manifest File       : {manifest_path}")
    print(f"  Output Directory    : {output_dir}")
    print(f"  YOLOv8 Device       : {yolo_device}")
    print(f"{'=' * 75}\n")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Merged video not found: {video_path}. Run merge_videos.py first!")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}. Run merge_videos.py first!")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    total_frames = manifest["total_frames"]
    segments = manifest["segments"]
    print(f"📊 Loaded Manifest: {len(segments)} segments, {total_frames} frames (~{manifest['total_duration_s']:.1f} s)")

    # 1. Load Models
    print("\n📦 Loading Models...")
    from H264_Compressed_Detector.model.model import SSD300_H264
    ssd_model = SSD300_H264()
    ssd_model.load_weights(detector_weights)
    ssd_gflops = keras_flops(ssd_model) / 1e9
    print(f"  [✓] Loaded SSD300_H264 Detector ({ssd_gflops:.2f} GFLOPs/inf)")

    action_recognizer = None
    if os.path.exists(action_model_path):
        act_dev = "cuda" if torch.cuda.is_available() else "cpu"
        action_recognizer = ActionRecognizer(action_model_path, device=act_dev)
        print(f"  [✓] Loaded TemporalCNN Action Recognizer on {act_dev}")

    from ultralytics import YOLO
    yolo = YOLO(yolo_model_path)
    yolo_net = yolo.model.to(yolo_device)
    dummy_input = torch.zeros(1, 3, 640, 640, device=yolo_device)
    yolo_gflops = torch_flops(yolo_net, dummy_input) / 1e9
    print(f"  [✓] Loaded YOLOv8 ({yolo_gflops:.2f} GFLOPs/inf on {yolo_device})")

    # ============================================================
    # PIPELINE 1: OUR COMPRESSED-DOMAIN HAR PIPELINE
    # ============================================================
    print(f"\n⚡ Running Our Compressed-Domain HAR Pipeline on Long-Duration Video...")
    coded_w, coded_h = manifest["resolution"]
    from benchmark_detection import run_ours_fast

    t0_ours = time.time()
    fast_ours = run_ours_fast(video_path, ssd_model, coded_w, coded_h, total_frames, conf_thresh=0.5)
    ours_total_s = fast_ours["total_s"]
    boxes_ours = fast_ours["boxes"]
    mvs = fast_ours.get("_p_mvs", {})
    maps_count = fast_ours["detector_calls"]
    timer_ours = fast_ours["stages_s"]

    ours_fps = total_frames / ours_total_s if ours_total_s > 0 else 0
    ours_gflops = maps_count * ssd_gflops
    print(f"\n✅ OURS COMPLETED: {ours_total_s:.2f} s | {ours_fps:.1f} FPS | "
          f"{ours_total_s / manifest['total_duration_s']:.2f}x Real-Time | Total Detector FLOPs: {ours_gflops:.1f} GFLOPs")

    # ============================================================
    # PIPELINE 2: YOLOv8 KEYFRAME + MV TRACKING (High-Efficiency Baseline)
    # ============================================================
    print(f"\n⚡ Running YOLOv8-KEY on Long-Duration Video ({yolo_device})...")
    timer_yolo_key = StageTimer()
    key_indices = set(get_keyframe_indices(video_path))
    boxes_yolo_key, det_times_yolo_key = {}, []

    with ResourceMonitor(use_gpu=(yolo_device == "cuda")) as mon_yolo_key:
        with timer_yolo_key("1_yolo_keyframe_inference"):
            print(f"  [Stage 1/2] Decoding and detecting on {len(key_indices)} keyframes...")
            for f_idx, frame in enumerate(ffmpeg_frames(video_path, coded_w, coded_h)):
                if f_idx in key_indices:
                    t_inf = time.perf_counter()
                    bxs, _ = yolo_detect(yolo, frame, imgsz=640, conf=0.25, device=yolo_device)
                    det_times_yolo_key.append(time.perf_counter() - t_inf)
                    boxes_yolo_key[f_idx] = bxs

        with timer_yolo_key("2_mv_propagation"):
            print("  [Stage 2/2] Propagating YOLO keyframe boxes using MVs...", flush=True)
            yolo_key_timeline = fco.propagate_timeline_fast(total_frames, boxes_yolo_key, mvs, coded_w, coded_h)

    yolo_key_total_s = timer_yolo_key.total
    yolo_key_fps = total_frames / yolo_key_total_s if yolo_key_total_s > 0 else 0
    yolo_key_gflops = len(key_indices) * yolo_gflops
    print(f"✅ YOLOv8-KEY COMPLETED: {yolo_key_total_s:.2f} s | {yolo_key_fps:.1f} FPS | FLOPs: {yolo_key_gflops:.1f} GFLOPs")

    # ============================================================
    # PIPELINE 3: YOLOv8 FULL-FRAME INFERENCE (Conventional Pixel Domain)
    # ============================================================
    yolo_full_total_s = None
    yolo_full_fps = None
    yolo_full_gflops = None
    boxes_yolo_full = {}
    
    if not skip_yolo_full:
        print(f"\n⚡ Running YOLOv8-FULL on Long-Duration Video ({yolo_device})...")
        timer_yolo_full = StageTimer()
        with ResourceMonitor(use_gpu=(yolo_device == "cuda")) as mon_yolo_full:
            with timer_yolo_full("1_decode_and_detect_all_frames"):
                for f_idx, frame in enumerate(ffmpeg_frames(video_path, coded_w, coded_h)):
                    bxs, _ = yolo_detect(yolo, frame, imgsz=640, conf=0.25, device=yolo_device)
                    boxes_yolo_full[f_idx] = bxs
                    if (f_idx + 1) % 2000 == 0:
                        print(f"    Processed {f_idx + 1} / {total_frames} frames ({ (f_idx+1)/total_frames*100:.1f} %)...")
        yolo_full_total_s = timer_yolo_full.total
        yolo_full_fps = total_frames / yolo_full_total_s if yolo_full_total_s > 0 else 0
        yolo_full_gflops = total_frames * yolo_gflops
        print(f"✅ YOLOv8-FULL COMPLETED: {yolo_full_total_s:.2f} s | {yolo_full_fps:.1f} FPS | FLOPs: {yolo_full_gflops:.1f} GFLOPs")

    # ============================================================
    # SEGMENT-BY-SEGMENT BOUNDARY & ACTION RECOGNITION EVALUATION
    # ============================================================
    print(f"\n📊 Evaluating 70 Video Segments along the Continuous Timeline...")
    segment_records = []
    correct_actions = 0

    for seg in segments:
        seg_idx = seg["segment_idx"]
        cls = seg["class_name"]
        src = seg["source_video_name"]
        s_f = seg["start_frame"]
        e_f = seg["end_frame"]
        n_f = seg["num_frames"]

        # 1. Action Recognition for this segment
        act_res = None
        if action_recognizer:
            act_res = evaluate_segment_action(action_recognizer, mvs, boxes_ours, seg, coded_w, coded_h)

        pred_action = act_res["predicted_class_name"] if act_res else "N/A"
        conf = act_res["confidence"] if act_res else 0.0
        is_correct = int(pred_action.lower() == cls.lower()) if act_res else 0
        if is_correct:
            correct_actions += 1

        # 2. Detection Agreement on Segment (Ours vs YOLO-KEY)
        seg_ours_boxes = {f: boxes_ours[f] for f in range(s_f, e_f + 1) if f in boxes_ours}
        seg_yolo_boxes = {f: yolo_key_timeline[f] for f in range(s_f, e_f + 1) if f in yolo_key_timeline}
        agree = evaluate_boxes(seg_ours_boxes, seg_yolo_boxes, iou_thr=0.5)

        seg_rec = {
            "Segment_Idx": seg_idx + 1,
            "Class": cls,
            "Source_Video": src,
            "Start_Frame": s_f,
            "End_Frame": e_f,
            "Num_Frames": n_f,
            "Duration_s": seg["duration_s"],
            "Predicted_Action": pred_action,
            "Confidence": round(conf, 3),
            "Action_Correct": is_correct,
            "Detection_Agreement_F1": round(agree["f1"], 3),
            "Mean_IoU": round(agree["mean_iou"], 3)
        }
        segment_records.append(seg_rec)

        if (seg_idx + 1) % 10 == 0 or (seg_idx + 1) == len(segments):
            print(f"  Segment [{seg_idx+1:02d}/70] {cls} ({src}): "
                  f"Predicted: '{pred_action}' ({conf:.1%}) | Agreement F1: {agree['f1']:.3f}")

    action_acc = (correct_actions / len(segments)) * 100 if segments else 0.0
    print(f"\n🎯 Continuous Timeline Action Recognition Accuracy: {correct_actions} / {len(segments)} ({action_acc:.1f} %)")

    # Write per-segment CSV
    with open(per_segment_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(segment_records[0].keys()))
        writer.writeheader()
        writer.writerows(segment_records)
    print(f"  [✓] Saved Per-Segment CSV: {per_segment_csv}")

    # Write summary CSV
    summary_data = {
        "Metric": [
            "Total Continuous Frames",
            "Video Duration (s)",
            "Total Execution Time (s)",
            "Throughput (FPS)",
            "Latency (ms / frame)",
            "Real-Time Factor",
            "Detector Calls",
            "Total Detector GFLOPs",
            "GFLOPs / Video Second",
            "Extra Peak RAM (MB)",
            "GPU Peak VRAM (MB)",
            "Action Recognition Accuracy (70 Segments)",
            "Mean Detection Agreement F1"
        ],
        "Our Compressed HAR Pipeline": [
            str(total_frames),
            f"{manifest['total_duration_s']:.1f} s",
            f"{ours_total_s:.2f} s",
            f"{ours_fps:.1f} FPS",
            f"{(ours_total_s / total_frames * 1000):.2f} ms",
            f"{(ours_fps / 30.0):.2f}x Real-Time",
            str(maps_count),
            f"{ours_gflops:.1f} GFLOPs",
            f"{(ours_gflops / manifest['total_duration_s']):.2f}",
            f"{fast_ours.get('extra_peak_rss_mb', 0.0) or 0.0:.1f} MB",
            "0.0 MB (CPU Detector)",
            f"{action_acc:.1f} %",
            f"{np.mean([r['Detection_Agreement_F1'] for r in segment_records]):.3f}"
        ],
        "YOLOv8-KEY (Keyframe + MV)": [
            str(total_frames),
            f"{manifest['total_duration_s']:.1f} s",
            f"{yolo_key_total_s:.2f} s",
            f"{yolo_key_fps:.1f} FPS",
            f"{(yolo_key_total_s / total_frames * 1000):.2f} ms",
            f"{(yolo_key_fps / 30.0):.2f}x Real-Time",
            str(len(key_indices)),
            f"{yolo_key_gflops:.1f} GFLOPs",
            f"{(yolo_key_gflops / manifest['total_duration_s']):.2f}",
            f"{mon_yolo_key.extra_peak_rss_mb or 0.0:.1f} MB",
            f"{(torch.cuda.max_memory_allocated() / 1024**2):.1f} MB",
            "N/A (Object Detector)",
            "1.000 (Reference)"
        ]
    }
    if not skip_yolo_full and yolo_full_total_s:
        summary_data["YOLOv8-FULL (Every Frame)"] = [
            str(total_frames),
            f"{manifest['total_duration_s']:.1f} s",
            f"{yolo_full_total_s:.2f} s",
            f"{yolo_full_fps:.1f} FPS",
            f"{(yolo_full_total_s / total_frames * 1000):.2f} ms",
            f"{(yolo_full_fps / 30.0):.2f}x Real-Time",
            str(total_frames),
            f"{yolo_full_gflops:.1f} GFLOPs",
            f"{(yolo_full_gflops / manifest['total_duration_s']):.2f}",
            f"{mon_yolo_full.extra_peak_rss_mb or 0.0:.1f} MB",
            f"{(torch.cuda.max_memory_allocated() / 1024**2):.1f} MB",
            "N/A (Object Detector)",
            "N/A"
        ]

    import pandas as pd
    pd.DataFrame(summary_data).to_csv(summary_csv, index=False)
    print(f"  [✓] Saved Summary CSV: {summary_csv}")

    # Write raw JSON
    raw_out = {
        "manifest": manifest,
        "performance": {
            "ours": {
                "total_time_s": ours_total_s,
                "fps": ours_fps,
                "stages_s": timer_ours,
                "detector_calls": maps_count,
                "total_gflops": ours_gflops,
                "extra_peak_rss_mb": fast_ours.get("extra_peak_rss_mb"),
                "cpu_seconds": fast_ours.get("cpu_seconds")
            },
            "yolo_key": {
                "total_time_s": yolo_key_total_s,
                "fps": yolo_key_fps,
                "stages_s": timer_yolo_key.stages,
                "detector_calls": len(key_indices),
                "total_gflops": yolo_key_gflops,
                "extra_peak_rss_mb": mon_yolo_key.extra_peak_rss_mb
            }
        },
        "action_accuracy": action_acc,
        "segments": segment_records
    }
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(raw_out, f, indent=2)
    print(f"  [✓] Saved Results JSON: {results_json}")

    print(f"\n{'=' * 75}")
    print(f"🎉 EXPERIMENT 2 COMPLETED SUCCESSFULLY")
    print(f"{'=' * 75}\n")


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Long-Duration Video Evaluation")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Path to merged 70-video .mp4")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Path to boundary manifest JSON")
    parser.add_argument("--output_dir", default=os.path.join(BASE_DIR, "experiments", "results"))
    parser.add_argument("--yolo_device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--yolo_model", default="yolov8n.pt")
    parser.add_argument("--detector_weights", default=os.path.join(BASE_DIR, "models", "best_model.h5"))
    parser.add_argument("--action_model", default=os.path.join(ACTION_DIR, "models", "best_model.pth"))
    parser.add_argument("--skip_yolo_full", action="store_true",
                        help="Skip YOLOv8-FULL to save 10-15 minutes if only comparing against YOLO-KEY")
    args = parser.parse_args()

    run_experiment_2(
        video_path=args.video,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        yolo_device=args.yolo_device,
        yolo_model_path=args.yolo_model,
        detector_weights=args.detector_weights,
        action_model_path=args.action_model,
        skip_yolo_full=args.skip_yolo_full
    )

if __name__ == "__main__":
    main()
