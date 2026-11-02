"""
run_multi_video_pipeline.py
===========================
Multi-Video, Multi-Class Compressed-Domain Action Recognition Pipeline:
Takes multiple video paths (from different classes or folders), seamlessly
merges them into one continuous stream, and runs end-to-end compressed-domain:
  1. Standardization and Concatenation into a single continuous video.
  2. Single-pass I-frame DCT + P-frame Motion Vector extraction.
  3. SSD300_H264 person detection on key I-frames.
  4. Temporal motion-vector bounding box propagation across all continuous frames.
  5. 15% margin motion vector cropping around human boxes.
  6. TemporalCNN 10-frame sliding window action recognition.
  7. Multi-class timeline tracking & per-segment action evaluation.
  8. Rendering a single merged annotated video with real-time dynamic action overlays,
     segment banners, and bounding boxes.

Usage Examples:
    # 1. Pass multiple video paths directly:
    python run_multi_video_pipeline.py --videos "path/to/Sitting (1).mp4" "path/to/Walking (1).mp4" "path/to/Clapping (1).mp4" --preview

    # 2. Pass a text file with one video path per line:
    python run_multi_video_pipeline.py --video_list my_videos.txt --preview

    # 3. Specify a custom merged output name:
    python run_multi_video_pipeline.py --videos "path1.mp4" "path2.mp4" --output_name multi_class_test
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
import re
from pathlib import Path

# Configure TensorFlow & PyTorch coexistence
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Reconfigure console for UTF-8 on Windows
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

import numpy as np
import cv2
import imageio
import imageio_ffmpeg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACTION_DIR = os.path.join(BASE_DIR, "Action_Recognition")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, ACTION_DIR)

# PyTorch action recognition
import torch
from src.inference import ActionRecognizer
from src.data_loader import load_bbox_csv

# Compressed-domain preprocessor (TensorFlow + FFmpeg)
from prepare_action_dataset import ActionDatasetPreprocessor


def probe_video_info(video_path, ffmpeg_exe):
    """Probes frame count, width, height, and fps of a video using OpenCV/FFmpeg."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 1080, 1920, 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count, width, height, fps


def infer_class_name(video_path):
    """Infers the class name from parent directory or filename."""
    parent_dir = os.path.basename(os.path.dirname(os.path.abspath(video_path)))
    filename = os.path.splitext(os.path.basename(video_path))[0]
    
    # Common HAR class names
    known_classes = [
        "clapping", "meet and split", "sitting", "standing still",
        "walking while reading book", "walking while using phone", "walking"
    ]
    
    clean_parent = parent_dir.lower().strip()
    for kc in known_classes:
        if kc == clean_parent:
            return kc
            
    clean_file = filename.lower().strip()
    for kc in known_classes:
        if kc in clean_file:
            return kc
            
    return parent_dir if parent_dir else "unknown"


def standardize_and_merge_videos(video_paths, output_merged_path, target_width=1080, target_height=1920, target_fps=30.0):
    """
    Standardizes each video in video_paths to uniform resolution and frame rate,
    then concatenates them into a single seamless continuous MP4 video.
    Returns: (output_merged_path, segments_metadata)
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    temp_dir = os.path.join(os.path.dirname(output_merged_path), "temp_merge_clips")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_merged_path), exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"🎬 Merging {len(video_paths)} Videos into Single Continuous Stream...")
    print(f"  Target Resolution : {target_width}x{target_height}")
    print(f"  Target FPS        : {target_fps}")
    print(f"  Merged Output     : {output_merged_path}")
    print(f"{'=' * 70}")

    standardized_files = []
    segments = []
    current_frame = 0

    for idx, v_path in enumerate(video_paths):
        v_path = os.path.abspath(v_path)
        if not os.path.isfile(v_path):
            raise FileNotFoundError(f"Input video not found: {v_path}")

        v_name = os.path.splitext(os.path.basename(v_path))[0]
        inferred_cls = infer_class_name(v_path)
        temp_clip = os.path.join(temp_dir, f"clip_{idx:03d}_{v_name}.mp4")

        # High-speed standardization transcode: scale + pad + 30fps + closed GOP
        vf_filter = (
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
            f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
        )
        cmd = [
            ffmpeg_exe, "-y",
            "-i", v_path,
            "-vf", vf_filter,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "18",
            "-r", str(target_fps),
            "-g", "60",
            "-pix_fmt", "yuv420p",
            "-an",
            temp_clip
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg transcode failed for {v_path}: {res.stderr}")

        n_frames, w, h, fps = probe_video_info(temp_clip, ffmpeg_exe)
        start_frame = current_frame
        end_frame = current_frame + n_frames - 1

        seg_meta = {
            "segment_idx": idx,
            "source_video_path": v_path,
            "source_video_name": os.path.basename(v_path),
            "ground_truth_class": inferred_cls,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "num_frames": n_frames,
            "start_time_s": round(start_frame / target_fps, 2),
            "end_time_s": round((end_frame + 1) / target_fps, 2),
            "duration_s": round(n_frames / target_fps, 2)
        }
        segments.append(seg_meta)
        standardized_files.append(temp_clip)

        print(f"  [{idx+1:02d}/{len(video_paths)}] {inferred_cls.upper()} -> {os.path.basename(v_path)} "
              f"(Frames {start_frame}..{end_frame}, {seg_meta['duration_s']}s)")
        current_frame += n_frames

    # Concatenate using FFmpeg concat demuxer
    concat_txt = os.path.join(temp_dir, "concat_list.txt")
    with open(concat_txt, "w", encoding="utf-8") as f:
        for s_file in standardized_files:
            clean_path = os.path.abspath(s_file).replace("\\", "/")
            f.write(f"file '{clean_path}'\n")

    print(f"\n🔗 Stitching into single continuous stream: {output_merged_path}")
    concat_cmd = [
        ffmpeg_exe, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_txt,
        "-c", "copy",
        output_merged_path
    ]
    res = subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg concat failed: {res.stderr}")

    total_frames, _, _, _ = probe_video_info(output_merged_path, ffmpeg_exe)
    print(f"✅ Merged video ready: {total_frames} frames ({round(total_frames / target_fps, 1)} seconds)")

    # Clean up temporary standardized clips
    shutil.rmtree(temp_dir, ignore_errors=True)

    return output_merged_path, segments


def build_frame_actions(clips, num_frames_total, max_extend=15):
    """Maps sliding-window clip predictions to continuous frame-by-frame action labels."""
    frame_actions = [None] * num_frames_total
    for clip in clips:
        for f in range(clip['start_frame'], min(clip['end_frame'] + 1, num_frames_total)):
            current = frame_actions[f]
            if current is None or clip['confidence'] > current[1]:
                frame_actions[f] = (clip['class_name'], clip['confidence'])

    if not clips:
        return frame_actions

    # Fill uncovered frames from the nearest clip within max_extend
    centers = np.array([(c['start_frame'] + c['end_frame']) / 2 for c in clips])
    for f in range(num_frames_total):
        if frame_actions[f] is not None:
            continue
        nearest = int(np.argmin(np.abs(centers - f)))
        clip = clips[nearest]
        dist = max(clip['start_frame'] - f, f - clip['end_frame'], 0)
        if dist <= max_extend:
            frame_actions[f] = (clip['class_name'], clip['confidence'])
    return frame_actions


def put_label(img, text, x, y, bg_color, scale, thickness, text_color=(0, 0, 0)):
    """Draws text on a filled background rectangle with boundary safety."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    h, w = img.shape[:2]
    x = int(max(0, min(x, w - tw - 12)))
    y = int(max(th + baseline + 8, min(y, h - 4)))
    cv2.rectangle(img, (x, y - th - baseline - 8), (x + tw + 10, y + 2), bg_color, -1)
    cv2.putText(img, text, (x + 5, y - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, text_color, thickness, cv2.LINE_AA)


def render_merged_annotated_video(
    video_path,
    out_path,
    boxes_by_frame,
    frame_actions,
    segments,
    overall_action,
    coded_width,
    coded_height,
    box_mode="largest"
):
    """
    Renders the continuous merged video with:
      - Tracked person bounding box (bright green).
      - Action prediction pill above the person.
      - Top-left HUD banner showing current segment, source class, and active timeline action.
      - Bottom frame counter and timestamp.
    """
    reader = imageio.get_reader(video_path, "ffmpeg")
    meta = reader.get_meta_data()
    fps = meta.get("fps", 30) or 30
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8, macro_block_size=1)

    frame_idx = 0
    seg_lookup = {}
    for seg in segments:
        for f in range(seg["start_frame"], seg["end_frame"] + 1):
            seg_lookup[f] = seg

    try:
        for rgb in reader:
            img = np.ascontiguousarray(rgb)
            h, w = img.shape[:2]
            sx, sy = w / float(coded_width), h / float(coded_height)
            scale = max(0.5, min(w, h) / 950.0)
            thick = max(2, int(round(min(w, h) / 360)))

            boxes = boxes_by_frame.get(frame_idx, [])
            action = frame_actions[frame_idx] if frame_idx < len(frame_actions) else None
            curr_seg = seg_lookup.get(frame_idx, None)

            # Draw tracked bounding boxes
            if boxes:
                areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                main_idx = int(np.argmax(areas))
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    p1 = (int(x1 * sx), int(y1 * sy))
                    p2 = (int(x2 * sx), int(y2 * sy))
                    cv2.rectangle(img, p1, p2, (0, 255, 0), thick)

                    is_actor = box_mode == "union" or i == main_idx
                    if is_actor and action is not None:
                        text = f"{action[0].upper()} {action[1]:.0%}"
                        put_label(img, text, p1[0], p1[1] - 6, (0, 255, 0), scale, max(1, thick - 1))
                    else:
                        put_label(img, "person", p1[0], p1[1] - 6, (0, 220, 220), scale, max(1, thick - 1))

            # Segment HUD banner
            if curr_seg:
                seg_banner = f"Segment {curr_seg['segment_idx']+1}/{len(segments)}: {curr_seg['ground_truth_class'].upper()}"
                put_label(img, seg_banner, 16, int(45 * scale) + 12, (255, 200, 0), scale * 1.1, thick)

                act_str = f"Recognized Action: {action[0].upper()} ({action[1]:.0%})" if action else "Recognized Action: Tracking..."
                put_label(img, act_str, 16, int(95 * scale) + 12, (0, 255, 128), scale * 1.0, thick)

            # Bottom timestamp
            sec = frame_idx / float(fps)
            time_str = f"Time: {int(sec // 60):02d}:{sec % 60:04.1f} | Frame {frame_idx}"
            put_label(img, time_str, 16, h - 14, (60, 60, 60), scale * 0.75, max(1, thick - 1), text_color=(255, 255, 255))

            writer.append_data(img)
            frame_idx += 1
            if frame_idx % 250 == 0:
                print(f"  Rendered {frame_idx} frames...")
    finally:
        reader.close()
        writer.close()

    return frame_idx


def evaluate_segments(segments, clips):
    """Computes action recognition accuracy and top prediction for each video segment."""
    segment_eval = []
    correct_count = 0

    for seg in segments:
        s_start, s_end = seg["start_frame"], seg["end_frame"]
        gt_cls = seg["ground_truth_class"].lower()

        # Gather clips whose midpoint falls in this segment
        seg_clips = [
            c for c in clips
            if s_start <= (c["start_frame"] + c["end_frame"]) / 2 <= s_end
        ]

        if seg_clips:
            # Average class probabilities across clips in this segment
            class_votes = {}
            for c in seg_clips:
                c_name = c["class_name"].lower()
                c_conf = c["confidence"]
                class_votes[c_name] = class_votes.get(c_name, 0.0) + c_conf

            pred_cls = max(class_votes.keys(), key=lambda k: class_votes[k])
            pred_conf = class_votes[pred_cls] / sum(class_votes.values())
        else:
            pred_cls = "none"
            pred_conf = 0.0

        is_correct = (pred_cls == gt_cls)
        if is_correct:
            correct_count += 1

        seg_res = {
            "segment_idx": seg["segment_idx"],
            "source_video": seg["source_video_name"],
            "ground_truth_class": gt_cls,
            "predicted_action": pred_cls,
            "confidence": round(pred_conf, 4),
            "is_correct": is_correct,
            "num_clips": len(seg_clips),
            "start_frame": s_start,
            "end_frame": s_end,
            "duration_s": seg["duration_s"]
        }
        segment_eval.append(seg_res)

    accuracy = correct_count / len(segments) if segments else 0.0
    return segment_eval, accuracy


def run_multi_pipeline(
    video_paths,
    output_name="multi_video_merged",
    output_dir=os.path.join(BASE_DIR, "results", "pipeline_output"),
    detector_weights=os.path.join(BASE_DIR, "models", "best_model.h5"),
    action_model=os.path.join(ACTION_DIR, "models", "best_model.pth"),
    conf_thresh=0.5,
    device="cuda",
    clip_stride=None,
    keep_intermediate=False,
    show=False,
    preview=False
):
    t_start = time.time()
    out_dir = os.path.join(output_dir, output_name)
    bbox_dir = os.path.join(out_dir, "bboxes")
    mv_root = os.path.join(out_dir, "motion_vectors")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'#' * 75}\n# MULTI-VIDEO CONTINUOUS ACTION RECOGNITION PIPELINE: {output_name}\n{'#' * 75}")

    # Stage 1: Merge all input videos into a single continuous stream
    merged_video_path = os.path.join(out_dir, f"{output_name}_input_merged.mp4")
    merged_video_path, segments = standardize_and_merge_videos(
        video_paths=video_paths,
        output_merged_path=merged_video_path,
        target_width=1080,
        target_height=1920,
        target_fps=30.0
    )

    # Stage 2: Load action recognition model
    print("\n[Stage A] Loading Action Recognition Model (TemporalCNN)...")
    recognizer = ActionRecognizer(model_path=action_model, device=device, clip_stride=clip_stride)
    pad_ratio = recognizer.pad_ratio
    print(f"  Supported Classes: {recognizer.class_names}")

    # Stage 3: Compressed-domain preprocessing on the merged continuous video
    print("\n[Stage B] Single-Pass Bitstream Extraction -> I-Frame Detection -> MV Propagation -> MV Crop")
    preprocessor = ActionDatasetPreprocessor(
        model_weights=detector_weights,
        conf_thresh=conf_thresh,
        padding_ratio=pad_ratio
    )
    info = preprocessor.process_single_video(
        video_path=merged_video_path,
        output_bbox_dir=bbox_dir,
        output_mv_dir=mv_root,
        keep_temp=False,
        video_name=output_name
    )
    if not info:
        raise RuntimeError("Compressed-domain extraction failed on merged video.")

    width, height = info["width"], info["height"]
    bbox_csv = os.path.join(bbox_dir, f"{output_name}_bboxes.csv")
    mv_dir = os.path.join(mv_root, output_name)
    boxes_by_frame = load_bbox_csv(bbox_csv)
    print(f"  Human BBoxes tracked on {len(boxes_by_frame)} / {info['num_frames']} frames")

    # Stage 4: Run Action Recognition on continuous timeline
    print("\n[Stage C] Running Action Recognition on Continuous Motion Vector Stream...")
    result = None
    try:
        result = recognizer.predict(mv_dir, bbox_csv, width=width, height=height)
        print(f"  Overall Stream Action: {result['predicted_class_name']} "
              f"({result['confidence']:.2%}) over {len(result['clips'])} clips")
        for p in result["top_5"]:
            print(f"    {p['rank']}. {p['class_name']}: {p['probability']:.4f}")
    except ValueError as e:
        print(f"  [Warning] Action recognition warning: {e}")

    clips = result["clips"] if result else []
    frame_actions = build_frame_actions(clips, info["num_frames"])
    overall = {"name": result["predicted_class_name"], "confidence": result["confidence"]} if result else None

    # Stage 5: Evaluate each individual segment in the continuous stream
    print("\n[Stage D] Evaluating Per-Segment Action Recognition across Timeline...")
    segment_eval, segment_accuracy = evaluate_segments(segments, clips)
    print(f"\n📊 Segment Performance Summary:")
    print(f"{'Seg':<4} | {'Source Video':<24} | {'Ground Truth':<22} | {'Predicted Action':<22} | {'Conf':<7} | {'Result'}")
    print("-" * 95)
    for s in segment_eval:
        status = "✅ PASS" if s["is_correct"] else "❌ FAIL"
        print(f"{s['segment_idx']+1:<4} | {s['source_video'][:24]:<24} | {s['ground_truth_class'][:22]:<22} | "
              f"{s['predicted_action'][:22]:<22} | {s['confidence']:>6.1%} | {status}")
    print("-" * 95)
    print(f"🎯 Multi-Class Segment Accuracy: {segment_accuracy:.1%} ({sum(1 for s in segment_eval if s['is_correct'])} / {len(segment_eval)} correct)")

    # Stage 6: Render single continuous annotated video
    print("\n[Stage E] Rendering Unified Continuous Annotated Output Video...")
    out_video = os.path.join(out_dir, f"{output_name}_action.mp4")
    n_rendered = render_merged_annotated_video(
        video_path=merged_video_path,
        out_path=out_video,
        boxes_by_frame=boxes_by_frame,
        frame_actions=frame_actions,
        segments=segments,
        overall_action=overall,
        coded_width=width,
        coded_height=height,
        box_mode=recognizer.box_mode
    )

    # Stage 7: Save summary JSON and CSV
    summary = {
        "output_name": output_name,
        "merged_video_input": merged_video_path,
        "output_annotated_video": out_video,
        "total_frames": n_rendered,
        "total_duration_s": round(n_rendered / 30.0, 2),
        "total_segments": len(segments),
        "segment_accuracy": round(segment_accuracy, 4),
        "overall_stream_action": overall,
        "segment_breakdown": segment_eval,
        "runtime_s": round(time.time() - t_start, 2)
    }
    result_json = os.path.join(out_dir, f"{output_name}_result.json")
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not keep_intermediate:
        shutil.rmtree(mv_root, ignore_errors=True)

    print(f"\n{'=' * 75}")
    print("🎉 MULTI-VIDEO PIPELINE COMPLETE!")
    print(f"  Merged Output Video : {out_video}")
    print(f"  Results JSON Report : {result_json}")
    print(f"  Segment Accuracy    : {segment_accuracy:.1%}")
    print(f"  Total Runtime       : {summary['runtime_s']} s")
    print(f"{'=' * 75}")

    # Optional Preview or OS player
    if show:
        try:
            if hasattr(os, "startfile"):
                os.startfile(out_video)
        except Exception as e:
            print(f"  [Warning] Could not automatically open video: {e}")

    if preview:
        try:
            from play_video import main as play_main
            print("  Opening interactive video player...")
            cap = cv2.VideoCapture(out_video)
            if cap.isOpened():
                win_name = "Multi-Video Action Recognition (Space = Pause, 'q' = Quit)"
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                delay = max(1, int(1000 / fps))
                paused = False
                while True:
                    if not paused:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        cv2.imshow(win_name, frame)
                    k = cv2.waitKey(1 if paused else delay) & 0xFF
                    if k in (27, ord('q'), ord('Q')):
                        break
                    elif k == 32:
                        paused = not paused
                cap.release()
                cv2.destroyAllWindows()
        except Exception as e:
            print(f"  [Warning] Interactive preview window unavailable: {e}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Video, Multi-Class Compressed-Domain Action Recognition Pipeline"
    )
    parser.add_argument("--videos", nargs="+", default=[],
                        help="List of input video paths (space separated)")
    parser.add_argument("--video_list", default=None,
                        help="Path to a text file containing video paths (one path per line)")
    parser.add_argument("--output_name", default="multi_class_merged",
                        help="Folder and filename prefix for the output")
    parser.add_argument("--output_dir", default=os.path.join(BASE_DIR, "results", "pipeline_output"),
                        help="Base output directory")
    parser.add_argument("--detector_weights", default=os.path.join(BASE_DIR, "models", "best_model.h5"),
                        help="Path to SSD300_H264 weights")
    parser.add_argument("--action_model", default=os.path.join(ACTION_DIR, "models", "best_model.pth"),
                        help="Path to TemporalCNN checkpoint")
    parser.add_argument("--conf_thresh", type=float, default=0.5,
                        help="I-frame person detection confidence threshold")
    parser.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")
    parser.add_argument("--clip_stride", type=int, default=None,
                        help="Sliding window stride for action clips")
    parser.add_argument("--keep_intermediate", action="store_true",
                        help="Keep cropped MV .npy files")
    parser.add_argument("--show", action="store_true",
                        help="Automatically open the annotated video in default player")
    parser.add_argument("--preview", action="store_true",
                        help="Launch interactive OpenCV playback window immediately")
    args = parser.parse_args()

    collected_paths = list(args.videos)
    if args.video_list:
        if not os.path.isfile(args.video_list):
            raise FileNotFoundError(f"Video list file not found: {args.video_list}")
        with open(args.video_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().strip('"').strip("'")
                if line and not line.startswith("#"):
                    collected_paths.append(line)

    if not collected_paths:
        parser.error("No input videos provided! Use --videos <v1> <v2> ... or --video_list <file.txt>")

    run_multi_pipeline(
        video_paths=collected_paths,
        output_name=args.output_name,
        output_dir=args.output_dir,
        detector_weights=args.detector_weights,
        action_model=args.action_model,
        conf_thresh=args.conf_thresh,
        device=args.device,
        clip_stride=args.clip_stride,
        keep_intermediate=args.keep_intermediate,
        show=args.show,
        preview=args.preview
    )


if __name__ == "__main__":
    main()
