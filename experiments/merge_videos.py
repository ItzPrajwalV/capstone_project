#!/usr/bin/env python3
"""
merge_videos.py
===============
Merges 10 videos from each of the 7 HAR classes (70 videos total) into one
continuous long-duration video, preserving exact source-video boundaries,
frame ranges, and class labels in a companion JSON manifest.

Usage:
    python experiments/merge_videos.py
    python experiments/merge_videos.py --videos_per_class 10 --output_dir experiments/results
"""

import os
import sys
import re
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path

# Ensure UTF-8 output on Windows
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

import imageio_ffmpeg

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = r"C:\Users\PESU-RF\Downloads\Human Activity Recognition - Video Dataset"

CLASSES = [
    "Clapping",
    "Meet and Split",
    "Sitting",
    "Standing Still",
    "Walking",
    "Walking While Reading Book",
    "Walking While Using Phone"
]

def natural_sort_key(s):
    """Sort strings containing numbers in human order (e.g. Walking (2) before Walking (10))."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def get_video_frame_count_and_info(video_path, ffmpeg_exe):
    """Probes frame count, width, height, and fps using ffprobe if available or opencv."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0, 0, 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count, width, height, fps

def select_videos(dataset_dir, classes, videos_per_class=10):
    """Selects exactly N videos per class from the dataset."""
    selected = {}
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Class folder not found: {cls_dir}")
        vids = [
            os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]
        vids.sort(key=natural_sort_key)
        if len(vids) < videos_per_class:
            raise ValueError(f"Class {cls} only has {len(vids)} videos; requested {videos_per_class}")
        selected[cls] = vids[:videos_per_class]
    return selected

def standardize_and_merge(
    selected_videos,
    output_video_path,
    manifest_path,
    target_width=1080,
    target_height=1920,
    target_fps=30.0
):
    """
    Standardizes each selected video to uniform resolution and frame rate,
    concatenates them, and outputs the merged video + boundary manifest.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    temp_dir = os.path.join(os.path.dirname(output_video_path), "temp_merge_clips")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    print(f"🎬 Starting Long-Duration Video Merge...")
    print(f"  Target Resolution : {target_width}x{target_height}")
    print(f"  Target Frame Rate : {target_fps} fps")
    print(f"  Output Video      : {output_video_path}")
    print(f"  Manifest File     : {manifest_path}")

    standardized_files = []
    segments = []
    current_frame = 0
    seg_idx = 0
    t0 = time.time()

    try:
        for cls, vids in selected_videos.items():
            print(f"\n📂 Processing Class: {cls} ({len(vids)} videos)")
            for v_path in vids:
                v_name = os.path.splitext(os.path.basename(v_path))[0]
                temp_out = os.path.join(temp_dir, f"clip_{seg_idx:03d}_{v_name}.mp4")

                # Transcode to uniform parameters:
                # -vf scales to fit within target and pads with black borders (preserves aspect ratio)
                # -r 30 ensures exact 30fps
                # -g 60 sets GOP size (I-frame every 2s)
                # -c:v libx264 -preset ultrafast -crf 18 ensures fast high-quality encode
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
                    "-pix_fmt", "yuv420p",
                    "-an",
                    temp_out
                ]
                res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                if res.returncode != 0:
                    raise RuntimeError(f"FFmpeg transcode failed for {v_path}: {res.stderr}")

                # Measure exact frame count of the standardized clip
                n_frames, w, h, fps = get_video_frame_count_and_info(temp_out, ffmpeg_exe)
                if n_frames <= 0:
                    raise RuntimeError(f"Could not read frames from {temp_out}")

                start_frame = current_frame
                end_frame = current_frame + n_frames - 1
                start_time_s = start_frame / target_fps
                end_time_s = (end_frame + 1) / target_fps

                segment_meta = {
                    "segment_idx": seg_idx,
                    "class_name": cls,
                    "source_video_path": os.path.abspath(v_path),
                    "source_video_name": os.path.basename(v_path),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "num_frames": n_frames,
                    "start_time_s": round(start_time_s, 3),
                    "end_time_s": round(end_time_s, 3),
                    "duration_s": round(n_frames / target_fps, 3)
                }
                segments.append(segment_meta)
                standardized_files.append(temp_out)

                print(f"  [{seg_idx+1:02d}/70] {cls} -> {os.path.basename(v_path)}: "
                      f"frames {start_frame}..{end_frame} ({n_frames} frames, {segment_meta['duration_s']}s)")

                current_frame += n_frames
                seg_idx += 1

        # Concatenate using concat demuxer
        concat_txt = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_txt, "w", encoding="utf-8") as f:
            for s_file in standardized_files:
                # Use forward slashes for ffmpeg concat list
                clean_path = os.path.abspath(s_file).replace("\\", "/")
                f.write(f"file '{clean_path}'\n")

        print(f"\n🔗 Concatenating {len(standardized_files)} standardized clips into {output_video_path}...")
        concat_cmd = [
            ffmpeg_exe, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_txt,
            "-c", "copy",
            output_video_path
        ]
        res = subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg concat failed: {res.stderr}")

        total_frames, final_w, final_h, final_fps = get_video_frame_count_and_info(output_video_path, ffmpeg_exe)

        manifest = {
            "title": "Long-Duration 70-Video HAR Benchmark Dataset",
            "experiment": "Experiment 2: Long-Duration Video Evaluation",
            "total_segments": len(segments),
            "videos_per_class": len(selected_videos[CLASSES[0]]),
            "classes": CLASSES,
            "total_frames": total_frames,
            "total_duration_s": round(total_frames / target_fps, 3),
            "resolution": [target_width, target_height],
            "fps": target_fps,
            "output_video": os.path.abspath(output_video_path),
            "created_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "segments": segments
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        elapsed = time.time() - t0
        print(f"\n{'=' * 70}")
        print(f"✅ MERGE COMPLETE in {elapsed:.1f} s")
        print(f"  Total Segments : {len(segments)}")
        print(f"  Total Frames   : {total_frames} (~{manifest['total_duration_s']:.1f} s)")
        print(f"  Resolution     : {target_width}x{target_height} @ {target_fps} fps")
        print(f"  Merged Video   : {output_video_path}")
        print(f"  Manifest File  : {manifest_path}")
        print(f"{'=' * 70}")

        return manifest

    finally:
        # Clean up temporary standardized clips
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Merge 70 HAR dataset videos into one continuous long video.")
    parser.add_argument("--dataset_dir", default=DATASET_DIR, help="Path to HAR Video Dataset directory")
    parser.add_argument("--videos_per_class", type=int, default=10, help="Number of videos per class (default: 10)")
    parser.add_argument("--output_dir", default=os.path.join(BASE_DIR, "experiments", "results"),
                        help="Directory to save merged video and manifest")
    parser.add_argument("--output_name", default="long_duration_70videos.mp4", help="Merged video filename")
    parser.add_argument("--width", type=int, default=1080, help="Target video width")
    parser.add_argument("--height", type=int, default=1920, help="Target video height")
    parser.add_argument("--fps", type=float, default=30.0, help="Target frame rate")
    args = parser.parse_args()

    selected = select_videos(args.dataset_dir, CLASSES, args.videos_per_class)
    output_video_path = os.path.join(args.output_dir, args.output_name)
    manifest_path = os.path.join(args.output_dir, "long_duration_manifest.json")

    standardize_and_merge(
        selected,
        output_video_path,
        manifest_path,
        target_width=args.width,
        target_height=args.height,
        target_fps=args.fps
    )

if __name__ == "__main__":
    main()
