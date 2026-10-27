"""
play_video.py
=============
Quickly preview any output video in an on-screen OpenCV playback window.
Controls:
  - Space: Pause / Resume
  - 'q' or ESC: Exit window
"""

import os
import sys
import glob
import cv2

def main():
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Default to latest output video
        candidates = sorted(
            glob.glob("results/pipeline_output/*/*_action.mp4"),
            key=os.path.getmtime,
            reverse=True,
        )
        if not candidates:
            print("No generated action videos found in results/pipeline_output/")
            return
        video_path = candidates[0]

    video_path = os.path.abspath(video_path)
    if not os.path.isfile(video_path):
        print(f"File not found: {video_path}")
        return

    print(f"Playing: {video_path}")
    print("Controls: Space = Pause/Play | 'q' or ESC = Close")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        return

    win_name = f"Action Recognition - {os.path.basename(video_path)} (Press 'q' to exit)"
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

        key = cv2.waitKey(1 if paused else delay) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key == 32:  # Space
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
