# Compressed-domain detection vs RGB YOLO - computational benchmark

## Environment

- **OS**: Windows 10
- **CPU**: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel
- **CPU cores (logical)**: 32
- **RAM (GB)**: 127.7
- **GPU**: NVIDIA RTX 4500 Ada Generation
- **TensorFlow**: 2.21.0 (SSD device: cpu)
- **PyTorch**: 2.6.0+cu124
- **Python**: 3.11.0

## 1. Model complexity

| Model | Input | Params (M) | GFLOPs / inference | Weights file (MB) | Median latency (ms) |
|---|---|---|---|---|---|
| SSD300_H264 (ours) | 300x300x3 DCT energy map | 44.29 | 3.47 | 506.91 | cpu: 11.49 |
| yolov8n | 640x640x3 RGB | 3.16 | 8.74 | 6.25 | cpu: 33.29, cuda: 3.53 |

GFLOPs = 2 x multiply-accumulates of all convolution and dense layers, counted the same way for both models.

## 2. End-to-end results

### Walking (1)  (1080x1920, 303 frames, 30.00 fps, 10.1 s, bitstream 9.58 MB)

| Pipeline | Device | Detector calls | Total time (s) | FPS | ms/frame | Real-time x | RT streams | Detector GFLOPs (video) | GFLOPs / video-s | Data to vision stack (MB) | CPU-s | Extra RAM (MB) | GPU mem (MB) | Energy (J) | F1@0.5 | mIoU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OURS (DCT-SSD + MV, original code) | cpu | 2 | 78.19 | 3.88 | 258.05 | 0.13 | 0.13 | 6.94 | 0.69 | 54.61 | 122.89 | 1,206 | - | - | - | - |
| OURS-VEC (DCT-SSD + MV, vectorized) | cpu | 2 | 3.99 | 75.99 | 13.16 | 2.53 | 2.53 | 6.94 | 0.69 | 54.52 | 4.47 | 700.89 | - | - | - | - |
| yolov8n-FULL (RGB every frame) | cpu | 303 | 14.17 | 21.38 | 46.77 | 0.71 | 0.71 | 2,649 | 262.32 | 1,798 | 36.09 | 379.04 | - | - | - | - |
| yolov8n-KEY (RGB keyframes + MV) | cpu | 2 | 3.95 | 76.80 | 13.02 | 2.56 | 2.56 | 17.49 | 1.73 | 63.40 | 1.66 | 45.45 | - | - | - | - |
| yolov8n-FULL (RGB every frame) | cuda | 303 | 12.42 | 24.39 | 41.01 | 0.81 | 0.81 | 2,649 | 262.32 | 1,798 | 21.16 | 388.43 | 27.43 | 125.09 | - | - |
| yolov8n-KEY (RGB keyframes + MV) | cuda | 2 | 3.77 | 80.29 | 12.45 | 2.68 | 2.68 | 17.49 | 1.73 | 63.40 | 1.53 | 23.79 | 27.43 | 25.58 | - | - |

**Stage breakdown (s)**

- OURS (DCT-SSD + MV, original code) (cpu): 1_ffprobe_frame_types = 1.284, 2_iframe_coeff_extraction_ffmpeg = 4.156, 3_dct_feature_map_build = 3.014, 4a_mv_extraction_ffmpeg = 1.341, 4b_ffprobe_p_frame_index = 1.613, 4c_mv_bin_parse_python = 1.717, 4d_mv_npy_write_python = 57.737, 5_ssd_detection_iframes = 0.210, 6_mv_box_propagation = 7.118
- OURS-VEC (DCT-SSD + MV, vectorized) (cpu): 1_ffprobe_frame_types = 1.352, 2_single_pass_extraction_ffmpeg = 1.344, 3_dct_feature_map_build_numpy = 0.064, 4_mv_parse_numpy = 1.063, 5_ssd_detection_iframes = 0.090, 6_mv_box_propagation = 0.073
- yolov8n-FULL (RGB every frame) (cpu): 1_rgb_decode_all_frames = 3.644, 2_yolo_detection_all_frames = 10.526
- yolov8n-KEY (RGB keyframes + MV) (cpu): 1_ffprobe_keyframe_index = 1.321, 2_rgb_decode_keyframes = 0.065, 3_yolo_detection_keyframes = 0.093, 2_single_pass_extraction_ffmpeg (from A) = 1.344, 4_mv_parse_numpy (from A) = 1.063, 5_mv_box_propagation = 0.059
- yolov8n-FULL (RGB every frame) (cuda): 1_rgb_decode_all_frames = 2.807, 2_yolo_detection_all_frames = 9.618
- yolov8n-KEY (RGB keyframes + MV) (cuda): 1_ffprobe_keyframe_index = 1.210, 2_rgb_decode_keyframes = 0.061, 3_yolo_detection_keyframes = 0.026, 2_single_pass_extraction_ffmpeg (from A) = 1.344, 4_mv_parse_numpy (from A) = 1.063, 5_mv_box_propagation = 0.071

**Time by cost category (s)**

| Pipeline | Device | Codec / bitstream (FFmpeg, ffprobe) | Python data handling (prototype overhead) | Detector (model + NMS + pre/post) | MV box tracking |
|---|---|---|---|---|---|
| OURS (DCT-SSD + MV, original code) | cpu | 8.394 | 62.468 | 0.210 | 7.118 |
| OURS-VEC (DCT-SSD + MV, vectorized) | cpu | 2.696 | 1.127 | 0.090 | 0.073 |
| yolov8n-FULL (RGB every frame) | cpu | 3.644 | 0.000 | 10.526 | 0.000 |
| yolov8n-KEY (RGB keyframes + MV) | cpu | 2.730 | 1.063 | 0.093 | 0.059 |
| yolov8n-FULL (RGB every frame) | cuda | 2.807 | 0.000 | 9.618 | 0.000 |
| yolov8n-KEY (RGB keyframes + MV) | cuda | 2.615 | 1.063 | 0.026 | 0.071 |

**OURS-VEC (DCT-SSD + MV, vectorized) relative to each other row** (>1 = ours better)

| Baseline | Device | Speed-up (time) | Detector-FLOPs reduction | Detector-call reduction | Data-volume reduction | CPU-s reduction | F1 difference (ours - baseline) |
|---|---|---|---|---|---|---|---|
| OURS (DCT-SSD + MV, original code) | cpu | 19.61x | 1.00x | 1.00x | 1.00x | 27.50x | - |
| yolov8n-FULL (RGB every frame) | cpu | 3.55x | 381.57x | 151.50x | 32.97x | 8.08x | - |
| yolov8n-KEY (RGB keyframes + MV) | cpu | 0.99x | 2.52x | 1.00x | 1.16x | 0.37x | - |
| yolov8n-FULL (RGB every frame) | cuda | 3.12x | 381.57x | 151.50x | 32.97x | 4.73x | - |
| yolov8n-KEY (RGB keyframes + MV) | cuda | 0.95x | 2.52x | 1.00x | 1.16x | 0.34x | - |

## 3. Notes for interpreting the numbers

- *Total time* of OURS is the sum of its sequential stages. For YOLO-FULL it is RGB decode (measured on its own pass) + YOLO time; the pipelined wall time with decode overlapping inference is in raw_results.json (`overlapped_decode_plus_detect_s`).
- The custom FFmpeg dumps coefficients/MVs right after entropy decoding and skips pixel reconstruction of I-frames, but P/B-frames are still fully reconstructed (the original code also decodes the video twice: one pass for I-frame coefficients, one for MVs; OURS-VEC decodes once). Skipping P/B reconstruction too would make extraction cheaper; report the measured times and present that saving as potential, not achieved.
- OURS (original code) parses the binaries with per-macroblock Python loops; OURS-VEC (fast_compressed_ops.py) does the same computation with NumPy and produces identical feature maps, motion vectors and boxes (verified bit-exact). The difference between the two rows is implementation overhead, not a property of the method.
- The custom FFmpeg must run with `-threads 1`: with frame-threaded decoding its MV dump interleaves records from several threads (corrupted, non-deterministic STRICT_FINAL_P.bin). Both OURS rows use -threads 1 for the MV pass.
- YOLO-KEY uses the same detection schedule and MV tracking as OURS, so OURS vs YOLO-KEY isolates the effect of the DCT input; OURS vs YOLO-FULL shows the gain of the full approach.
- Compare pipelines on the same device. TensorFlow has no native-Windows GPU support (TF >= 2.11), so on Windows the SSD runs on CPU; compare it with the YOLO CPU rows, or run under WSL2/Linux for GPU-vs-GPU numbers.
- Detector FLOPs exclude NMS, pre- and post-processing; those are inside the measured times.
