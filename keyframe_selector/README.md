# Keyframe Selector

This directory is independent from the existing Skating-Mixer training code. It
selects one static keyframe per 5-second video window. For training
preprocessing, pass `--feature-path` or `--num-segments` so the selector emits
one keyframe for each overlapping dynamic feature timestep.

The selector uses:

- Laplacian variance as a sharpness score. Frames below the per-segment
  quantile threshold are discarded before ranking.
- Frame difference and Farneback optical flow to score motion-change peaks,
  not raw maximum motion.
- A two-stage jump-event strategy. It first detects likely
  `entry -> takeoff -> air -> landing` events from pose/flow signals. For
  5-second segments that overlap a jump event, it prefers:
  - the clearest high-confidence frame 3-8 frames before takeoff;
  - the clearest frame 0-5 frames after landing with visible hips, knees, and
    ankles.
  Segments without a detected jump fall back to the generic scorer.
  If a 5-second segment contains both takeoff and landing candidates, landing
  has hard priority. The selector falls back to takeoff only when the landing
  frame is very blurry, controlled by `--landing-min-sharpness-ratio`.
- Optional Torchvision Keypoint R-CNN pose speed when pretrained weights are
  available. If the pose model cannot be loaded, the script falls back to
  sharpness + frame difference + optical flow.

Full-run example:

```bash
python -u keyframe_selector/select_keyframes.py \
  --video-root "../FS1000 Dataset/fs1000" \
  --feature-root "../FS1000 Dataset/Timesformer_output_feature_fs800" \
  --output-dir "keyframe_selector/outputs/fs1000_all_frame_indices" \
  --clip-len 5 \
  --sample-fps 3 \
  --event-fps 2 \
  --pose-backend auto \
  --gpu auto \
  --pose-batch-size 8 \
  --decode-backend auto \
  --flow-backend auto \
  --decode-backend auto \
  --flow-backend auto \
  --frames-only \
  --skip-existing \
  --continue-on-error \
  --require-feature
```

Outputs:

- `selected_frames/`: clean best frame for each segment.
- `selected_frames_annotated/`: same frames with score overlays.
- `selected_frame_indices.csv`: compact frame-number output for ResNet preprocessing.
- `selected_frame_indices.txt`: one selected frame number per line.
- `selected_keyframes.csv`: one row per selected segment.
- `candidate_scores.csv`: all candidate frame scores.
- `jump_timeline.csv`: coarse full-video pose/flow timeline used for event detection.
- `jump_events.csv`: detected entry/takeoff/air/landing events.
- `contact_sheet.jpg`: quick overview of selected frames.

Use `--disable-jump-events` to reproduce the previous generic-only behavior.
Use `--generic-use-pose` only if you also want the slower fallback scorer to
run pose estimation; the two-stage jump logic uses pose without this flag.
Use `--frames-only` when the output is meant for `action.py` preprocessing.
Use `--feature-path path/to/Timesformer_feature.npy` or `--num-segments T_dyn`
to preserve the original 5-second overlapping window count. Without these
arguments, the script falls back to non-overlapping 5-second windows, which is
useful for quick visual checks but produces fewer frame rows than the training
feature sequence.
Generic candidate metrics are cached by default across overlapping windows, so
the full run can keep the same `--sample-fps` and `--event-fps` while avoiding
repeated frame decoding, Laplacian, frame-difference, and optical-flow work.
Use `--disable-candidate-cache` only for debugging against the old per-window
implementation.
`--decode-backend auto` tries GPU NVDEC through Decord first and falls back to
OpenCV if unavailable. `--flow-backend auto` tries OpenCV CUDA Farneback and
falls back to CPU if the installed OpenCV was not built with CUDA.
`--decode-backend auto` tries GPU NVDEC through Decord first and falls back to
OpenCV if unavailable. `--flow-backend auto` tries OpenCV CUDA Farneback and
falls back to CPU if the installed OpenCV was not built with CUDA.
Use `--gpu auto` to select the visible GPU with the most free memory, `--gpu 2`
to pin the run to physical GPU 2, or `--gpu cpu` to disable GPU pose/decode.
