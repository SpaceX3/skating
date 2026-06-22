# Keyframe Selector

This directory is independent from the existing Skating-Mixer training code. It
selects one static keyframe per 5-second video segment for visual inspection.

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

Example:

```powershell
conda run -n skating-mixer python keyframe_selector/select_keyframes.py `
  --video "D:\University\3fal\skate\FS1000 Dataset\fs1000\2019_Final_MS_Kevin.mp4" `
  --output-dir "D:\University\3fal\skate\skating-best\keyframe_selector\outputs\2019_Final_MS_Kevin" `
  --clip-len 5 `
  --sample-fps 4 `
  --pose-backend auto `
  --proxy http://127.0.0.1:6518
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
