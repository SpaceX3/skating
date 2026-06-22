from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class SegmentResult:
    segment_idx: int
    segment_start: float
    segment_end: float
    best_local_idx: int
    best_frame_index: int
    best_time_sec: float
    best_score: float


@dataclass
class TimelineData:
    frame_indices: np.ndarray
    times: np.ndarray
    sharpness_norm: np.ndarray
    diff_energy_norm: np.ndarray
    flow_energy_norm: np.ndarray
    diff_change_norm: np.ndarray
    flow_change_norm: np.ndarray
    motion_energy_norm: np.ndarray
    pose_conf_norm: np.ndarray
    pose_energy_norm: np.ndarray
    pose_change_norm: np.ndarray
    lower_body_visible: np.ndarray
    jump_signal: np.ndarray
    air_signal: np.ndarray


@dataclass
class JumpEvent:
    event_id: int
    entry_time: float
    takeoff_time: float
    air_start_time: float
    air_end_time: float
    landing_time: float
    end_time: float
    entry_frame: int
    takeoff_frame: int
    landing_frame: int
    end_frame: int
    score: float
    takeoff_signal: float
    landing_signal: float
    air_signal: float


LOWER_BODY_KPTS = (11, 12, 13, 14, 15, 16)  # COCO hips, knees, ankles.


class PoseEstimator:
    def __init__(
        self,
        backend: str,
        device: str,
        batch_size: int,
        max_side: int,
        proxy: str = "",
    ) -> None:
        self.backend = backend
        self.device = device
        self.batch_size = batch_size
        self.max_side = max_side
        self.enabled = False
        self.model = None
        self.torch = None

        if backend == "none":
            return

        if proxy:
            os.environ["HTTP_PROXY"] = proxy
            os.environ["HTTPS_PROXY"] = proxy
            os.environ["ALL_PROXY"] = proxy

        try:
            import torch
            from torchvision.models.detection import (
                KeypointRCNN_ResNet50_FPN_Weights,
                keypointrcnn_resnet50_fpn,
            )

            if device.startswith("cuda") and not torch.cuda.is_available():
                print("[pose][warn] CUDA requested but unavailable; using CPU.")
                device = "cpu"

            weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
            model = keypointrcnn_resnet50_fpn(weights=weights)
            model.eval().to(device)

            self.torch = torch
            self.device = device
            self.model = model
            self.enabled = True
            print(f"[pose] Torchvision Keypoint R-CNN loaded on {device}.")
        except Exception as exc:  # noqa: BLE001 - keep the selector usable offline.
            if backend == "torchvision":
                print(f"[pose][warn] failed to load Torchvision pose model: {exc}")
            else:
                print(f"[pose][warn] pose unavailable, fallback to no-pose mode: {exc}")

    def estimate(
        self, frames_bgr: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(frames_bgr)
        pose_conf = np.zeros(n, dtype=np.float32)
        keypoints = np.full((n, 17, 2), np.nan, dtype=np.float32)
        visible = np.zeros((n, 17), dtype=bool)

        if not self.enabled or self.model is None or self.torch is None or n == 0:
            return pose_conf, keypoints, visible

        torch = self.torch

        for start in range(0, n, self.batch_size):
            batch_frames = frames_bgr[start : start + self.batch_size]
            tensors = []
            sizes = []
            for frame_bgr in batch_frames:
                resized = resize_max_side(frame_bgr, self.max_side)
                h, w = resized.shape[:2]
                sizes.append((h, w))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                arr = np.ascontiguousarray(rgb.transpose(2, 0, 1))
                tensor = torch.from_numpy(arr).float().div(255.0).to(self.device)
                tensors.append(tensor)

            with torch.no_grad():
                outputs = self.model(tensors)

            for offset, output in enumerate(outputs):
                out_idx = start + offset
                scores = output.get("scores")
                if scores is None or len(scores) == 0:
                    continue

                scores_np = scores.detach().cpu().numpy()
                best = int(np.argmax(scores_np))
                det_score = float(scores_np[best])
                if det_score < 0.2:
                    continue

                kps = output["keypoints"][best].detach().cpu().numpy()[:, :2]
                h, w = sizes[offset]
                kps[:, 0] /= max(w, 1)
                kps[:, 1] /= max(h, 1)
                keypoints[out_idx] = kps.astype(np.float32)

                if "keypoints_scores" in output:
                    kp_scores = output["keypoints_scores"][best].detach().cpu().numpy()
                    kp_conf = 1.0 / (1.0 + np.exp(-kp_scores))
                    visible[out_idx] = kp_conf > 0.35
                    mean_kp_conf = float(np.mean(kp_conf))
                else:
                    visible[out_idx] = True
                    mean_kp_conf = 1.0

                pose_conf[out_idx] = float(np.clip(det_score * mean_kp_conf, 0.0, 1.0))

        return pose_conf, keypoints, visible


def resize_max_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return frame
    scale = max_side / float(side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def robust_norm(values: np.ndarray, low_q: float = 10.0, high_q: float = 90.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=np.float32)

    valid = values[finite]
    lo = float(np.percentile(valid, low_q))
    hi = float(np.percentile(valid, high_q))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo + 1e-8:
        vmax = float(np.max(valid))
        vmin = float(np.min(valid))
        if vmax <= vmin + 1e-8:
            out = np.zeros_like(values, dtype=np.float32)
            out[finite] = 0.5
            return out
        lo, hi = vmin, vmax

    out = (values - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[~finite] = 0.0
    return out.astype(np.float32)


def local_change_peak(energy: np.ndarray) -> np.ndarray:
    energy = np.asarray(energy, dtype=np.float32)
    n = len(energy)
    if n <= 1:
        return np.zeros(n, dtype=np.float32)

    prev_e = np.empty_like(energy)
    next_e = np.empty_like(energy)
    prev_e[0] = energy[0]
    prev_e[1:] = energy[:-1]
    next_e[-1] = energy[-1]
    next_e[:-1] = energy[1:]

    curvature = np.abs(next_e - 2.0 * energy + prev_e)
    slope = np.abs(next_e - prev_e)
    prominence = np.maximum(0.0, energy - 0.5 * (prev_e + next_e))
    return (0.45 * curvature + 0.35 * slope + 0.20 * prominence).astype(np.float32)


def candidate_indices(
    fps: float,
    total_frames: int,
    segment_start: float,
    segment_end: float,
    sample_fps: float,
) -> np.ndarray:
    start_frame = int(round(segment_start * fps))
    end_frame = int(round(segment_end * fps))
    start_frame = max(0, min(start_frame, max(total_frames - 1, 0)))
    end_frame = max(start_frame + 1, min(end_frame, total_frames))
    step = max(1, int(round(fps / max(sample_fps, 1e-6))))
    indices = np.arange(start_frame, end_frame, step, dtype=np.int64)
    if len(indices) == 0:
        indices = np.array([(start_frame + end_frame - 1) // 2], dtype=np.int64)
    return np.unique(np.clip(indices, 0, total_frames - 1))


def read_frames(video_path: str, frame_indices: Sequence[int]) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[warn] failed to read frame {frame_idx}")
            continue
        frames.append(frame)
    cap.release()
    return frames


def make_gray_frames(frames_bgr: Sequence[np.ndarray], max_width: int = 360) -> List[np.ndarray]:
    grays = []
    for frame in frames_bgr:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            gray = cv2.resize(
                gray,
                (max_width, max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        grays.append(gray)
    return grays


def sharpness_scores(frames_bgr: Sequence[np.ndarray]) -> np.ndarray:
    scores = []
    for frame in frames_bgr:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        scores.append(float(lap.var()))
    return np.asarray(scores, dtype=np.float32)


def frame_diff_pair_energy(grays: Sequence[np.ndarray]) -> np.ndarray:
    n = len(grays)
    pair = np.zeros(max(n - 1, 0), dtype=np.float32)
    for i in range(1, n):
        diff = cv2.absdiff(grays[i - 1], grays[i])
        pair[i - 1] = float(np.mean(diff)) / 255.0
    return pair


def flow_pair_energy(grays: Sequence[np.ndarray]) -> np.ndarray:
    n = len(grays)
    pair = np.zeros(max(n - 1, 0), dtype=np.float32)
    for i in range(1, n):
        flow = cv2.calcOpticalFlowFarneback(
            grays[i - 1],
            grays[i],
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        u = flow[..., 0]
        v = flow[..., 1]
        u = u - np.median(u)
        v = v - np.median(v)
        mag = np.sqrt(u * u + v * v)
        pair[i - 1] = float(np.mean(mag))
    return pair


def center_pair_to_candidates(pair: np.ndarray, n: int) -> np.ndarray:
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if n == 1 or len(pair) == 0:
        return np.zeros(n, dtype=np.float32)

    energy = np.zeros(n, dtype=np.float32)
    energy[0] = pair[0]
    energy[-1] = pair[-1]
    if n > 2:
        energy[1:-1] = 0.5 * (pair[:-1] + pair[1:])
    return energy


def pose_pair_energy(
    keypoints: np.ndarray,
    visible: np.ndarray,
    pose_conf: np.ndarray,
    min_visible: int = 5,
) -> np.ndarray:
    n = len(pose_conf)
    pair = np.zeros(max(n - 1, 0), dtype=np.float32)
    for i in range(1, n):
        if pose_conf[i - 1] <= 0.05 or pose_conf[i] <= 0.05:
            continue
        common = visible[i - 1] & visible[i]
        common &= np.isfinite(keypoints[i - 1, :, 0]) & np.isfinite(keypoints[i, :, 0])
        if int(common.sum()) < min_visible:
            continue

        prev = keypoints[i - 1, common]
        curr = keypoints[i, common]
        prev_center = np.mean(prev, axis=0)
        curr_center = np.mean(curr, axis=0)
        rel_prev = prev - prev_center
        rel_curr = curr - curr_center
        rel_speed = np.linalg.norm(rel_curr - rel_prev, axis=1).mean()
        center_speed = float(np.linalg.norm(curr_center - prev_center))
        pair[i - 1] = float(0.6 * rel_speed + 0.4 * center_speed)
    return pair


def lower_body_visibility(visible: np.ndarray, pose_conf: np.ndarray) -> np.ndarray:
    if visible.size == 0:
        return np.zeros(len(pose_conf), dtype=np.float32)
    lower = visible[:, LOWER_BODY_KPTS].mean(axis=1).astype(np.float32)
    return lower * (pose_conf > 0.05).astype(np.float32)


def timeline_rows(timeline: TimelineData) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for i in range(len(timeline.times)):
        rows.append(
            {
                "timeline_idx": i,
                "frame_index": int(timeline.frame_indices[i]),
                "time_sec": float(timeline.times[i]),
                "sharpness_norm": float(timeline.sharpness_norm[i]),
                "frame_diff_energy_norm": float(timeline.diff_energy_norm[i]),
                "flow_energy_norm": float(timeline.flow_energy_norm[i]),
                "frame_diff_change_norm": float(timeline.diff_change_norm[i]),
                "flow_change_norm": float(timeline.flow_change_norm[i]),
                "motion_energy_norm": float(timeline.motion_energy_norm[i]),
                "pose_conf_norm": float(timeline.pose_conf_norm[i]),
                "pose_energy_norm": float(timeline.pose_energy_norm[i]),
                "pose_change_norm": float(timeline.pose_change_norm[i]),
                "lower_body_visible": float(timeline.lower_body_visible[i]),
                "jump_signal": float(timeline.jump_signal[i]),
                "air_signal": float(timeline.air_signal[i]),
            }
        )
    return rows


def jump_event_rows(events: Sequence[JumpEvent]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for event in events:
        rows.append(
            {
                "event_id": event.event_id,
                "entry_time": event.entry_time,
                "takeoff_time": event.takeoff_time,
                "air_start_time": event.air_start_time,
                "air_end_time": event.air_end_time,
                "landing_time": event.landing_time,
                "end_time": event.end_time,
                "entry_frame": event.entry_frame,
                "takeoff_frame": event.takeoff_frame,
                "landing_frame": event.landing_frame,
                "end_frame": event.end_frame,
                "score": event.score,
                "takeoff_signal": event.takeoff_signal,
                "landing_signal": event.landing_signal,
                "air_signal": event.air_signal,
            }
        )
    return rows


def analyze_video_timeline(
    video_path: str,
    fps: float,
    total_frames: int,
    analysis_start: float,
    analysis_end: float,
    event_fps: float,
    pose_estimator: Optional[PoseEstimator],
) -> TimelineData:
    frame_indices = candidate_indices(
        fps=fps,
        total_frames=total_frames,
        segment_start=analysis_start,
        segment_end=analysis_end,
        sample_fps=event_fps,
    )
    frames = read_frames(video_path, frame_indices)
    if len(frames) != len(frame_indices):
        valid_count = min(len(frames), len(frame_indices))
        frames = frames[:valid_count]
        frame_indices = frame_indices[:valid_count]
    if not frames:
        raise RuntimeError("jump-event analysis could not read any frames")

    sharp_raw = sharpness_scores(frames)
    sharp_norm = robust_norm(np.log1p(sharp_raw))
    grays = make_gray_frames(frames)
    diff_energy = center_pair_to_candidates(frame_diff_pair_energy(grays), len(frames))
    flow_energy = center_pair_to_candidates(flow_pair_energy(grays), len(frames))
    diff_change = local_change_peak(diff_energy)
    flow_change = local_change_peak(flow_energy)

    diff_energy_norm = robust_norm(diff_energy)
    flow_energy_norm = robust_norm(flow_energy)
    diff_change_norm = robust_norm(diff_change)
    flow_change_norm = robust_norm(flow_change)
    motion_energy_norm = robust_norm(0.45 * diff_energy + 0.55 * flow_energy)

    pose_conf = np.zeros(len(frames), dtype=np.float32)
    pose_conf_norm = np.zeros(len(frames), dtype=np.float32)
    pose_energy_norm = np.zeros(len(frames), dtype=np.float32)
    pose_change_norm = np.zeros(len(frames), dtype=np.float32)
    lower_visible = np.zeros(len(frames), dtype=np.float32)
    pose_used = False
    if pose_estimator is not None and pose_estimator.enabled:
        pose_conf, keypoints, visible = pose_estimator.estimate(frames)
        pose_energy = center_pair_to_candidates(pose_pair_energy(keypoints, visible, pose_conf), len(frames))
        pose_change = local_change_peak(pose_energy)
        pose_used = bool(np.any(pose_conf > 0.05))
        pose_conf_norm = pose_conf.astype(np.float32) if pose_used else pose_conf_norm
        pose_energy_norm = robust_norm(pose_energy) if pose_used else pose_energy_norm
        pose_change_norm = robust_norm(pose_change) if pose_used else pose_change_norm
        lower_visible = lower_body_visibility(visible, pose_conf) if pose_used else lower_visible

    if pose_used:
        jump_signal = (
            0.40 * flow_change_norm
            + 0.22 * diff_change_norm
            + 0.23 * pose_change_norm
            + 0.15 * motion_energy_norm * np.maximum(lower_visible, 0.25)
        )
        air_signal = (
            0.44 * motion_energy_norm
            + 0.22 * flow_change_norm
            + 0.16 * pose_energy_norm
            + 0.10 * (1.0 - sharp_norm)
            + 0.08 * (1.0 - pose_conf_norm)
        )
    else:
        jump_signal = 0.62 * flow_change_norm + 0.38 * diff_change_norm
        air_signal = 0.62 * motion_energy_norm + 0.23 * flow_change_norm + 0.15 * (1.0 - sharp_norm)

    quality_gate = 0.45 + 0.55 * sharp_norm
    jump_signal = np.clip(jump_signal * quality_gate, 0.0, 1.0).astype(np.float32)
    air_signal = np.clip(air_signal, 0.0, 1.0).astype(np.float32)

    return TimelineData(
        frame_indices=frame_indices,
        times=frame_indices.astype(np.float32) / float(fps),
        sharpness_norm=sharp_norm,
        diff_energy_norm=diff_energy_norm,
        flow_energy_norm=flow_energy_norm,
        diff_change_norm=diff_change_norm,
        flow_change_norm=flow_change_norm,
        motion_energy_norm=motion_energy_norm,
        pose_conf_norm=pose_conf_norm.astype(np.float32),
        pose_energy_norm=pose_energy_norm,
        pose_change_norm=pose_change_norm,
        lower_body_visible=lower_visible,
        jump_signal=jump_signal,
        air_signal=air_signal,
    )


def local_maxima_indices(values: np.ndarray, threshold: float) -> List[int]:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return []
    if len(values) == 1:
        return [0] if values[0] >= threshold else []

    peaks = []
    for i in range(len(values)):
        prev_v = values[i - 1] if i > 0 else values[i]
        next_v = values[i + 1] if i < len(values) - 1 else values[i]
        if values[i] >= threshold and values[i] >= prev_v and values[i] >= next_v:
            peaks.append(i)
    return peaks


def detect_jump_events(
    timeline: TimelineData,
    fps: float,
    threshold_quantile: float,
    min_event_score: float,
    min_air_sec: float,
    max_air_sec: float,
    entry_window_sec: float,
    landing_window_sec: float,
    min_event_gap_sec: float,
) -> List[JumpEvent]:
    n = len(timeline.times)
    if n < 4:
        return []

    threshold = float(np.quantile(timeline.jump_signal, threshold_quantile))
    threshold = max(threshold, float(np.mean(timeline.jump_signal) + 0.25 * np.std(timeline.jump_signal)))
    peaks = local_maxima_indices(timeline.jump_signal, threshold)
    if not peaks:
        return []

    proposed: List[JumpEvent] = []
    for take_idx in sorted(peaks, key=lambda i: float(timeline.jump_signal[i]), reverse=True):
        take_time = float(timeline.times[take_idx])
        search_mask = (
            (timeline.times >= take_time + min_air_sec)
            & (timeline.times <= take_time + max_air_sec)
        )
        landing_candidates = np.where(search_mask)[0]
        if len(landing_candidates) == 0:
            continue

        landing_rank = (
            timeline.jump_signal[landing_candidates]
            + 0.22 * timeline.sharpness_norm[landing_candidates]
            + 0.18 * timeline.lower_body_visible[landing_candidates]
        )
        landing_idx = int(landing_candidates[int(np.argmax(landing_rank))])
        if landing_idx <= take_idx:
            continue

        air_slice = slice(take_idx + 1, landing_idx)
        air_mean = float(np.mean(timeline.air_signal[air_slice])) if landing_idx > take_idx + 1 else 0.0
        score = (
            0.42 * float(timeline.jump_signal[take_idx])
            + 0.42 * float(timeline.jump_signal[landing_idx])
            + 0.16 * air_mean
        )
        if score < max(threshold * 0.90, min_event_score):
            continue

        proposed.append(
            JumpEvent(
                event_id=-1,
                entry_time=max(0.0, take_time - entry_window_sec),
                takeoff_time=take_time,
                air_start_time=float(timeline.times[min(take_idx + 1, n - 1)]),
                air_end_time=float(timeline.times[max(take_idx + 1, landing_idx - 1)]),
                landing_time=float(timeline.times[landing_idx]),
                end_time=float(timeline.times[landing_idx] + landing_window_sec),
                entry_frame=max(0, int(round((take_time - entry_window_sec) * fps))),
                takeoff_frame=int(timeline.frame_indices[take_idx]),
                landing_frame=int(timeline.frame_indices[landing_idx]),
                end_frame=int(round((float(timeline.times[landing_idx]) + landing_window_sec) * fps)),
                score=score,
                takeoff_signal=float(timeline.jump_signal[take_idx]),
                landing_signal=float(timeline.jump_signal[landing_idx]),
                air_signal=air_mean,
            )
        )

    selected: List[JumpEvent] = []
    for event in sorted(proposed, key=lambda item: item.score, reverse=True):
        if any(abs(event.takeoff_time - kept.takeoff_time) < min_event_gap_sec for kept in selected):
            continue
        if any(event.entry_time <= kept.end_time and event.end_time >= kept.entry_time for kept in selected):
            if abs(event.takeoff_time - kept.takeoff_time) < min_event_gap_sec * 1.5:
                continue
        selected.append(event)

    selected.sort(key=lambda item: item.takeoff_time)
    for event_id, event in enumerate(selected):
        event.event_id = event_id
    return selected


def events_for_segment(events: Sequence[JumpEvent], segment_start: float, segment_end: float) -> List[JumpEvent]:
    return [
        event
        for event in events
        if event.entry_time < segment_end and event.end_time >= segment_start
    ]


def score_jump_window(
    frames_bgr: Sequence[np.ndarray],
    frame_indices: np.ndarray,
    fps: float,
    segment_idx: int,
    segment_start: float,
    segment_end: float,
    blur_quantile: float,
    pose_estimator: Optional[PoseEstimator],
    event: JumpEvent,
    phase_name: str,
    require_lower_body: bool,
) -> Tuple[int, List[Dict[str, float]]]:
    sharp_raw = sharpness_scores(frames_bgr)
    sharp_norm = robust_norm(np.log1p(sharp_raw))
    blur_threshold = float(np.quantile(sharp_raw, blur_quantile))
    eligible = sharp_raw >= blur_threshold

    pose_conf = np.zeros(len(frames_bgr), dtype=np.float32)
    lower_visible = np.zeros(len(frames_bgr), dtype=np.float32)
    if pose_estimator is not None and pose_estimator.enabled:
        pose_conf, _, visible = pose_estimator.estimate(frames_bgr)
        lower_visible = lower_body_visibility(visible, pose_conf)

    if require_lower_body and np.any(pose_conf > 0.05):
        phase_score = 0.45 * sharp_norm + 0.25 * pose_conf + 0.30 * lower_visible
        eligible &= lower_visible >= 0.25
    elif np.any(pose_conf > 0.05):
        phase_score = 0.62 * sharp_norm + 0.30 * pose_conf + 0.08 * lower_visible
    else:
        phase_score = sharp_norm

    if not np.any(eligible):
        eligible[np.argmax(phase_score)] = True
    phase_score = phase_score.astype(np.float32)
    phase_score[~eligible] = -np.inf

    best_idx = int(np.argmax(phase_score))
    rows: List[Dict[str, float]] = []
    for i in range(len(frames_bgr)):
        rows.append(
            {
                "segment_idx": segment_idx,
                "segment_start": segment_start,
                "segment_end": segment_end,
                "candidate_local_idx": -1,
                "frame_index": int(frame_indices[i]),
                "time_sec": float(frame_indices[i] / fps),
                "selected": int(i == best_idx),
                "eligible_after_blur_filter": int(eligible[i]),
                "final_score": float(phase_score[i]) if np.isfinite(phase_score[i]) else float("-inf"),
                "sharpness_laplacian_var": float(sharp_raw[i]),
                "blur_threshold": blur_threshold,
                "sharpness_norm": float(sharp_norm[i]),
                "frame_diff_energy_norm": 0.0,
                "flow_energy_norm": 0.0,
                "motion_energy_norm": 0.0,
                "frame_diff_change_norm": 0.0,
                "flow_change_norm": 0.0,
                "pose_conf_norm": float(pose_conf[i]),
                "pose_energy_norm": 0.0,
                "pose_change_norm": 0.0,
                "quality_score": float(phase_score[i]),
                "motion_change_score": 0.0,
                "interior_bias": 0.0,
                "glide_penalty": 0.0,
                "blur_motion_penalty": 0.0,
                "lower_body_visible": float(lower_visible[i]),
                "selection_mode": f"jump_{phase_name}",
                "jump_event_id": event.event_id,
                "jump_phase": phase_name,
                "event_takeoff_time": event.takeoff_time,
                "event_landing_time": event.landing_time,
            }
        )
    return best_idx, rows


def select_jump_phase_frame(
    video_path: str,
    fps: float,
    total_frames: int,
    segment_idx: int,
    segment_start: float,
    segment_end: float,
    events: Sequence[JumpEvent],
    blur_quantile: float,
    pose_estimator: Optional[PoseEstimator],
    segment_blur_threshold: float,
    landing_min_sharpness_ratio: float,
) -> Optional[Tuple[np.ndarray, Dict[str, float], List[Dict[str, float]]]]:
    scored: List[Tuple[str, float, np.ndarray, Dict[str, float], List[Dict[str, float]]]] = []

    for event in events:
        windows = [
            (
                "takeoff_pre",
                event.takeoff_frame - 8,
                event.takeoff_frame - 3,
                False,
                0.00,
            ),
            (
                "landing_post",
                event.landing_frame,
                event.landing_frame + 5,
                True,
                0.04,
            ),
        ]
        for phase_name, start_frame, end_frame, require_lower_body, phase_bonus in windows:
            start_frame = max(0, int(start_frame), int(math.floor(segment_start * fps)))
            end_frame = min(total_frames - 1, int(end_frame), int(math.ceil(segment_end * fps)) - 1)
            if end_frame < start_frame:
                continue
            frame_indices = np.arange(start_frame, end_frame + 1, dtype=np.int64)
            frames = read_frames(video_path, frame_indices)
            if len(frames) != len(frame_indices):
                valid_count = min(len(frames), len(frame_indices))
                frames = frames[:valid_count]
                frame_indices = frame_indices[:valid_count]
            if not frames:
                continue

            best_idx, rows = score_jump_window(
                frames_bgr=frames,
                frame_indices=frame_indices,
                fps=fps,
                segment_idx=segment_idx,
                segment_start=segment_start,
                segment_end=segment_end,
                blur_quantile=blur_quantile,
                pose_estimator=pose_estimator,
                event=event,
                phase_name=phase_name,
                require_lower_body=require_lower_body,
            )
            best_row = rows[best_idx]
            rank_score = float(best_row["final_score"]) + phase_bonus + 0.08 * event.score
            scored.append((phase_name, rank_score, frames[best_idx], best_row, rows))

    if not scored:
        return None

    landing_scored = [item for item in scored if item[0] == "landing_post"]
    takeoff_scored = [item for item in scored if item[0] == "takeoff_pre"]
    landing_scored.sort(key=lambda item: item[1], reverse=True)
    takeoff_scored.sort(key=lambda item: item[1], reverse=True)

    if landing_scored:
        phase_name, _, frame, best_row, rows = landing_scored[0]
        landing_sharpness = float(best_row["sharpness_laplacian_var"])
        landing_floor = float(segment_blur_threshold) * float(landing_min_sharpness_ratio)
        landing_is_very_blurry = landing_sharpness < landing_floor
        if not landing_is_very_blurry or not takeoff_scored:
            best_row["selection_reason"] = (
                "landing_priority"
                if not landing_is_very_blurry
                else "landing_priority_no_takeoff_fallback"
            )
            best_row["segment_blur_threshold"] = float(segment_blur_threshold)
            best_row["landing_blur_floor"] = landing_floor
            best_row["landing_is_very_blurry"] = int(landing_is_very_blurry)
            return frame, best_row, rows

    if takeoff_scored:
        phase_name, _, frame, best_row, rows = takeoff_scored[0]
        best_row["selection_reason"] = "takeoff_fallback_landing_very_blurry"
        best_row["segment_blur_threshold"] = float(segment_blur_threshold)
        best_row["landing_blur_floor"] = float(segment_blur_threshold) * float(landing_min_sharpness_ratio)
        best_row["landing_is_very_blurry"] = 1
        return frame, best_row, rows

    scored.sort(key=lambda item: item[1], reverse=True)
    _, _, frame, best_row, rows = scored[0]
    best_row["selection_reason"] = "jump_phase_best_score"
    best_row["segment_blur_threshold"] = float(segment_blur_threshold)
    best_row["landing_blur_floor"] = float(segment_blur_threshold) * float(landing_min_sharpness_ratio)
    best_row["landing_is_very_blurry"] = ""
    return frame, best_row, rows


def score_segment(
    frames_bgr: Sequence[np.ndarray],
    frame_indices: np.ndarray,
    fps: float,
    segment_idx: int,
    segment_start: float,
    segment_end: float,
    blur_quantile: float,
    pose_estimator: Optional[PoseEstimator],
) -> Tuple[SegmentResult, List[Dict[str, float]]]:
    n = len(frames_bgr)
    if n == 0:
        raise RuntimeError(f"segment {segment_idx} has no readable frames")

    sharp_raw = sharpness_scores(frames_bgr)
    sharp_norm = robust_norm(np.log1p(sharp_raw))
    blur_threshold = float(np.quantile(sharp_raw, blur_quantile))
    eligible = sharp_raw >= blur_threshold
    if not np.any(eligible):
        eligible[np.argmax(sharp_raw)] = True

    grays = make_gray_frames(frames_bgr)
    diff_energy = center_pair_to_candidates(frame_diff_pair_energy(grays), n)
    flow_energy = center_pair_to_candidates(flow_pair_energy(grays), n)

    diff_change = local_change_peak(diff_energy)
    flow_change = local_change_peak(flow_energy)

    diff_energy_norm = robust_norm(diff_energy)
    flow_energy_norm = robust_norm(flow_energy)
    diff_change_norm = robust_norm(diff_change)
    flow_change_norm = robust_norm(flow_change)
    motion_energy_norm = robust_norm(0.45 * diff_energy + 0.55 * flow_energy)

    pose_conf = np.zeros(n, dtype=np.float32)
    pose_change_norm = np.zeros(n, dtype=np.float32)
    pose_energy_norm = np.zeros(n, dtype=np.float32)
    pose_used = False
    if pose_estimator is not None and pose_estimator.enabled:
        pose_conf, keypoints, visible = pose_estimator.estimate(frames_bgr)
        pose_energy = center_pair_to_candidates(pose_pair_energy(keypoints, visible, pose_conf), n)
        pose_change = local_change_peak(pose_energy)
        pose_energy_norm = robust_norm(pose_energy)
        pose_change_norm = robust_norm(pose_change)
        pose_used = bool(np.any(pose_conf > 0.05))

    pose_conf_norm = robust_norm(pose_conf) if pose_used else np.zeros(n, dtype=np.float32)

    if pose_used:
        quality = 0.65 * sharp_norm + 0.35 * pose_conf_norm
        motion_change = 0.42 * flow_change_norm + 0.28 * diff_change_norm + 0.30 * pose_change_norm
        motion_energy = (
            0.42 * flow_energy_norm + 0.28 * diff_energy_norm + 0.30 * pose_energy_norm
        )
        blur_motion_penalty = motion_energy * (1.0 - sharp_norm) * (1.0 - pose_conf_norm)
    else:
        quality = sharp_norm
        motion_change = 0.60 * flow_change_norm + 0.40 * diff_change_norm
        motion_energy = 0.60 * flow_energy_norm + 0.40 * diff_energy_norm
        blur_motion_penalty = motion_energy * (1.0 - sharp_norm)

    glide_penalty = 1.0 - motion_energy
    final_score = (
        0.52 * motion_change
        + 0.20 * motion_energy
        + 0.28 * quality
        - 0.22 * glide_penalty
        - 0.35 * blur_motion_penalty
    )
    frame_times = frame_indices.astype(np.float32) / float(fps)
    rel_pos = (frame_times - float(segment_start)) / max(float(segment_end - segment_start), 1e-6)
    interior_bias = np.clip(1.0 - np.abs(rel_pos - 0.5) / 0.5, 0.0, 1.0)
    final_score = final_score + 0.03 * interior_bias
    final_score = final_score.astype(np.float32)
    final_score[~eligible] = -np.inf

    best_local_idx = int(np.argmax(final_score))
    if not np.isfinite(final_score[best_local_idx]):
        best_local_idx = int(np.argmax(sharp_raw))

    rows: List[Dict[str, float]] = []
    for i in range(n):
        rows.append(
            {
                "segment_idx": segment_idx,
                "segment_start": segment_start,
                "segment_end": segment_end,
                "candidate_local_idx": i,
                "frame_index": int(frame_indices[i]),
                "time_sec": float(frame_indices[i] / fps),
                "selected": int(i == best_local_idx),
                "eligible_after_blur_filter": int(eligible[i]),
                "final_score": float(final_score[i]) if np.isfinite(final_score[i]) else float("-inf"),
                "sharpness_laplacian_var": float(sharp_raw[i]),
                "blur_threshold": blur_threshold,
                "sharpness_norm": float(sharp_norm[i]),
                "frame_diff_energy_norm": float(diff_energy_norm[i]),
                "flow_energy_norm": float(flow_energy_norm[i]),
                "motion_energy_norm": float(motion_energy_norm[i]),
                "frame_diff_change_norm": float(diff_change_norm[i]),
                "flow_change_norm": float(flow_change_norm[i]),
                "pose_conf_norm": float(pose_conf_norm[i]),
                "pose_energy_norm": float(pose_energy_norm[i]),
                "pose_change_norm": float(pose_change_norm[i]),
                "quality_score": float(quality[i]),
                "motion_change_score": float(motion_change[i]),
                "interior_bias": float(interior_bias[i]),
                "glide_penalty": float(glide_penalty[i]),
                "blur_motion_penalty": float(blur_motion_penalty[i]),
                "lower_body_visible": 0.0,
                "selection_mode": "generic",
                "jump_event_id": "",
                "jump_phase": "",
                "event_takeoff_time": "",
                "event_landing_time": "",
            }
        )

    best_row = rows[best_local_idx]
    result = SegmentResult(
        segment_idx=segment_idx,
        segment_start=segment_start,
        segment_end=segment_end,
        best_local_idx=best_local_idx,
        best_frame_index=int(best_row["frame_index"]),
        best_time_sec=float(best_row["time_sec"]),
        best_score=float(best_row["final_score"]),
    )
    return result, rows


def annotate_frame(frame_bgr: np.ndarray, row: Dict[str, float]) -> np.ndarray:
    img = frame_bgr.copy()
    mode = str(row.get("selection_mode", "generic"))
    event_id = row.get("jump_event_id", "")
    phase = row.get("jump_phase", "")
    event_text = f"  mode={mode}"
    if event_id != "":
        event_text += f"  event={event_id}  phase={phase}"
    lines = [
        f"seg {int(row['segment_idx']):03d}  t={row['time_sec']:.2f}s  f={int(row['frame_index'])}",
        f"score={row['final_score']:.3f}  sharp={row['sharpness_norm']:.2f}  motion={row['motion_change_score']:.2f}",
        f"flow={row['flow_change_norm']:.2f}  diff={row['frame_diff_change_norm']:.2f}  pose={row['pose_conf_norm']:.2f}  lower={float(row.get('lower_body_visible', 0.0)):.2f}",
        event_text.strip(),
    ]
    x, y = 16, 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, min(img.shape[1], img.shape[0]) / 1100.0)
    thickness = max(1, int(round(font_scale * 2)))
    for i, text in enumerate(lines):
        yy = y + i * int(30 * font_scale + 10)
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(img, (x - 6, yy - th - 8), (x + tw + 6, yy + 8), (0, 0, 0), -1)
        cv2.putText(img, text, (x, yy), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


def write_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def selected_frame_index_rows(rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for row in rows:
        out.append(
            {
                "segment_idx": int(row["segment_idx"]),
                "segment_start": float(row["segment_start"]),
                "segment_end": float(row["segment_end"]),
                "frame_index": int(row["frame_index"]),
                "time_sec": float(row["time_sec"]),
                "selection_mode": row.get("selection_mode", "generic"),
                "jump_event_id": row.get("jump_event_id", ""),
                "jump_phase": row.get("jump_phase", ""),
                "selection_reason": row.get("selection_reason", ""),
            }
        )
    return out


def write_frame_index_txt(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{int(row['frame_index'])}\n")


def make_contact_sheet(image_paths: Sequence[Path], out_path: Path, columns: int = 5, thumb_width: int = 360) -> None:
    if not image_paths:
        return

    thumbs = []
    max_h = 0
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = thumb_width / float(w)
        thumb_h = max(1, int(round(h * scale)))
        thumb = cv2.resize(img, (thumb_width, thumb_h), interpolation=cv2.INTER_AREA)
        thumbs.append(thumb)
        max_h = max(max_h, thumb_h)

    if not thumbs:
        return

    rows = int(math.ceil(len(thumbs) / float(columns)))
    canvas = np.full((rows * max_h, columns * thumb_width, 3), 245, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        r = idx // columns
        c = idx % columns
        h, w = thumb.shape[:2]
        y0 = r * max_h
        x0 = c * thumb_width
        canvas[y0 : y0 + h, x0 : x0 + w] = thumb

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def select_keyframes(args: argparse.Namespace) -> None:
    video_path = str(Path(args.video))
    output_dir = Path(args.output_dir)
    clean_dir = output_dir / "selected_frames"
    annotated_dir = output_dir / "selected_frames_annotated"
    if not args.frames_only:
        clean_dir.mkdir(parents=True, exist_ok=True)
        annotated_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if total_frames > 0 else 0.0
    cap.release()

    if total_frames <= 0 or duration <= 0:
        raise RuntimeError(f"invalid video metadata: frames={total_frames}, fps={fps}")

    pose = PoseEstimator(
        backend=args.pose_backend,
        device=args.device,
        batch_size=args.pose_batch_size,
        max_side=args.pose_max_side,
        proxy=args.proxy,
    )

    num_segments = int(math.ceil(duration / args.clip_len))
    start_segment = max(0, args.start_segment)
    end_segment = num_segments
    if args.max_segments is not None:
        end_segment = min(end_segment, start_segment + args.max_segments)

    print(f"[video] {video_path}")
    print(f"[video] fps={fps:.3f}, frames={total_frames}, duration={duration:.2f}s")
    print(f"[run] segments={start_segment}..{end_segment - 1}, clip_len={args.clip_len}s")

    jump_events: List[JumpEvent] = []
    if not args.disable_jump_events:
        analysis_start = max(0.0, start_segment * args.clip_len - args.event_margin_sec)
        analysis_end = min(duration, end_segment * args.clip_len + args.event_margin_sec)
        print(
            "[jump] analyzing "
            f"{analysis_start:.2f}-{analysis_end:.2f}s at {args.event_fps:.2f} fps"
        )
        timeline = analyze_video_timeline(
            video_path=video_path,
            fps=fps,
            total_frames=total_frames,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
            event_fps=args.event_fps,
            pose_estimator=pose,
        )
        jump_events = detect_jump_events(
            timeline=timeline,
            fps=fps,
            threshold_quantile=args.jump_threshold_quantile,
            min_event_score=args.min_jump_event_score,
            min_air_sec=args.min_air_sec,
            max_air_sec=args.max_air_sec,
            entry_window_sec=args.entry_window_sec,
            landing_window_sec=args.landing_window_sec,
            min_event_gap_sec=args.min_jump_gap_sec,
        )
        write_csv(output_dir / "jump_timeline.csv", timeline_rows(timeline))
        write_csv(output_dir / "jump_events.csv", jump_event_rows(jump_events))
        print(f"[jump] detected events: {len(jump_events)}")
        for event in jump_events:
            print(
                "[jump] "
                f"event={event.event_id:02d} "
                f"entry={event.entry_time:.2f}s "
                f"takeoff={event.takeoff_time:.2f}s "
                f"air={event.air_start_time:.2f}-{event.air_end_time:.2f}s "
                f"landing={event.landing_time:.2f}s "
                f"score={event.score:.3f}"
            )

    all_rows: List[Dict[str, float]] = []
    selected_rows: List[Dict[str, float]] = []
    annotated_paths: List[Path] = []

    for segment_idx in range(start_segment, end_segment):
        seg_start = segment_idx * args.clip_len
        seg_end = min((segment_idx + 1) * args.clip_len, duration)
        indices = candidate_indices(fps, total_frames, seg_start, seg_end, args.sample_fps)
        frames = read_frames(video_path, indices)
        if len(frames) != len(indices):
            valid_count = min(len(frames), len(indices))
            frames = frames[:valid_count]
            indices = indices[:valid_count]
        if not frames:
            print(f"[warn] segment {segment_idx:03d}: no readable frames")
            continue

        result, rows = score_segment(
            frames_bgr=frames,
            frame_indices=indices,
            fps=fps,
            segment_idx=segment_idx,
            segment_start=seg_start,
            segment_end=seg_end,
            blur_quantile=args.blur_quantile,
            pose_estimator=pose if args.generic_use_pose else None,
        )
        best_frame = frames[result.best_local_idx]
        best_row = rows[result.best_local_idx]

        segment_events = events_for_segment(jump_events, seg_start, seg_end)
        jump_selection = None
        if segment_events:
            jump_selection = select_jump_phase_frame(
                video_path=video_path,
                fps=fps,
                total_frames=total_frames,
                segment_idx=segment_idx,
                segment_start=seg_start,
                segment_end=seg_end,
                events=segment_events,
                blur_quantile=args.blur_quantile,
                pose_estimator=pose,
                segment_blur_threshold=float(rows[0]["blur_threshold"]) if rows else 0.0,
                landing_min_sharpness_ratio=args.landing_min_sharpness_ratio,
            )

        if jump_selection is not None:
            for row in rows:
                row["selected"] = 0
            best_frame, best_row, jump_rows = jump_selection
            all_rows.extend(rows)
            all_rows.extend(jump_rows)
        else:
            all_rows.extend(rows)

        selected_rows.append(best_row)

        stem = (
            f"seg_{segment_idx:03d}_"
            f"t{seg_start:07.2f}-{seg_end:07.2f}_"
            f"f{int(best_row['frame_index']):06d}_score{float(best_row['final_score']):+.3f}_"
            f"{best_row.get('selection_mode', 'generic')}"
        )
        clean_path = clean_dir / f"{stem}.jpg"
        annotated_path = annotated_dir / f"{stem}.jpg"
        if not args.frames_only:
            cv2.imwrite(str(clean_path), best_frame)
            cv2.imwrite(str(annotated_path), annotate_frame(best_frame, best_row))
            annotated_paths.append(annotated_path)

        print(
            "[segment] "
            f"{segment_idx:03d} {seg_start:7.2f}-{seg_end:7.2f}s "
            f"best={float(best_row['time_sec']):7.2f}s "
            f"score={float(best_row['final_score']):+.3f} "
            f"sharp={best_row['sharpness_norm']:.2f} "
            f"motion={best_row['motion_change_score']:.2f} "
            f"pose={best_row['pose_conf_norm']:.2f} "
            f"mode={best_row.get('selection_mode', 'generic')}"
        )

    write_csv(output_dir / "candidate_scores.csv", all_rows)
    write_csv(output_dir / "selected_keyframes.csv", selected_rows)
    frame_rows = selected_frame_index_rows(selected_rows)
    write_csv(output_dir / "selected_frame_indices.csv", frame_rows)
    write_frame_index_txt(output_dir / "selected_frame_indices.txt", frame_rows)
    if not args.frames_only and not args.no_contact_sheet:
        make_contact_sheet(annotated_paths, output_dir / "contact_sheet.jpg", columns=args.sheet_columns)

    if not args.frames_only:
        print(f"[done] selected frames: {clean_dir}")
        print(f"[done] annotated frames: {annotated_dir}")
    print(f"[done] selected csv: {output_dir / 'selected_keyframes.csv'}")
    print(f"[done] selected frame indices csv: {output_dir / 'selected_frame_indices.csv'}")
    print(f"[done] selected frame indices txt: {output_dir / 'selected_frame_indices.txt'}")
    if not args.disable_jump_events:
        print(f"[done] jump events csv: {output_dir / 'jump_events.csv'}")
    if not args.frames_only and not args.no_contact_sheet:
        print(f"[done] contact sheet: {output_dir / 'contact_sheet.jpg'}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select one keyframe per video segment.")
    parser.add_argument("--video", required=True, help="Path to an input video.")
    parser.add_argument("--output-dir", required=True, help="Directory for selected frames and CSV files.")
    parser.add_argument("--clip-len", type=float, default=5.0, help="Segment length in seconds.")
    parser.add_argument("--sample-fps", type=float, default=4.0, help="Candidate frame sampling rate.")
    parser.add_argument(
        "--blur-quantile",
        type=float,
        default=0.25,
        help="Discard frames below this per-segment Laplacian variance quantile.",
    )
    parser.add_argument(
        "--pose-backend",
        choices=["auto", "torchvision", "none"],
        default="auto",
        help="Optional pose backend for pose-speed scoring.",
    )
    parser.add_argument("--device", default="cuda", help="Pose device, for example cuda or cpu.")
    parser.add_argument("--pose-batch-size", type=int, default=4)
    parser.add_argument("--pose-max-side", type=int, default=640)
    parser.add_argument(
        "--proxy",
        default="",
        help="Optional HTTP/HTTPS proxy for downloading pose weights, e.g. http://127.0.0.1:6518.",
    )
    parser.add_argument(
        "--disable-jump-events",
        action="store_true",
        help="Disable two-stage jump-event detection and use only the generic scorer.",
    )
    parser.add_argument(
        "--generic-use-pose",
        action="store_true",
        help="Also use the pose model inside the fallback generic scorer. Slower; off by default.",
    )
    parser.add_argument("--event-fps", type=float, default=3.0, help="Coarse sampling FPS for jump-event detection.")
    parser.add_argument(
        "--event-margin-sec",
        type=float,
        default=1.0,
        help="Extra seconds around the requested segment range for jump-event analysis.",
    )
    parser.add_argument(
        "--jump-threshold-quantile",
        type=float,
        default=0.93,
        help="Quantile threshold for jump takeoff/landing peak proposals.",
    )
    parser.add_argument(
        "--min-jump-event-score",
        type=float,
        default=0.55,
        help="Minimum fused event score required to accept a jump proposal.",
    )
    parser.add_argument("--min-air-sec", type=float, default=0.20)
    parser.add_argument("--max-air-sec", type=float, default=1.25)
    parser.add_argument("--entry-window-sec", type=float, default=0.80)
    parser.add_argument("--landing-window-sec", type=float, default=0.50)
    parser.add_argument(
        "--landing-min-sharpness-ratio",
        type=float,
        default=0.75,
        help=(
            "When a segment contains both takeoff and landing, keep landing unless "
            "its Laplacian variance is below this ratio times the segment blur threshold."
        ),
    )
    parser.add_argument("--min-jump-gap-sec", type=float, default=4.00)
    parser.add_argument("--start-segment", type=int, default=0)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--frames-only", action="store_true", help="Only write selected frame-number files and CSV logs.")
    parser.add_argument("--no-contact-sheet", action="store_true")
    parser.add_argument("--sheet-columns", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not 0.0 <= args.blur_quantile <= 1.0:
        raise ValueError("--blur-quantile must be in [0, 1]")
    if not 0.0 <= args.jump_threshold_quantile <= 1.0:
        raise ValueError("--jump-threshold-quantile must be in [0, 1]")
    select_keyframes(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
