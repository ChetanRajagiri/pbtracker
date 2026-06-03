"""
Tracklet Merger — Post-Processing ID Stitch Healer
===================================================
Heals broken / fragmented tracking IDs by correlating terminated tracklets
with newly initialized ones using three gating criteria:

  1. Temporal proximity  : new track starts within TEMPORAL_WINDOW frames of old track ending.
  2. Spatial proximity   : Euclidean distance between ending and starting bounding-box centroids < SPATIAL_THRESHOLD px.
  3. Visual similarity   : HSV color histogram correlation (cv2.HISTCMP_CORREL) > HIST_THRESHOLD.

Usage
-----
    python tracklet_merger.py                         # defaults
    python tracklet_merger.py --video input_videos/input.mp4 --pkl tracker_stubs/player_detections.pkl
    python tracklet_merger.py --temporal 60 --spatial 200 --hist 0.80   # override thresholds
"""

import argparse
import os
import pickle
import sys

import cv2
import numpy as np


# ──────────────────────────────────────────────
# Default thresholds
# ──────────────────────────────────────────────
TEMPORAL_WINDOW   = 45     # frames
SPATIAL_THRESHOLD = 150.0  # pixels (Euclidean centroid distance)
HIST_THRESHOLD    = 0.85   # cv2.HISTCMP_CORREL score


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def bbox_centroid(bbox):
    """Return (cx, cy) from [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def euclidean(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def crop_player(frame, bbox):
    """Extract the bounding-box region from a frame (clamped to frame edges)."""
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def hsv_histogram(crop):
    """Compute a normalised HSV histogram for *crop* (BGR image)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Use 50 bins for H (0–180), 60 bins for S (0–256)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


# ──────────────────────────────────────────────
# Tracklet analysis
# ──────────────────────────────────────────────
def build_tracklet_segments(player_detections):
    """
    Parse the per-frame detection list and return a dict mapping each
    track ID to its lifespan metadata:

        {track_id: {
            'first_frame': int,
            'last_frame':  int,
            'first_bbox':  [x1, y1, x2, y2],
            'last_bbox':   [x1, y1, x2, y2],
            'total_frames': int
        }}
    """
    segments = {}
    for frame_idx, frame_dict in enumerate(player_detections):
        for track_id, data in frame_dict.items():
            bbox = data['bbox'] if isinstance(data, dict) else data
            if track_id not in segments:
                segments[track_id] = {
                    'first_frame': frame_idx,
                    'last_frame': frame_idx,
                    'first_bbox': bbox,
                    'last_bbox': bbox,
                    'total_frames': 1,
                }
            else:
                segments[track_id]['last_frame'] = frame_idx
                segments[track_id]['last_bbox'] = bbox
                segments[track_id]['total_frames'] += 1
    return segments


def find_merge_candidates(segments, temporal_window, spatial_threshold):
    """
    Return a list of (old_id, new_id) candidate pairs that pass the
    temporal and spatial proximity gates.
    """
    candidates = []
    ids = list(segments.keys())
    for old_id in ids:
        old_seg = segments[old_id]
        for new_id in ids:
            if new_id == old_id:
                continue
            new_seg = segments[new_id]

            # New track must start *after* old track ends
            frame_gap = new_seg['first_frame'] - old_seg['last_frame']
            if frame_gap < 0 or frame_gap > temporal_window:
                continue

            # Spatial gate: centroid distance between old end and new start
            dist = euclidean(
                bbox_centroid(old_seg['last_bbox']),
                bbox_centroid(new_seg['first_bbox']),
            )
            if dist > spatial_threshold:
                continue

            candidates.append((old_id, new_id, frame_gap, dist))

    # Sort by smallest gap first, then smallest distance
    candidates.sort(key=lambda c: (c[2], c[3]))
    return candidates


# ──────────────────────────────────────────────
# Visual correlation gate
# ──────────────────────────────────────────────
def score_visual_similarity(video_path, segments, old_id, new_id):
    """
    Open the video, extract the crops for old_id's last frame and
    new_id's first frame, compute HSV histogram correlation.
    Returns the correlation score (float in [-1, 1]).
    """
    old_frame_idx = segments[old_id]['last_frame']
    new_frame_idx = segments[new_id]['first_frame']
    old_bbox = segments[old_id]['last_bbox']
    new_bbox = segments[new_id]['first_bbox']

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return -1.0

    frames_needed = {old_frame_idx: old_bbox, new_frame_idx: new_bbox}
    crops = {}
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx in frames_needed:
            crop = crop_player(frame, frames_needed[idx])
            if crop is not None and crop.size > 0:
                crops[idx] = crop
            if len(crops) == len(frames_needed):
                break
        idx += 1
    cap.release()

    if old_frame_idx not in crops or new_frame_idx not in crops:
        print(f"  [WARN] Could not extract crops for IDs {old_id}->{new_id}")
        return -1.0

    hist_old = hsv_histogram(crops[old_frame_idx])
    hist_new = hsv_histogram(crops[new_frame_idx])
    score = cv2.compareHist(hist_old, hist_new, cv2.HISTCMP_CORREL)
    return score


# ──────────────────────────────────────────────
# Remapping
# ──────────────────────────────────────────────
def apply_remap(player_detections, remap):
    """
    Walk through every frame and replace new_id keys with old_id keys
    according to the remap dict {new_id: old_id}.
    Returns a NEW list (original is not mutated).
    """
    remapped = []
    for frame_dict in player_detections:
        new_frame = {}
        for track_id, data in frame_dict.items():
            resolved_id = remap.get(track_id, track_id)
            # If a collision happens (both old and new id exist in same frame),
            # keep the one with the canonical (old) id already present.
            if resolved_id in new_frame:
                continue
            new_frame[resolved_id] = data
        remapped.append(new_frame)
    return remapped


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────
def run_merger(video_path, pkl_path, temporal_window, spatial_threshold, hist_threshold):
    # 1. Load detections
    print(f"Loading detections from: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        player_detections = pickle.load(f)
    print(f"  Total frames: {len(player_detections)}")

    # 2. Build tracklet segments
    segments = build_tracklet_segments(player_detections)
    print(f"  Unique Track IDs found: {sorted(segments.keys())}")
    for tid, seg in sorted(segments.items()):
        span = seg['last_frame'] - seg['first_frame'] + 1
        print(f"    ID {tid}: frames {seg['first_frame']}–{seg['last_frame']} "
              f"(present {seg['total_frames']}/{span} frames)")

    # 3. Find spatio-temporal candidates
    candidates = find_merge_candidates(segments, temporal_window, spatial_threshold)
    if not candidates:
        print("\n[INFO] No merge candidates found. Tracking IDs look clean.")
        return

    print(f"\n  Spatio-temporal candidates ({len(candidates)}):")
    for old_id, new_id, gap, dist in candidates:
        print(f"    {old_id} -> {new_id}  gap={gap} frames  dist={dist:.1f}px")

    # 4. Score each candidate visually and decide merges
    remap = {}       # {new_id: old_id}
    consumed = set() # IDs already remapped (prevent chains)

    for old_id, new_id, gap, dist in candidates:
        if old_id in consumed or new_id in consumed:
            continue

        score = score_visual_similarity(video_path, segments, old_id, new_id)
        status = "MERGE" if score >= hist_threshold else "SKIP"
        print(f"    Visual score {old_id}->{new_id}: {score:.4f}  [{status}]")

        if score >= hist_threshold:
            remap[new_id] = old_id
            consumed.add(new_id)

    if not remap:
        print("\n[INFO] No merges passed the visual similarity gate.")
        return

    # 5. Apply remap
    print(f"\n  Applying {len(remap)} merge(s): {remap}")
    merged_detections = apply_remap(player_detections, remap)

    # 6. Save back (overwrite original stub)
    print(f"  Overwriting stub: {pkl_path}")
    with open(pkl_path, 'wb') as f:
        pickle.dump(merged_detections, f)

    # 7. Summary
    new_segments = build_tracklet_segments(merged_detections)
    print(f"\n  Post-merge Track IDs: {sorted(new_segments.keys())}")
    for tid, seg in sorted(new_segments.items()):
        span = seg['last_frame'] - seg['first_frame'] + 1
        print(f"    ID {tid}: frames {seg['first_frame']}–{seg['last_frame']} "
              f"(present {seg['total_frames']}/{span} frames)")
    print("\n[DONE] Tracklet merge complete.")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Heal fragmented tracking IDs via spatio-temporal + visual correlation."
    )
    parser.add_argument(
        "--video", default="input_videos/input.mp4",
        help="Path to the source video (for crop extraction)."
    )
    parser.add_argument(
        "--pkl", default="tracker_stubs/player_detections.pkl",
        help="Path to the player detections pickle stub."
    )
    parser.add_argument(
        "--temporal", type=int, default=TEMPORAL_WINDOW,
        help=f"Max frame gap to consider a merge (default {TEMPORAL_WINDOW})."
    )
    parser.add_argument(
        "--spatial", type=float, default=SPATIAL_THRESHOLD,
        help=f"Max centroid distance in px (default {SPATIAL_THRESHOLD})."
    )
    parser.add_argument(
        "--hist", type=float, default=HIST_THRESHOLD,
        help=f"Min HSV histogram correlation to accept merge (default {HIST_THRESHOLD})."
    )
    args = parser.parse_args()

    if not os.path.exists(args.pkl):
        print(f"[ERROR] Pickle file not found: {args.pkl}")
        sys.exit(1)
    if not os.path.exists(args.video):
        print(f"[ERROR] Video file not found: {args.video}")
        sys.exit(1)

    run_merger(args.video, args.pkl, args.temporal, args.spatial, args.hist)


if __name__ == "__main__":
    main()
