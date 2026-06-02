"""
Deep Re-ID Healer — Neural Person Re-Identification Track Stitcher
===================================================================
Uses a headless ResNet50 backbone to extract deep visual embeddings
from player crops, builds baseline identity profiles from early frames,
and auto-stitches broken fragment tracklets back to their master IDs
using cosine similarity gating.

Usage
-----
    python deep_reid_healer.py
    python deep_reid_healer.py --video input_videos/input.mp4 --pkl tracker_stubs/player_detections.pkl
    python deep_reid_healer.py --threshold 0.90 --drop 35 8 56
    python deep_reid_healer.py --profile-frames 200 --profile-crops 50
    python deep_reid_healer.py --drop-noise                        # auto-strip all non-primary IDs
    python deep_reid_healer.py --drop-noise --keep 1 2 3 4 5       # customise which IDs survive
"""

import argparse
import os
import pickle
import random
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torchvision.models import ResNet50_Weights
from tqdm import tqdm


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
COSINE_THRESHOLD       = 0.88   # minimum cosine similarity to confirm identity
PROFILE_FRAME_WINDOW   = 150    # scan first N frames for baseline profiles
PROFILE_CROP_COUNT     = 30     # number of crops to average for ground-truth profile
FRAGMENT_SAMPLE_COUNT  = 5      # crops sampled from each fragment tracklet
FRAGMENT_MIN_FRAME     = 300    # track IDs first appearing after this frame are fragments
EMBEDDING_DIM          = 2048   # ResNet50 avgpool output dimension


# ──────────────────────────────────────────────
# Device Selection
# ──────────────────────────────────────────────
def get_device():
    """Select the best available compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[DEVICE] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[DEVICE] Using Apple Metal (MPS) GPU")
    else:
        device = torch.device("cpu")
        print("[DEVICE] Using CPU fallback")
    return device


# ──────────────────────────────────────────────
# Feature Extractor (headless ResNet50)
# ──────────────────────────────────────────────
class DeepReIDExtractor:
    """Wraps a headless ResNet50 for deep feature extraction."""

    def __init__(self, device):
        self.device = device

        # Standard ImageNet preprocessing pipeline
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Load pretrained ResNet50 and strip the classification head
        print("[MODEL] Loading ResNet50 backbone (ImageNet weights)...")
        base = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Remove the final FC layer — keep everything up to avgpool
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()
        print(f"[MODEL] Backbone ready on {self.device}  (output dim: {EMBEDDING_DIM})")

    @torch.no_grad()
    def extract(self, bgr_crop):
        """
        Extract a 2048-D embedding vector from a BGR OpenCV crop.

        Parameters
        ----------
        bgr_crop : np.ndarray  (H, W, 3) BGR uint8 image

        Returns
        -------
        np.ndarray  (2048,) L2-normalised embedding vector, or None on failure.
        """
        if bgr_crop is None or bgr_crop.size == 0:
            return None

        # BGR → RGB for torchvision
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)

        feat = self.backbone(tensor)          # (1, 2048, 1, 1)
        feat = feat.flatten(start_dim=1)      # (1, 2048)

        # L2 normalise for cosine similarity
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        return feat.cpu().numpy().flatten()    # (2048,)


# ──────────────────────────────────────────────
# Crop Utility
# ──────────────────────────────────────────────
def safe_crop(frame, bbox):
    """Extract a clamped bounding-box region from a frame."""
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


# ──────────────────────────────────────────────
# Tracklet Segmentation
# ──────────────────────────────────────────────
def segment_tracks(player_detections):
    """
    Build per-track metadata from the frame list.

    Returns
    -------
    dict : {track_id: {'first_frame', 'last_frame', 'total_frames',
                       'frame_indices': [int], 'bboxes': {frame_idx: bbox}}}
    """
    segments = {}
    for frame_idx, frame_dict in enumerate(player_detections):
        for track_id, data in frame_dict.items():
            bbox = data['bbox'] if isinstance(data, dict) else data
            if track_id not in segments:
                segments[track_id] = {
                    'first_frame': frame_idx,
                    'last_frame': frame_idx,
                    'total_frames': 1,
                    'frame_indices': [frame_idx],
                    'bboxes': {frame_idx: bbox},
                }
            else:
                seg = segments[track_id]
                seg['last_frame'] = frame_idx
                seg['total_frames'] += 1
                seg['frame_indices'].append(frame_idx)
                seg['bboxes'][frame_idx] = bbox
    return segments


# ──────────────────────────────────────────────
# Baseline Profile Builder
# ──────────────────────────────────────────────
def build_baseline_profiles(
    extractor, video_path, player_detections, segments,
    profile_frame_window, profile_crop_count, fragment_min_frame,
):
    """
    Build a rolling-mean embedding for each master track ID by scanning
    the first `profile_frame_window` frames.

    Master tracks = those whose first appearance is within the early window
    AND which are NOT short-lived noise (>= 30 frames total).

    Returns
    -------
    dict : {track_id: np.ndarray (2048,)}  — L2-normalised mean profile
    """
    # Identify master track IDs
    master_ids = [
        tid for tid, seg in segments.items()
        if seg['first_frame'] < fragment_min_frame and seg['total_frames'] >= profile_crop_count
    ]
    master_ids.sort()
    print(f"\n[PROFILE] Master Track IDs: {master_ids}")
    print(f"[PROFILE] Scanning first {profile_frame_window} frames for baseline crops...")

    # Collect frame indices needed for each master (capped to profile window)
    crop_plan = defaultdict(list)  # {track_id: [frame_idx, ...]}
    for tid in master_ids:
        frames_in_window = [
            fi for fi in segments[tid]['frame_indices']
            if fi < profile_frame_window
        ]
        # Take up to profile_crop_count evenly spaced frames
        if len(frames_in_window) > profile_crop_count:
            step = len(frames_in_window) / profile_crop_count
            frames_in_window = [frames_in_window[int(i * step)] for i in range(profile_crop_count)]
        crop_plan[tid] = frames_in_window

    # All frames we need to read
    frames_needed = set()
    for flist in crop_plan.values():
        frames_needed.update(flist)

    if not frames_needed:
        print("[WARN] No frames to scan for profiles. Aborting.")
        return {}

    max_frame = max(frames_needed)

    # Build bbox lookup: {frame_idx: {track_id: bbox}}
    bbox_lookup = defaultdict(dict)
    for tid in master_ids:
        for fi in crop_plan[tid]:
            bbox_lookup[fi][tid] = segments[tid]['bboxes'][fi]

    # Single video pass to collect crops
    embeddings = defaultdict(list)  # {track_id: [np.ndarray, ...]}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return {}

    for fi in tqdm(range(max_frame + 1), desc="Scanning baseline frames", unit="fr"):
        ret, frame = cap.read()
        if not ret:
            break
        if fi not in bbox_lookup:
            continue
        for tid, bbox in bbox_lookup[fi].items():
            crop = safe_crop(frame, bbox)
            emb = extractor.extract(crop)
            if emb is not None:
                embeddings[tid].append(emb)
    cap.release()

    # Average embeddings into ground-truth profiles
    profiles = {}
    for tid in master_ids:
        embs = embeddings.get(tid, [])
        if not embs:
            print(f"  [WARN] Track {tid}: 0 valid crops — skipping profile")
            continue
        mean_emb = np.mean(embs, axis=0)
        # Re-normalise the mean vector
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb /= norm
        profiles[tid] = mean_emb
        print(f"  Track {tid}: profile built from {len(embs)} crops")

    return profiles


# ──────────────────────────────────────────────
# Fragment Matching
# ──────────────────────────────────────────────
def match_fragments(
    extractor, video_path, segments, profiles,
    fragment_min_frame, fragment_sample_count, cosine_threshold,
):
    """
    Identify fragment tracklets and match them against baseline profiles.

    Returns
    -------
    dict : {fragment_id: master_id}  — confirmed re-ID merges
    """
    master_ids = set(profiles.keys())

    # Fragment = track first seen after fragment_min_frame, not already a master
    fragment_ids = [
        tid for tid, seg in segments.items()
        if seg['first_frame'] >= fragment_min_frame and tid not in master_ids
    ]
    fragment_ids.sort()

    if not fragment_ids:
        print("\n[MATCH] No fragment tracklets found after frame {fragment_min_frame}.")
        return {}

    print(f"\n[MATCH] Fragment Track IDs to evaluate: {fragment_ids}")

    # Build crop sample plan for fragments
    crop_plan = {}  # {fragment_id: [frame_idx, ...]}
    for tid in fragment_ids:
        avail = segments[tid]['frame_indices']
        if len(avail) <= fragment_sample_count:
            crop_plan[tid] = avail
        else:
            crop_plan[tid] = sorted(random.sample(avail, fragment_sample_count))

    # All frames needed
    frames_needed = set()
    for flist in crop_plan.values():
        frames_needed.update(flist)

    max_frame = max(frames_needed) if frames_needed else 0

    # bbox lookup
    bbox_lookup = defaultdict(dict)
    for tid in fragment_ids:
        for fi in crop_plan[tid]:
            bbox_lookup[fi][tid] = segments[tid]['bboxes'][fi]

    # Single video pass
    embeddings = defaultdict(list)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return {}

    for fi in tqdm(range(max_frame + 1), desc="Scanning fragment crops", unit="fr"):
        ret, frame = cap.read()
        if not ret:
            break
        if fi not in bbox_lookup:
            continue
        for tid, bbox in bbox_lookup[fi].items():
            crop = safe_crop(frame, bbox)
            emb = extractor.extract(crop)
            if emb is not None:
                embeddings[tid].append(emb)
    cap.release()

    # Score each fragment against every master profile
    remap = {}
    print(f"\n{'Fragment':>10} | {'Master':>8} | {'Cosine':>8} | Decision")
    print("-" * 50)

    for fid in fragment_ids:
        embs = embeddings.get(fid, [])
        if not embs:
            print(f"  {fid:>8} | {'---':>8} | {'N/A':>8} | NO CROPS")
            continue

        # Mean embedding for the fragment
        frag_emb = np.mean(embs, axis=0)
        norm = np.linalg.norm(frag_emb)
        if norm > 0:
            frag_emb /= norm

        best_master = None
        best_score = -1.0

        for mid, profile in profiles.items():
            # Cosine similarity (both are L2-normalised → dot product)
            score = float(np.dot(frag_emb, profile))
            if score > best_score:
                best_score = score
                best_master = mid

        if best_score >= cosine_threshold:
            remap[fid] = best_master
            decision = f"✅ MERGE → {best_master}"
        else:
            decision = f"❌ SKIP (best={best_master})"

        print(f"  {fid:>8} | {best_master:>8} | {best_score:>8.4f} | {decision}")

    return remap


# ──────────────────────────────────────────────
# Global Remapping & Track Dropping
# ──────────────────────────────────────────────
def apply_remap_and_drop(player_detections, remap, drop_ids):
    """
    Walk every frame and:
      1. Replace fragment IDs with their master IDs per `remap`.
      2. Completely remove any track IDs in `drop_ids`.

    Returns a NEW list (original is not mutated).
    """
    drop_set = set(drop_ids)
    healed = []

    for frame_dict in player_detections:
        new_frame = {}
        for track_id, data in frame_dict.items():
            # Drop unwanted tracks
            if track_id in drop_set:
                continue

            # Remap fragment → master
            resolved_id = remap.get(track_id, track_id)

            # Collision guard: keep the first (master) entry if both exist
            if resolved_id in new_frame:
                continue

            new_frame[resolved_id] = data
        healed.append(new_frame)

    return healed


# ──────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────
def print_track_summary(label, player_detections):
    """Print a concise table of all track IDs and their lifespans."""
    segments = segment_tracks(player_detections)
    ids = sorted(segments.keys())
    print(f"\n[{label}] {len(ids)} unique Track IDs: {ids}")
    for tid in ids:
        seg = segments[tid]
        span = seg['last_frame'] - seg['first_frame'] + 1
        cov = seg['total_frames'] / span * 100 if span > 0 else 0
        print(f"    ID {tid:>3}: frames {seg['first_frame']:>5}–{seg['last_frame']:>5} "
              f"({seg['total_frames']:>4}/{span:>4} = {cov:5.1f}%)")


# ──────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────
def run_healer(video_path, pkl_path, cosine_threshold, profile_frame_window,
               profile_crop_count, fragment_min_frame, fragment_sample_count,
               drop_ids, drop_noise=False, keep_ids=None):
    """Execute the full deep Re-ID healing pipeline."""

    # ── 1. Load detections ──
    print(f"\n{'='*60}")
    print("  Deep Re-ID Healer — Neural Track Stitcher")
    print(f"{'='*60}")
    print(f"\n[LOAD] Detections: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        player_detections = pickle.load(f)
    print(f"[LOAD] Total frames: {len(player_detections)}")

    # ── 2. Segment tracks ──
    segments = segment_tracks(player_detections)
    print_track_summary("PRE-HEAL", player_detections)

    # ── 3. Init feature extractor ──
    device = get_device()
    extractor = DeepReIDExtractor(device)

    # ── 4. Build baseline profiles ──
    profiles = build_baseline_profiles(
        extractor, video_path, player_detections, segments,
        profile_frame_window, profile_crop_count, fragment_min_frame,
    )

    if not profiles:
        print("[ABORT] No baseline profiles could be built. Exiting.")
        return

    # ── 5. Match fragments ──
    remap = match_fragments(
        extractor, video_path, segments, profiles,
        fragment_min_frame, fragment_sample_count, cosine_threshold,
    )

    # ── 6. Auto-detect noise IDs if --drop-noise is active ──
    if drop_noise:
        primary_ids = set(keep_ids) if keep_ids else {1, 2, 3, 4}
        all_ids = set(segment_tracks(player_detections).keys())
        # IDs that were remapped INTO a primary survive; only their old fragment key disappears
        surviving_via_remap = set(remap.values())
        noise_ids = [
            tid for tid in sorted(all_ids)
            if tid not in primary_ids and tid not in surviving_via_remap
        ]
        # Merge with any explicit --drop list (deduplicate)
        combined = set(drop_ids) | set(noise_ids)
        drop_ids = sorted(combined)
        print(f"\n[DROP-NOISE] Primary keep set: {sorted(primary_ids)}")
        print(f"[DROP-NOISE] Auto-detected noise IDs to strip: {noise_ids}")

    # ── 7. Apply remap & drops ──
    total_changes = len(remap) + len(drop_ids)
    if total_changes == 0:
        print("\n[INFO] No merges or drops to apply. Data is clean.")
        return

    if remap:
        print(f"\n[REMAP] Applying {len(remap)} merge(s): {remap}")
    if drop_ids:
        print(f"[DROP]  Stripping {len(drop_ids)} track ID(s): {drop_ids}")

    healed = apply_remap_and_drop(player_detections, remap, drop_ids)

    # ── 8. Save ──
    print(f"\n[SAVE] Overwriting: {pkl_path}")
    with open(pkl_path, 'wb') as f:
        pickle.dump(healed, f)

    print_track_summary("POST-HEAL", healed)
    print(f"  Frame array length preserved: {len(healed)} frames (no index shift)")
    print(f"\n{'='*60}")
    print("  ✅  Deep Re-ID healing complete.")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Deep Re-ID Healer: Neural person re-identification for track stitching."
    )
    parser.add_argument(
        "--video", default="input_videos/input.mp4",
        help="Path to the source video for crop extraction.",
    )
    parser.add_argument(
        "--pkl", default="tracker_stubs/player_detections.pkl",
        help="Path to the player detections pickle stub.",
    )
    parser.add_argument(
        "--threshold", type=float, default=COSINE_THRESHOLD,
        help=f"Cosine similarity threshold for identity confirmation (default {COSINE_THRESHOLD}).",
    )
    parser.add_argument(
        "--profile-frames", type=int, default=PROFILE_FRAME_WINDOW,
        help=f"Number of early frames to scan for baseline profiles (default {PROFILE_FRAME_WINDOW}).",
    )
    parser.add_argument(
        "--profile-crops", type=int, default=PROFILE_CROP_COUNT,
        help=f"Number of crops averaged per master profile (default {PROFILE_CROP_COUNT}).",
    )
    parser.add_argument(
        "--fragment-after", type=int, default=FRAGMENT_MIN_FRAME,
        help=f"Treat track IDs first appearing after this frame as fragments (default {FRAGMENT_MIN_FRAME}).",
    )
    parser.add_argument(
        "--fragment-samples", type=int, default=FRAGMENT_SAMPLE_COUNT,
        help=f"Crops sampled per fragment tracklet (default {FRAGMENT_SAMPLE_COUNT}).",
    )
    parser.add_argument(
        "--drop", type=int, nargs="*", default=[],
        help="Track IDs to completely strip from the dataset (e.g. --drop 35 8 56).",
    )
    parser.add_argument(
        "--drop-noise", action="store_true", default=False,
        help="Auto-strip all track IDs NOT in the --keep set (default keep: 1 2 3 4).",
    )
    parser.add_argument(
        "--keep", type=int, nargs="*", default=None,
        help="Primary player IDs to retain when --drop-noise is active (default: 1 2 3 4).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.pkl):
        print(f"[ERROR] Pickle file not found: {args.pkl}")
        sys.exit(1)
    if not os.path.exists(args.video):
        print(f"[ERROR] Video file not found: {args.video}")
        sys.exit(1)

    run_healer(
        video_path=args.video,
        pkl_path=args.pkl,
        cosine_threshold=args.threshold,
        profile_frame_window=args.profile_frames,
        profile_crop_count=args.profile_crops,
        fragment_min_frame=args.fragment_after,
        fragment_sample_count=args.fragment_samples,
        drop_ids=args.drop,
        drop_noise=args.drop_noise,
        keep_ids=args.keep,
    )


if __name__ == "__main__":
    main()
