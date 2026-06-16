#!/usr/bin/env python3
import os
import sys
import pickle
import argparse
import cv2
import numpy as np

# Re-use expanded polygon helper function logic from auto_player_filter
def expand_polygon_asymmetric(polygon, frame_height=1080):
    if polygon is None:
        return None
    pts = polygon.reshape(-1, 2).astype(np.float32)
    centroid = np.mean(pts, axis=0)
    expanded_pts = []
    for pt in pts:
        y_val = pt[1]
        # Classify vertex: near baseline gets 80px, far baseline gets 30px, others get 40px
        if y_val > frame_height * 0.55:
            dist = 80.0
        else:
            dist = 30.0
            
        direction = pt - centroid
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            unit_direction = direction / norm
            new_pt = pt + dist * unit_direction
        else:
            new_pt = pt
        expanded_pts.append(new_pt)
    return np.array(expanded_pts, dtype=np.int32)

def main():
    parser = argparse.ArgumentParser(description="Read-only diagnostic for missing Master ID player boxes")
    parser.add_argument("--frames", required=True, help="Comma-separated frame indices, e.g. 26,87,150")
    parser.add_argument("--master-id", type=int, required=True, help="Master ID to investigate, e.g. 4")
    parser.add_argument("--raw-id", type=int, default=None, help="Optional raw ID to trace embedding similarity and fallbacks")
    parser.add_argument("--lock-frame", type=int, default=None, help="Optional manual seeding lock frame overrides auto-calculation")
    args = parser.parse_args()

    # Parse frames
    try:
        frame_list = [int(f.strip()) for f in args.frames.split(",") if f.strip()]
    except ValueError:
        print("Error: --frames must be a comma-separated list of integers.")
        sys.exit(1)

    master_id = args.master_id
    raw_id = args.raw_id

    # File paths
    detections_pkl = "tracker_stubs/player_detections.pkl"
    raw_pkl = "tracker_stubs/player_detections_raw.pkl"
    court_pkl = "tracker_stubs/court_keypoints.pkl"

    # 1. Load player_detections.pkl (final post-filtration)
    if not os.path.exists(detections_pkl):
        print(f"Error: Final detections cache not found at: {detections_pkl}")
        sys.exit(1)

    with open(detections_pkl, 'rb') as f:
        final_detections = pickle.load(f)
    print(f"[LOAD] Loaded {len(final_detections)} frames of final detections.")

    # 2. Check for player_detections_raw.pkl
    raw_detections = None
    if os.path.exists(raw_pkl):
        with open(raw_pkl, 'rb') as f:
            raw_detections = pickle.load(f)
        print(f"[LOAD] Loaded {len(raw_detections)} frames of raw detections.")
    else:
        print(f"[INFO] Raw detections cache NOT found at {raw_pkl} (skipping raw comparisons).")

    # 3. Setup Court Polygon
    polygon = None
    frame_height = 1080
    if os.path.exists(court_pkl):
        with open(court_pkl, 'rb') as f:
            court_keypoints = pickle.load(f)
        if len(court_keypoints) >= 12:
            corner1 = court_keypoints[0]
            corner2 = court_keypoints[2]
            corner3 = court_keypoints[11]
            corner4 = court_keypoints[9]
            polygon = np.array([corner1, corner2, corner3, corner4], dtype=np.int32)
            polygon = expand_polygon_asymmetric(polygon, frame_height)

    # 4. Recompute Seeding Lock Frame and gallery template embeddings if raw_detections exists
    lock_frame = args.lock_frame
    locked_gallery = {}
    if raw_detections is not None:
        consecutive_frames = {}
        stable_tracks = set()
        gallery_features = {}
        gallery_bboxes = {}
        gallery_seeded = False
        
        for idx in range(len(raw_detections)):
            frame_dict = raw_detections[idx]
            active_tids = set(frame_dict.keys())
            for tid in active_tids:
                if tid <= 0:
                    continue
                data = frame_dict[tid]
                bbox = data.get('bbox')
                if bbox is None:
                    continue
                is_inside = True
                if polygon is not None:
                    pt1 = (bbox[0], bbox[3])
                    pt2 = ((bbox[0] + bbox[2]) / 2.0, bbox[3])
                    pt3 = (bbox[2], bbox[3])
                    is_inside = (
                        cv2.pointPolygonTest(polygon, pt1, False) >= 0 or
                        cv2.pointPolygonTest(polygon, pt2, False) >= 0 or
                        cv2.pointPolygonTest(polygon, pt3, False) >= 0
                    )
                if is_inside:
                    consecutive_frames[tid] = consecutive_frames.get(tid, 0) + 1
                    if 'embedding' in data and data['embedding'] is not None:
                        gallery_features.setdefault(tid, []).append(data['embedding'])
                        gallery_bboxes.setdefault(tid, []).append(bbox)
                else:
                    consecutive_frames[tid] = 0

            for tid in list(consecutive_frames.keys()):
                if tid not in active_tids:
                    consecutive_frames[tid] = 0

            for tid, count in consecutive_frames.items():
                if count >= 4:
                    stable_tracks.add(tid)

            should_lock = False
            if len(stable_tracks) >= 4:
                should_lock = True
            elif idx >= 150:
                should_lock = True
            elif idx == len(raw_detections) - 1:
                should_lock = True

            if should_lock and not gallery_seeded:
                selected_tids = sorted(list(stable_tracks), key=lambda t: len(gallery_features.get(t, [])), reverse=True)[:4]
                if len(selected_tids) > 0:
                    avg_coords = {}
                    for tid in selected_tids:
                        bboxes = gallery_bboxes[tid]
                        xs = [(b[0] + b[2]) / 2.0 for b in bboxes]
                        ys = [b[3] for b in bboxes]
                        avg_coords[tid] = (np.mean(xs), np.mean(ys))
                    sorted_by_y = sorted(selected_tids, key=lambda tid: avg_coords[tid][1])
                    half = len(sorted_by_y) // 2
                    far_pair = sorted_by_y[:half]
                    near_pair = sorted_by_y[half:]
                    far_sorted_x = sorted(far_pair, key=lambda tid: avg_coords[tid][0])
                    near_sorted_x = sorted(near_pair, key=lambda tid: avg_coords[tid][0])
                    id_mapping = {}
                    if len(near_sorted_x) >= 1: id_mapping[near_sorted_x[0]] = 1
                    if len(near_sorted_x) >= 2: id_mapping[near_sorted_x[1]] = 2
                    if len(far_sorted_x) >= 1: id_mapping[far_sorted_x[0]] = 3
                    if len(far_sorted_x) >= 2: id_mapping[far_sorted_x[1]] = 4
                    for raw_tid, m_id in id_mapping.items():
                        embs = gallery_features[raw_tid]
                        avg_emb = np.mean(embs, axis=0)
                        norm_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-6)
                        locked_gallery[m_id] = norm_emb
                    
                    if lock_frame is None:
                        lock_frame = idx
                    gallery_seeded = True

    if lock_frame is not None:
        print(f"[REID] Gallery seeding lock point detected/calculated at frame: {lock_frame}")
    else:
        print("[WARNING] Could not calculate seeding lock frame automatically (no raw detections). Specify via --lock-frame if needed.")

    # 5. Process each requested frame
    for f_idx in frame_list:
        print(f"\n==================================================")
        print(f"DIAGNOSTIC FOR FRAME {f_idx} (Master ID {master_id})")
        print(f"==================================================")

        # Seeding relation
        if lock_frame is not None:
            rel = "BEFORE" if f_idx < lock_frame else "AFTER"
            print(f"Timeline location: {rel} the gallery seeding lock point (lock frame = {lock_frame})")
        else:
            print("Timeline location: Unknown (seeding lock frame not available)")

        # final checks
        if f_idx >= len(final_detections):
            print(f"Result: Frame index {f_idx} is out of bounds for player_detections.pkl (length={len(final_detections)})")
            continue

        frame_final = final_detections[f_idx]
        has_master = master_id in frame_final

        print(f"\n[Final Detections check (player_detections.pkl)]")
        print(f"Master ID {master_id} present: {'YES' if has_master else 'NO'}")
        if has_master:
            data = frame_final[master_id]
            print(f"  - BBox: {data.get('bbox')}")
            print(f"  - is_on_court: {data.get('is_on_court')}")
            if 'similarity' in data:
                print(f"  - Similarity score: {data.get('similarity')}")
            else:
                # Also check fallback_events.csv if it has a failed similarity score logged for this frame/master ID
                failed_score = None
                if os.path.exists("tracker_stubs/fallback_events.csv"):
                    with open("tracker_stubs/fallback_events.csv", 'r') as csv_file:
                        for line in csv_file:
                            parts = line.strip().split(",")
                            if len(parts) >= 3 and parts[0] == str(f_idx) and parts[1] == str(master_id):
                                try:
                                    failed_score = float(parts[2])
                                except ValueError:
                                    pass
                if failed_score is not None:
                    print(f"  - Similarity score (from fallback csv): {failed_score} (Assigned via fallback)")
                else:
                    print("  - Similarity score: Not present in pickle data")

        # raw checks
        if raw_detections is not None:
            if f_idx >= len(raw_detections):
                print(f"Result: Frame index {f_idx} is out of bounds for player_detections_raw.pkl (length={len(raw_detections)})")
                continue

            frame_raw = raw_detections[f_idx]
            print(f"\n[Raw Detections check (player_detections_raw.pkl)]")
            print(f"Raw track IDs present in frame: {list(frame_raw.keys())}")
            for tid, data in frame_raw.items():
                print(f"  - Raw ID {tid}: BBox = {data.get('bbox')}, is_on_court = {data.get('is_on_court')}")

            # If master ID is missing and it's after the lock point, analyze raw tracks for spatial proximity / polygon tests
            if not has_master and lock_frame is not None and f_idx >= lock_frame:
                print(f"\n[Missing Master ID Analysis (Spatial & Polygon test for raw tracks)]")
                for tid, data in frame_raw.items():
                    bbox = data.get('bbox')
                    if bbox is None:
                        continue
                    
                    # Compute individual point Polygon tests
                    pt1 = (bbox[0], bbox[3])
                    pt2 = ((bbox[0] + bbox[2]) / 2.0, bbox[3])
                    pt3 = (bbox[2], bbox[3])
                    
                    val1 = cv2.pointPolygonTest(polygon, pt1, True) if polygon is not None else 0.0
                    val2 = cv2.pointPolygonTest(polygon, pt2, True) if polygon is not None else 0.0
                    val3 = cv2.pointPolygonTest(polygon, pt3, True) if polygon is not None else 0.0
                    
                    is_inside = val1 >= 0 or val2 >= 0 or val3 >= 0
                    
                    print(f"  - Raw ID {tid}:")
                    print(f"    - BBox: {bbox}")
                    print(f"    - Polygon Test (feet points distance):")
                    print(f"      * Left foot {pt1}: {val1:+.2f} px")
                    print(f"      * Middle foot {pt2}: {val2:+.2f} px")
                    print(f"      * Right foot {pt3}: {val3:+.2f} px")
                    print(f"    - Passed court test: {'YES' if is_inside else 'NO'}")

            # Deeper trace for raw track if raw_id is specified
            if raw_id is not None:
                print(f"\n[Deeper Trace for Raw ID {raw_id}]")
                if raw_id not in frame_raw:
                    print(f"Raw ID {raw_id} is NOT present in frame {f_idx} (raw detections).")
                else:
                    raw_data = frame_raw[raw_id]
                    raw_emb = raw_data.get('embedding')
                    raw_bbox = raw_data.get('bbox')
                    
                    if raw_emb is None:
                        print(f"Raw ID {raw_id} exists in frame {f_idx} but has NO embedding.")
                    elif not locked_gallery:
                        print("Gallery was not seeded, cannot run similarity check trace.")
                    else:
                        # Normalize raw track's embedding
                        norm_emb = raw_emb / (np.linalg.norm(raw_emb) + 1e-6)
                        
                        # Compute similarity against all locked Master ID templates
                        print("Cosine Similarity against locked Master templates:")
                        exceeded_ids = []
                        best_sim = -1.0
                        best_matched_master = None
                        
                        for m_id, gal_emb in locked_gallery.items():
                            sim = np.dot(norm_emb, gal_emb)
                            print(f"  - Master {m_id} Template: Similarity = {sim:.6f}")
                            if sim > 0.35:
                                exceeded_ids.append(m_id)
                            if sim > best_sim:
                                best_sim = sim
                                best_matched_master = m_id
                                
                        print(f"Best Matched Master ID: {best_matched_master} with score {best_sim:.6f}")
                        if exceeded_ids:
                            print(f"Exceeded 0.35 threshold for Master ID(s): {exceeded_ids}")
                        else:
                            print("Exceeded 0.35 threshold: None")
                            
                        # Re-simulate overlap-override logic
                        print("\nOverlap-Override Simulation:")
                        if best_matched_master is not None and best_sim > 0.35:
                            override_found = False
                            for other_tid, other_data in frame_raw.items():
                                if other_tid == raw_id or other_tid <= 0:
                                    continue
                                other_emb = other_data.get('embedding')
                                if other_emb is None:
                                    continue
                                other_norm_emb = other_emb / (np.linalg.norm(other_emb) + 1e-6)
                                other_sim = np.dot(other_norm_emb, locked_gallery[best_matched_master])
                                if other_sim > best_sim:
                                    print(f"  - OVERRIDDEN: Raw ID {other_tid} scored HIGHER against Master {best_matched_master} (Sim={other_sim:.6f} vs ours={best_sim:.6f})")
                                    override_found = True
                            if not override_found:
                                print("  - No override by another raw track detected (this track had the highest score for its target Master ID).")
                        else:
                            print("  - N/A: Similarity score did not exceed 0.35 threshold for any Master ID.")
                            
                        # Re-simulate positional fallback logic
                        print("\nPositional Fallback Simulation:")
                        # Find out which half of the court the target master_id should be on
                        # Master IDs: 1 & 2 are near court, 3 & 4 are far court
                        near_half = master_id in [1, 2]
                        half_label = "near" if near_half else "far"
                        half_ids = [1, 2] if near_half else [3, 4]
                        
                        # Reconstruct frame_clean simulating auto_player_filter matching pass
                        frame_clean_sim = {}
                        unmatched_tracks_sim = []
                        for sim_tid, sim_data in frame_raw.items():
                            if sim_tid <= 0:
                                continue
                            sim_emb = sim_data.get('embedding')
                            sim_bbox = sim_data.get('bbox')
                            if sim_emb is None or sim_bbox is None:
                                continue
                            
                            # check polygon test
                            sim_pt1 = (sim_bbox[0], sim_bbox[3])
                            sim_pt2 = ((sim_bbox[0] + sim_bbox[2]) / 2.0, sim_bbox[3])
                            sim_pt3 = (sim_bbox[2], sim_bbox[3])
                            sim_is_inside = (
                                cv2.pointPolygonTest(polygon, sim_pt1, False) >= 0 or
                                cv2.pointPolygonTest(polygon, sim_pt2, False) >= 0 or
                                cv2.pointPolygonTest(polygon, sim_pt3, False) >= 0
                            ) if polygon is not None else True
                            
                            if not sim_is_inside:
                                continue
                                
                            sim_norm_emb = sim_emb / (np.linalg.norm(sim_emb) + 1e-6)
                            s_best_master = None
                            s_best_sim = -1.0
                            for m_id, gal_emb in locked_gallery.items():
                                sim_val = np.dot(sim_norm_emb, gal_emb)
                                if sim_val > s_best_sim:
                                    s_best_sim = sim_val
                                    s_best_master = m_id
                                    
                            if s_best_master is not None and s_best_sim > 0.35:
                                if s_best_master in frame_clean_sim:
                                    p_sim = frame_clean_sim[s_best_master]['similarity']
                                    if s_best_sim > p_sim:
                                        p_bbox = frame_clean_sim[s_best_master]['bbox']
                                        unmatched_tracks_sim.append(p_bbox)
                                        frame_clean_sim[s_best_master] = {
                                            'bbox': sim_bbox,
                                            'similarity': s_best_sim,
                                            'raw_id': sim_tid
                                        }
                                    else:
                                        unmatched_tracks_sim.append(sim_bbox)
                                else:
                                    frame_clean_sim[s_best_master] = {
                                        'bbox': sim_bbox,
                                        'similarity': s_best_sim,
                                        'raw_id': sim_tid
                                    }
                            else:
                                unmatched_tracks_sim.append(sim_bbox)
                                
                        # Print simulation baseline mappings
                        print(f"  - Simulated Direct mappings: { {k: v['raw_id'] for k, v in frame_clean_sim.items()} }")
                        print(f"  - Target Master ID {master_id} vacant in simulated direct match stage: {'YES' if master_id not in frame_clean_sim else 'NO'}")
                        
                        # Trace fallback step
                        assigned_half = [id_ for id_ in half_ids if id_ in frame_clean_sim]
                        vacant_id = None
                        if len(assigned_half) == 1:
                            assigned_id = assigned_half[0]
                            vacant_id = half_ids[1] if assigned_id == half_ids[0] else half_ids[0]
                            print(f"  - One ID assigned on {half_label} half ({assigned_id}), leaving vacant ID: {vacant_id}")
                        else:
                            print(f"  - Number of assigned IDs on {half_label} half: {len(assigned_half)} (Fallback requires exactly 1 assigned ID on court half)")
                            
                        # Check if Raw ID 38 was in unmatched_tracks_sim
                        is_unmatched = False
                        for bbox in unmatched_tracks_sim:
                            if np.array_equal(bbox, raw_bbox):
                                is_unmatched = True
                                break
                        print(f"  - Raw ID {raw_id} present in unmatched tracks list: {'YES' if is_unmatched else 'NO'}")
                        
                        if is_unmatched and vacant_id is not None:
                            print(f"  - SUCCESS: Fallback logic would assign Raw ID {raw_id} to Vacant Master ID {vacant_id}")
                        else:
                            print("  - EXCLUSION: Fallback logic could not resolve this raw track.")
                            if not is_unmatched:
                                print(f"    * Reason: Raw ID {raw_id} was not in the unmatched list (likely dropped, out of court, or assigned directly to another Master ID).")
                            if vacant_id is None:
                                print(f"    * Reason: No vacant ID on the {half_label} half, or there was a conflict (more than one player already matching, or zero players matching on this half).")

        print(f"==================================================")

if __name__ == "__main__":
    main()
