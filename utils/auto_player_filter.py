import os
import pickle
import cv2
import numpy as np

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

class AutoPlayerFilter:
    def __init__(self, detections_pkl='tracker_stubs/player_detections.pkl', court_pkl='tracker_stubs/court_keypoints.pkl', video_path=None):
        self.detections_pkl = detections_pkl
        self.court_pkl = court_pkl
        self.video_path = video_path
        self.player_detections = None


    def load_data(self):
        if not os.path.exists(self.detections_pkl):
            raise FileNotFoundError(f"Detections stub not found at: {self.detections_pkl}")
            
        with open(self.detections_pkl, 'rb') as f:
            self.player_detections = pickle.load(f)
            
        print(f"[LOAD] Loaded {len(self.player_detections)} frames of player detections.")

    def run_filtration(self):
        if self.player_detections is None:
            self.load_data()

        # Determine frame height
        frame_height = 1080
        video_file = self.video_path
        if not video_file:
            import glob
            video_files = []
            for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
                video_files.extend(glob.glob(os.path.join("input_videos", ext)))
            if video_files:
                video_file = video_files[0]
        if video_file and os.path.exists(video_file):
            cap = cv2.VideoCapture(video_file)
            if cap.isOpened():
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

        # Load court keypoints to define expanded polygon
        polygon = None
        if self.court_pkl and os.path.exists(self.court_pkl):
            with open(self.court_pkl, 'rb') as f:
                court_keypoints = pickle.load(f)
            if len(court_keypoints) >= 12:
                corner1 = court_keypoints[0]   # Top-Left Far Baseline
                corner2 = court_keypoints[2]   # Top-Right Far Baseline
                corner3 = court_keypoints[11]  # Bottom-Right Near Baseline
                corner4 = court_keypoints[9]   # Bottom-Left Near Baseline
                polygon = np.array([corner1, corner2, corner3, corner4], dtype=np.int32)
                polygon = expand_polygon_asymmetric(polygon, frame_height)

        # Seeding phase variables
        consecutive_frames = {}  # {raw_tid: count}
        stable_tracks = set()
        gallery_features = {}    # {raw_tid: [embeddings]}
        gallery_bboxes = {}      # {raw_tid: [bboxes]}
        
        lock_frame = -1
        gallery_seeded = False
        id_mapping = {}
        locked_gallery = {}
        
        # Seeding Pass: Loop frame-by-frame until all 4 slots are filled, or frame 150, or video end
        for idx in range(len(self.player_detections)):
            if not gallery_seeded:
                frame_dict = self.player_detections[idx]
                active_tids = set(frame_dict.keys())
                
                # Update consecutive frames for active tracks inside the expanded polygon
                for tid in active_tids:
                    if tid <= 0:
                        continue
                    
                    data = frame_dict[tid]
                    bbox = data.get('bbox')
                    if bbox is None:
                        continue
                        
                    # Check expanded polygon using 3-point feet check
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
                        
                        # Collect features for seeding
                        if 'embedding' in data and data['embedding'] is not None:
                            gallery_features.setdefault(tid, []).append(data['embedding'])
                            gallery_bboxes.setdefault(tid, []).append(bbox)
                    else:
                        consecutive_frames[tid] = 0
                        
                # Reset consecutive count for any tracks not in active_tids or reset
                for tid in list(consecutive_frames.keys()):
                    if tid not in active_tids:
                        consecutive_frames[tid] = 0
                        
                # Check for stable tracks (>= 4 consecutive frames in court)
                for tid, count in consecutive_frames.items():
                    if count >= 4:
                        stable_tracks.add(tid)
                        
                # Lock condition check
                should_lock = False
                if len(stable_tracks) >= 4:
                    should_lock = True
                elif idx >= 150:
                    should_lock = True
                elif idx == len(self.player_detections) - 1:
                    should_lock = True
                    
                if should_lock:
                    # Select up to 4 stable tracks sorted by total observation count
                    selected_tids = sorted(list(stable_tracks), key=lambda t: len(gallery_features.get(t, [])), reverse=True)[:4]
                    
                    if len(selected_tids) > 0:
                        # Quadrant sort by average coordinate centers
                        avg_coords = {}
                        for tid in selected_tids:
                            bboxes = gallery_bboxes[tid]
                            xs = [(b[0] + b[2]) / 2.0 for b in bboxes]
                            ys = [b[3] for b in bboxes]
                            avg_coords[tid] = (np.mean(xs), np.mean(ys))
                            
                        # Y sort: Far vs Near court
                        sorted_by_y = sorted(selected_tids, key=lambda tid: avg_coords[tid][1])
                        
                        half = len(sorted_by_y) // 2
                        far_pair = sorted_by_y[:half]
                        near_pair = sorted_by_y[half:]
                        
                        # X sort: Left to Right
                        far_sorted_x = sorted(far_pair, key=lambda tid: avg_coords[tid][0])
                        near_sorted_x = sorted(near_pair, key=lambda tid: avg_coords[tid][0])
                        
                        # Map to Master IDs 1-4
                        id_mapping = {}
                        if len(near_sorted_x) >= 1:
                            id_mapping[near_sorted_x[0]] = 1
                        if len(near_sorted_x) >= 2:
                            id_mapping[near_sorted_x[1]] = 2
                        if len(far_sorted_x) >= 1:
                            id_mapping[far_sorted_x[0]] = 3
                        if len(far_sorted_x) >= 2:
                            id_mapping[far_sorted_x[1]] = 4
                            
                        # Compute L2 normalized template embeddings
                        for raw_tid, master_id in id_mapping.items():
                            embs = gallery_features[raw_tid]
                            avg_emb = np.mean(embs, axis=0)
                            norm_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-6)
                            locked_gallery[master_id] = norm_emb
                            
                        gallery_seeded = True
                        lock_frame = idx
                        print(f"[REID] Seeding complete! Gallery locked at frame {lock_frame} with {len(selected_tids)} players:")
                        for raw_tid, master_id in id_mapping.items():
                            print(f"  - Raw ID {raw_tid} -> Master Player {master_id} (seeded with {len(gallery_features[raw_tid])} crops)")
                    else:
                        print("[WARNING] Seeding lock condition met, but no stable on-court tracks were found.")
                        # Force lock anyway using whatever active tracks we have as a safety fallback
                        active_on_court = [tid for tid in gallery_features.keys() if len(gallery_features[tid]) > 0]
                        selected_tids = sorted(active_on_court, key=lambda t: len(gallery_features[t]), reverse=True)[:4]
                        if len(selected_tids) > 0:
                            # Run quadrant mapping and locked gallery initialization on fallback set
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
                            for raw_tid, master_id in id_mapping.items():
                                embs = gallery_features[raw_tid]
                                avg_emb = np.mean(embs, axis=0)
                                norm_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-6)
                                locked_gallery[master_id] = norm_emb
                            gallery_seeded = True
                            lock_frame = idx

        # Matching Pass: Map all frames (retroactively and forward) against locked gallery
        self.osnet_direct_match_count = 0
        self.tie_break_resolved_count = 0
        self.court_side_fallback_count = 0
        self.unresolved_count = 0
        self.court_side_fallback_logs = []
        self.tiebreak_logs = []
        
        last_known_positions = {}
        if gallery_seeded:
            for raw_tid, master_id in id_mapping.items():
                if raw_tid in gallery_bboxes and len(gallery_bboxes[raw_tid]) > 0:
                    bboxes = gallery_bboxes[raw_tid]
                    last_known_positions[master_id] = (np.mean([(b[0]+b[2])/2.0 for b in bboxes]), np.mean([(b[1]+b[3])/2.0 for b in bboxes]))

        cleaned_detections = []
        for idx, frame_dict in enumerate(self.player_detections):
            frame_clean = {}
            unmatched_tracks = []
            if gallery_seeded:
                for raw_tid, data in frame_dict.items():
                    if raw_tid <= 0:
                        continue
                        
                    emb = data.get('embedding')
                    bbox = data.get('bbox')
                    
                    if emb is None or bbox is None:
                        continue
                        
                    # Normalize embedding
                    norm_emb = emb / (np.linalg.norm(emb) + 1e-6)
                    
                    # Cosine Similarity matching against Master players
                    scores = []
                    for master_id, gal_emb in locked_gallery.items():
                        sim = np.dot(norm_emb, gal_emb)
                        scores.append((master_id, sim))
                        
                    scores.sort(key=lambda x: x[1], reverse=True)
                    best_master_id = scores[0][0]
                    best_sim = scores[0][1]
                    
                    assigned_master_id = None
                    resolved_by_tie_break = False
                    
                    if best_sim > 0.35:
                        if len(scores) >= 2:
                            second_master_id = scores[1][0]
                            second_sim = scores[1][1]
                            
                            TIE_BREAK_MARGIN = 0.05
                            if best_sim - second_sim <= TIE_BREAK_MARGIN:
                                track_cx = (bbox[0] + bbox[2]) / 2.0
                                track_cy = (bbox[1] + bbox[3]) / 2.0
                                
                                pos1 = last_known_positions.get(best_master_id)
                                pos2 = last_known_positions.get(second_master_id)
                                
                                dist1 = float('inf')
                                dist2 = float('inf')
                                
                                if pos1:
                                    dist1 = np.linalg.norm([track_cx - pos1[0], track_cy - pos1[1]])
                                if pos2:
                                    dist2 = np.linalg.norm([track_cx - pos2[0], track_cy - pos2[1]])
                                    
                                winning_master = best_master_id
                                losing_master = second_master_id
                                if dist2 < dist1:
                                    best_master_id = second_master_id
                                    best_sim = second_sim
                                    resolved_by_tie_break = True
                                    winning_master = second_master_id
                                    losing_master = scores[0][0]
                                else:
                                    resolved_by_tie_break = True  # It was evaluated via tie break logic
                                    
                                # Log tie break event
                                self.tiebreak_logs.append({
                                    'frame_index': idx,
                                    'master_id': winning_master,
                                    'top_score': scores[0][1],
                                    'second_score': scores[1][1],
                                    'score_gap': scores[0][1] - scores[1][1],
                                    'winning_raw_id': raw_tid,
                                    'losing_raw_id': losing_master
                                })
                                    
                        assigned_master_id = best_master_id

                    if assigned_master_id is not None:
                        if assigned_master_id in frame_clean:
                            prev_sim = frame_clean[assigned_master_id]['similarity']
                            if best_sim > prev_sim:
                                # Overridden track goes to unmatched_tracks
                                prev_bbox = frame_clean[assigned_master_id]['bbox']
                                prev_raw_tid = frame_clean[assigned_master_id].get('raw_tid', -1)
                                unmatched_tracks.append({'bbox': prev_bbox, 'raw_tid': prev_raw_tid})
                                frame_clean[assigned_master_id] = {
                                    'bbox': bbox,
                                    'is_on_court': True,
                                    'similarity': best_sim,
                                    'resolved_by': 'tie_break' if resolved_by_tie_break else 'direct',
                                    'raw_tid': raw_tid
                                }
                            else:
                                unmatched_tracks.append({'bbox': bbox, 'raw_tid': raw_tid})
                        else:
                            frame_clean[assigned_master_id] = {
                                    'bbox': bbox,
                                    'is_on_court': True,
                                    'similarity': best_sim,
                                    'resolved_by': 'tie_break' if resolved_by_tie_break else 'direct',
                                    'raw_tid': raw_tid
                            }
                    else:
                        unmatched_tracks.append({'bbox': bbox, 'raw_tid': raw_tid})
                        
                for mid, val in frame_clean.items():
                    res_by = val.get('resolved_by')
                    if res_by == 'direct':
                        self.osnet_direct_match_count += 1
                    elif res_by == 'tie_break':
                        self.tie_break_resolved_count += 1
                    
                    b = val['bbox']
                    last_known_positions[mid] = ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0)

                # Fallback Assignment: assign unmatched tracks inside court to nearest unoccupied ID
                for item in unmatched_tracks:
                    bbox = item['bbox']
                    raw_tid = item['raw_tid']
                    
                    # Filter by polygon before fallback
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
                        
                    if not is_inside:
                        continue

                    foot_y = bbox[3]
                    assigned = False
                    if foot_y > frame_height * 0.5:
                        # Near side (IDs 1 and 2)
                        assigned_near = [id_ for id_ in [1, 2] if id_ in frame_clean]
                        if len(assigned_near) == 1:
                            assigned_id = assigned_near[0]
                            vacant_id = 2 if assigned_id == 1 else 1
                            frame_clean[vacant_id] = {
                                'bbox': bbox,
                                'is_on_court': True,
                                'similarity': 0.0
                            }
                            self.court_side_fallback_count += 1
                            self.court_side_fallback_logs.append({
                                'frame': idx,
                                'raw_tid': raw_tid,
                                'master_id': vacant_id
                            })
                            last_known_positions[vacant_id] = ((bbox[0]+bbox[2])/2.0, (bbox[1]+bbox[3])/2.0)
                            assigned = True
                    else:
                        # Far side (IDs 3 and 4)
                        assigned_far = [id_ for id_ in [3, 4] if id_ in frame_clean]
                        if len(assigned_far) == 1:
                            assigned_id = assigned_far[0]
                            vacant_id = 4 if assigned_id == 3 else 3
                            frame_clean[vacant_id] = {
                                'bbox': bbox,
                                'is_on_court': True,
                                'similarity': 0.0
                            }
                            self.court_side_fallback_count += 1
                            self.court_side_fallback_logs.append({
                                'frame': idx,
                                'raw_tid': raw_tid,
                                'master_id': vacant_id
                            })
                            last_known_positions[vacant_id] = ((bbox[0]+bbox[2])/2.0, (bbox[1]+bbox[3])/2.0)
                            assigned = True
                            
                    if not assigned:
                        self.unresolved_count += 1
                            
            # Remove similarity key and format cleanly
            final_frame_dict = {}
            for master_id, val in frame_clean.items():
                final_frame_dict[master_id] = {
                    'bbox': val['bbox'],
                    'is_on_court': val['is_on_court']
                }
            cleaned_detections.append(final_frame_dict)

        # Overwrite stub pickle with clean mapped data
        with open(self.detections_pkl, 'wb') as f:
            pickle.dump(cleaned_detections, f)
            
        print(f"[REID] Successfully mapped all frames to Master IDs 1-4. Saved to {self.detections_pkl}")
        
        # Save tiebreak logs
        tiebreak_csv = 'tracker_stubs/tiebreak_events.csv'
        try:
            import csv
            with open(tiebreak_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['frame_index', 'master_id', 'top_score', 'second_score', 'score_gap', 'winning_raw_id', 'losing_raw_id'])
                writer.writeheader()
                for row in self.tiebreak_logs:
                    writer.writerow(row)
            print(f"[INSTRUMENTATION] Saved tie-break events to {tiebreak_csv}")
        except Exception as e:
            print(f"[WARNING] Failed to write tiebreak_events.csv: {e}")
        
        total_assignments = self.osnet_direct_match_count + self.tie_break_resolved_count + self.court_side_fallback_count + self.unresolved_count
        if total_assignments > 0:
            print("\n[INSTRUMENTATION] Filtration Summary:")
            print(f"  - Total assignments attempted: {total_assignments}")
            print(f"  - Direct OSNet matches: {self.osnet_direct_match_count} ({(self.osnet_direct_match_count/total_assignments)*100:.1f}%)")
            print(f"  - Tie-break resolved: {self.tie_break_resolved_count} ({(self.tie_break_resolved_count/total_assignments)*100:.1f}%)")
            print(f"  - Positional Fallback: {self.court_side_fallback_count} ({(self.court_side_fallback_count/total_assignments)*100:.1f}%)")
            print(f"  - Unresolved (Dropped): {self.unresolved_count} ({(self.unresolved_count/total_assignments)*100:.1f}%)")
            print(f"  - Fallback logs: {len(self.court_side_fallback_logs)}")
            
        return cleaned_detections

if __name__ == "__main__":
    raw_path = 'tracker_stubs/player_detections_raw.pkl'
    target_path = 'tracker_stubs/player_detections.pkl'
    if os.path.exists(raw_path):
        print(f"[FILTER] Found raw detections at: {raw_path}")
        filter_engine = AutoPlayerFilter(detections_pkl=raw_path)
        cleaned = filter_engine.run_filtration()
        with open(target_path, 'wb') as f:
            pickle.dump(cleaned, f)
        print(f"[FILTER] Successfully saved filtered players to target: {target_path}")
    else:
        print(f"[FILTER] Filtering on {target_path}...")
        filter_engine = AutoPlayerFilter(detections_pkl=target_path)
        filter_engine.run_filtration()
