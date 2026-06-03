import os
import pickle
import numpy as np
import cv2

class AutoPlayerFilter:
    def __init__(self, detections_pkl='tracker_stubs/player_detections.pkl', court_pkl='tracker_stubs/court_keypoints.pkl'):
        self.detections_pkl = detections_pkl
        self.court_pkl = court_pkl
        self.player_detections = None
        self.court_keypoints = None

    def load_data(self):
        if not os.path.exists(self.detections_pkl):
            raise FileNotFoundError(f"Detections stub not found at: {self.detections_pkl}")
        if not os.path.exists(self.court_pkl):
            raise FileNotFoundError(f"Court keypoints stub not found at: {self.court_pkl}")
            
        with open(self.detections_pkl, 'rb') as f:
            self.player_detections = pickle.load(f)
        with open(self.court_pkl, 'rb') as f:
            self.court_keypoints = pickle.load(f)
            
        print(f"[LOAD] Loaded {len(self.player_detections)} frames of player detections.")
        print(f"[LOAD] Loaded {len(self.court_keypoints)} court keypoints.")

    def run_filtration(self, margin=30, kinetic_threshold=150):
        if self.player_detections is None or self.court_keypoints is None:
            self.load_data()

        # Step 1: Establish Strict Court Outer Polygon (Trapezoid)
        # Outer 4 corner indexes from manual 12-point selector:
        # Point 1 (Index 0): Top-Left Far Baseline
        # Point 3 (Index 2): Top-Right Far Baseline
        # Point 12 (Index 11): Bottom-Right Near Baseline
        # Point 10 (Index 9): Bottom-Left Near Baseline
        court_points = np.array(self.court_keypoints)
        corners = np.array([
            court_points[0],   # Top-Left Far
            court_points[2],   # Top-Right Far
            court_points[11],  # Bottom-Right Near
            court_points[9]    # Bottom-Left Near
        ], dtype=np.int32)
        
        print(f"[SPATIAL] Active Play Area defined by strict trapezoid: {corners.tolist()}")

        # Compile lifespans, spatial center sequences, and coordinates for all active Track IDs
        track_lifespans = {}  # {track_id: frame_count}
        track_spatial_coordinates = {}  # {track_id: list of foot coordinate tuples}
        track_box_centers = {}  # {track_id: list of (x_center, y_center)}
        
        for frame_idx, frame_dict in enumerate(self.player_detections):
            for track_id, data in frame_dict.items():
                bbox = data.get('bbox') if isinstance(data, dict) else data
                if bbox:
                    foot_x = (bbox[0] + bbox[2]) / 2.0
                    foot_y = bbox[3]
                    
                    x_center = (bbox[0] + bbox[2]) / 2.0
                    y_center = (bbox[1] + bbox[3]) / 2.0
                    
                    if track_id not in track_lifespans:
                        track_lifespans[track_id] = 0
                        track_spatial_coordinates[track_id] = []
                        track_box_centers[track_id] = []
                    
                    track_lifespans[track_id] += 1
                    track_spatial_coordinates[track_id].append((foot_x, foot_y))
                    track_box_centers[track_id].append((x_center, y_center))

        # --- Layer 1: Spatial Exclusion Filter (Strict Polygon Test) ---
        spatially_valid_ids = set()
        for track_id, coords in track_spatial_coordinates.items():
            ever_in_play_area = False
            for x, y in coords:
                # cv2.pointPolygonTest returns distance to boundary (+ inside, - outside)
                # We check if point is inside or within 'margin' pixels outside the polygon boundary
                dist = cv2.pointPolygonTest(corners, (float(x), float(y)), True)
                if dist >= -margin:
                    ever_in_play_area = True
                    break
            
            if ever_in_play_area:
                spatially_valid_ids.add(track_id)
            else:
                print(f"[SPATIAL PURGE] Track ID {track_id} deleted: Never entered the active court boundary (Lifespan: {track_lifespans[track_id]} frames).")

        # --- Layer 1.5: Kinetic Movement Filter (Motion Variance check) ---
        kinetically_valid_ids = set()
        for track_id in spatially_valid_ids:
            centers = track_box_centers[track_id]
            xs = [c[0] for c in centers]
            ys = [c[1] for c in centers]
            
            # Standard deviations
            std_x = np.std(xs) if len(xs) > 1 else 0.0
            std_y = np.std(ys) if len(ys) > 1 else 0.0
            
            # Coordinate range spread
            range_x = max(xs) - min(xs) if len(xs) > 0 else 0.0
            range_y = max(ys) - min(ys) if len(ys) > 0 else 0.0
            max_range = max(range_x, range_y)
            
            # Spectator rejection threshold: must have visible displacement/standard deviation
            # Real players show std_x or std_y >= 10.0 and a max range >= 60.0 pixels.
            if max(std_x, std_y) >= 10.0 or max_range >= 60.0:
                kinetically_valid_ids.add(track_id)
            else:
                print(f"[KINETIC PURGE] ID {track_id} deleted: Static spectator/referee "
                      f"(Std: ({std_x:.1f}, {std_y:.1f}), Range: {max_range:.1f} px "
                      f"over {track_lifespans[track_id]} frames)")

        # --- Layer 2: Temporal Filtering (Lifespan Sorter) ---
        surviving_lifespans = {tid: track_lifespans[tid] for tid in kinetically_valid_ids}
        
        # Sort by frame duration descending
        sorted_tracks = sorted(surviving_lifespans.items(), key=lambda item: item[1], reverse=True)
        
        print("\n--- Surviving Candidate Lifespans (Sorted Descending) ---")
        for rank, (tid, count) in enumerate(sorted_tracks, 1):
            print(f"Rank {rank}: Track ID {tid:<3} present in {count:<5} frames")
        
        # Take the top 4 longest-surviving players
        top_4_tracks = sorted_tracks[:4]
        master_player_ids = {tid for tid, _ in top_4_tracks}
        
        # Log temporal exclusions
        for rank, (tid, count) in enumerate(sorted_tracks[4:], 5):
            print(f"[TEMPORAL PURGE] Track ID {tid} deleted: Rank {rank} lifespan ({count} frames) fell below Top 4 threshold.")

        # --- Layer 3: Master ID Lock & Clean Remapping ---
        # Remap the top 4 tracks sequentially to keys: 1, 2, 3, and 4
        # Sorting the original IDs numerically before remapping preserves top/bottom court assignment consistency
        sorted_master_ids = sorted(list(master_player_ids))
        id_remap_scheme = {old_id: idx + 1 for idx, old_id in enumerate(sorted_master_ids)}
        
        print("\n--- Remapping Master IDs ---")
        for old_id, new_id in id_remap_scheme.items():
            print(f"[REMAPPED] Track ID {old_id} -> Master Player {new_id} (Lifespan: {track_lifespans[old_id]} frames)")

        # Generate clean player_detections dictionary
        cleaned_detections = []
        for frame_idx, frame_dict in enumerate(self.player_detections):
            new_frame_dict = {}
            for track_id, data in frame_dict.items():
                if track_id in id_remap_scheme:
                    new_id = id_remap_scheme[track_id]
                    # Retain data payload (bbox, is_on_court, etc.)
                    new_frame_dict[new_id] = data
            cleaned_detections.append(new_frame_dict)

        # Overwrite stub pickle
        with open(self.detections_pkl, 'wb') as f:
            pickle.dump(cleaned_detections, f)
            
        print(f"\n[SAVE] Saved clean filtered player detections back to: {self.detections_pkl}")
        return cleaned_detections

if __name__ == "__main__":
    raw_path = 'tracker_stubs/player_detections_raw.pkl'
    target_path = 'tracker_stubs/player_detections.pkl'
    if os.path.exists(raw_path):
        print(f"[FILTER] Found raw detections at: {raw_path}")
        filter_engine = AutoPlayerFilter(detections_pkl=raw_path)
        cleaned = filter_engine.run_filtration()
        # Save specifically to player_detections.pkl target stub
        with open(target_path, 'wb') as f:
            pickle.dump(cleaned, f)
        print(f"[FILTER] Successfully saved filtered players to target: {target_path}")
    else:
        print(f"[FILTER] Raw detections not found at {raw_path}. Filtering in-place on {target_path}...")
        filter_engine = AutoPlayerFilter(detections_pkl=target_path)
        filter_engine.run_filtration()
