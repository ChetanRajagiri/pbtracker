import pickle
import numpy as np
import cv2

def main():
    with open("tracker_stubs/player_detections_raw.pkl", 'rb') as f:
        player_detections = pickle.load(f)
    with open("tracker_stubs/court_keypoints.pkl", 'rb') as f:
        court_keypoints = pickle.load(f)

    margin = 30
    kinetic_threshold = 150
    
    court_points = np.array(court_keypoints)
    corners = np.array([
        court_points[0],   # Top-Left Far
        court_points[2],   # Top-Right Far
        court_points[11],  # Bottom-Right Near
        court_points[9]    # Bottom-Left Near
    ], dtype=np.int32)
    
    print(f"Trapezoid corners: {corners.tolist()}")
    
    track_lifespans = {}
    track_spatial_coordinates = {}
    track_box_centers = {}
    
    for frame_idx, frame_dict in enumerate(player_detections):
        for track_id, data in frame_dict.items():
            bbox = data.get('bbox')
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

    print("\n--- Filtration Analysis ---")
    for track_id, coords in track_spatial_coordinates.items():
        if track_lifespans[track_id] < 10: # ignore very short tracks for cleaner print
            continue
            
        # Layer 1: Spatial exclusion
        ever_in_play_area = False
        inside_count = 0
        min_dist = float('inf')
        for x, y in coords:
            dist = cv2.pointPolygonTest(corners, (float(x), float(y)), True)
            if dist >= -margin:
                ever_in_play_area = True
            if dist >= 0:
                inside_count += 1
            if dist < min_dist:
                min_dist = dist
                
        # Layer 1.5: Kinetic movement
        centers = track_box_centers[track_id]
        cumulative_movement = 0.0
        for i in range(1, len(centers)):
            prev = centers[i-1]
            curr = centers[i]
            dist = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
            cumulative_movement += dist
            
        center_xs = [c[0] for c in centers]
        center_ys = [c[1] for c in centers]
        x_range = max(center_xs) - min(center_xs)
        y_range = max(center_ys) - min(center_ys)
        
        # Calculate mean coords to see where they are
        mean_x = np.mean(center_xs)
        mean_y = np.mean(center_ys)
        
        print(f"Raw ID {track_id:<3}: Lifespan = {track_lifespans[track_id]:<4} | "
              f"Spatial = {'PASS' if ever_in_play_area else 'FAIL'} (MinDist = {min_dist:+.1f}, InsidePct = {inside_count/len(coords)*100:.1f}%) | "
              f"Kinetic = {cumulative_movement:.1f} px | "
              f"Range = X:{x_range:.1f} Y:{y_range:.1f} | "
              f"Mean Pos = ({mean_x:.1f}, {mean_y:.1f})")

if __name__ == "__main__":
    main()
