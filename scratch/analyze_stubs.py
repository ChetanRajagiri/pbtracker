import pickle
import os
import cv2
import numpy as np

def main():
    player_stub = "tracker_stubs/player_detections.pkl"
    court_stub = "tracker_stubs/court_keypoints.pkl"
    
    if not os.path.exists(player_stub):
        print("No player stub found!")
        return
        
    with open(player_stub, 'rb') as f:
        player_detections = pickle.load(f)
        
    print(f"Total frames in player detections: {len(player_detections)}")
    
    # Analyze all unique IDs and their lifespan
    id_lifespans = {}
    id_on_court_counts = {}
    id_bboxes = {}
    
    for frame_idx, frame_dict in enumerate(player_detections):
        for track_id, data in frame_dict.items():
            bbox = data.get('bbox') if isinstance(data, dict) else data
            is_on_court = data.get('is_on_court', True) if isinstance(data, dict) else True
            
            if track_id not in id_lifespans:
                id_lifespans[track_id] = 0
                id_on_court_counts[track_id] = 0
                id_bboxes[track_id] = []
            id_lifespans[track_id] += 1
            if is_on_court:
                id_on_court_counts[track_id] += 1
            id_bboxes[track_id].append(bbox)
            
    print("\n--- Track ID Statistics ---")
    for tid in sorted(id_lifespans.keys()):
        count = id_lifespans[tid]
        on_court = id_on_court_counts[tid]
        bboxes = id_bboxes[tid]
        # Calculate coordinate range
        xs = [ (b[0] + b[2])/2.0 for b in bboxes ]
        ys = [ (b[1] + b[3])/2.0 for b in bboxes ]
        x_range = max(xs) - min(xs) if xs else 0
        y_range = max(ys) - min(ys) if ys else 0
        print(f"Track ID {tid}: Lifespan = {count} frames, On-Court = {on_court} frames, X-Range = {x_range:.1f}px, Y-Range = {y_range:.1f}px")

if __name__ == "__main__":
    main()
