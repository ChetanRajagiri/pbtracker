import os
import cv2
import pickle
import argparse
from math import hypot

def get_centroid(bbox):
    """Calculate the centroid of a bounding box."""
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

def draw_bboxes(frame, frame_dict):
    """Draw bounding boxes and Master ID labels on the frame."""
    annotated_frame = frame.copy()
    for track_id, data in frame_dict.items():
        if track_id not in [1, 2, 3, 4]:
            continue
            
        bbox = data['bbox']
        is_on_court = data.get('is_on_court', True)
        x1, y1, x2, y2 = map(int, bbox)
        
        # Bounding box color configuration
        color = (0, 255, 0) if is_on_court else (0, 165, 255)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Label with Master ID
        label = f"Player {track_id}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + w + 10, y1), color, -1)
        cv2.putText(annotated_frame, label, (x1 + 5, y1 - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                    
    return annotated_frame

def main():
    parser = argparse.ArgumentParser(description="Extract frames where players cross paths or cluster together.")
    parser.add_argument("--video", type=str, required=True, help="Path to the source video file")
    parser.add_argument("--same-half-threshold", type=float, default=150.0, 
                        help="Centroid distance threshold for pairs on the same half (default: 150px)")
    parser.add_argument("--cross-half-threshold", type=float, default=250.0, 
                        help="Centroid distance threshold for pairs across different halves (default: 250px)")
    args = parser.parse_args()

    pkl_path = "tracker_stubs/player_detections.pkl"
    if not os.path.exists(pkl_path):
        print(f"Error: Cached detections not found at {pkl_path}")
        return

    print(f"Loading cached player detections from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        detections = pickle.load(f)

    # Master ID groups
    # 1: Near-Left, 2: Near-Right, 3: Far-Left, 4: Far-Right
    same_half_pairs = [(1, 2), (3, 4)]
    cross_half_pairs = [(1, 3), (1, 4), (2, 3), (2, 4)]

    flagged_frames = []

    print("Analyzing frames for crossover candidates...")
    for idx, frame_dict in enumerate(detections):
        pairs_triggered = []
        centroids = {}
        
        # Extract centroids for master IDs
        for track_id, data in frame_dict.items():
            if track_id in [1, 2, 3, 4]:
                centroids[track_id] = get_centroid(data['bbox'])
        
        # Check same half thresholds
        for pair in same_half_pairs:
            if pair[0] in centroids and pair[1] in centroids:
                dist = hypot(centroids[pair[0]][0] - centroids[pair[1]][0], 
                             centroids[pair[0]][1] - centroids[pair[1]][1])
                if dist < args.same_half_threshold:
                    pairs_triggered.append(pair)
                    
        # Check cross half thresholds
        for pair in cross_half_pairs:
            if pair[0] in centroids and pair[1] in centroids:
                dist = hypot(centroids[pair[0]][0] - centroids[pair[1]][0], 
                             centroids[pair[0]][1] - centroids[pair[1]][1])
                if dist < args.cross_half_threshold:
                    pairs_triggered.append(pair)
                    
        if pairs_triggered:
            flagged_frames.append({
                'frame': idx,
                'pairs': pairs_triggered
            })

    if not flagged_frames:
        print("No crossover frames found based on the provided thresholds.")
        return

    # Group contiguous flagged frames into windows with padding
    padding = 10
    windows = []
    
    current_window = {
        'start': max(0, flagged_frames[0]['frame'] - padding),
        'end': flagged_frames[0]['frame'] + padding,
        'pairs': set(flagged_frames[0]['pairs'])
    }
    
    for item in flagged_frames[1:]:
        f = item['frame']
        p = item['pairs']
        
        window_start = max(0, f - padding)
        window_end = f + padding
        
        # If overlaps or contiguous with current window
        if window_start <= current_window['end'] + 1:
            current_window['end'] = max(current_window['end'], window_end)
            current_window['pairs'].update(p)
        else:
            windows.append(current_window)
            current_window = {
                'start': window_start,
                'end': window_end,
                'pairs': set(p)
            }
            
    # Append the last window
    windows.append(current_window)
    
    # Ensure end boundaries don't exceed video length based on detections array length
    max_frame = len(detections) - 1
    for w in windows:
        w['end'] = min(w['end'], max_frame)

    print(f"Found {len(windows)} crossover windows. Extracting frames...")

    # Extract and draw frames
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video file at {args.video}")
        return

    out_base_dir = os.path.join("debug", "crossover_frames")
    os.makedirs(out_base_dir, exist_ok=True)

    current_frame_idx = 0
    window_idx = 0
    
    while window_idx < len(windows):
        w = windows[window_idx]
        
        # Fast-forward until we are at the start of the window
        while current_frame_idx < w['start']:
            ret = cap.grab()
            if not ret:
                break
            current_frame_idx += 1
            
        # Process frames within the window
        while current_frame_idx <= w['end']:
            ret, frame = cap.read()
            if not ret:
                break
                
            window_dir = os.path.join(out_base_dir, f"window_{w['start']}_{w['end']}")
            os.makedirs(window_dir, exist_ok=True)
            
            frame_dict = detections[current_frame_idx] if current_frame_idx < len(detections) else {}
            annotated_frame = draw_bboxes(frame, frame_dict)
            
            out_path = os.path.join(window_dir, f"frame_{current_frame_idx}.jpg")
            cv2.imwrite(out_path, annotated_frame)
            
            current_frame_idx += 1
            
        window_idx += 1

    cap.release()
    
    # Summary
    print("\n" + "="*50)
    print("             CROSSOVER EXTRACTION SUMMARY")
    print("="*50)
    print(f"Total crossover windows found: {len(windows)}")
    for i, w in enumerate(windows):
        # Format the pairs nicely (e.g. 1&2, 2&4)
        pairs_str = ", ".join([f"{p[0]}&{p[1]}" for p in sorted(list(w['pairs']))])
        print(f"Window {i+1}: Frames {w['start']} to {w['end']} | Triggered by pairs: {pairs_str}")
    print("="*50)

if __name__ == "__main__":
    main()
