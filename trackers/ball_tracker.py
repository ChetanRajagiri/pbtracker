import os
import pickle
import cv2
import pandas as pd
from ultralytics import YOLO

class BallTracker:
    def __init__(self, model_path='models/yolo5_last.pt'):
        print(f"Initializing BallTracker with model: {model_path}")
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        import numpy as np
        print("Interpolating missing ball positions using Pandas...")
        # Extract coordinates to a list, using None for missing frames
        extracted_coords = []
        # Track which frames had an actual YOLO detection
        detected_mask = []
        for pos in ball_positions:
            if 1 in pos:
                data = pos[1]
                # Support both dict and raw list formats
                bbox = data['bbox'] if isinstance(data, dict) else data
                extracted_coords.append(bbox)
                detected_mask.append(True)
            else:
                extracted_coords.append([None, None, None, None])
                detected_mask.append(False)
        
        # Create DataFrame
        df = pd.DataFrame(extracted_coords, columns=['x1', 'y1', 'x2', 'y2'])
        df['x_center'] = (df['x1'] + df['x2']) / 2.0
        df['y_center'] = (df['y1'] + df['y2']) / 2.0
        
        # Pass 1: Remove single-frame spikes
        for t in range(1, len(df) - 1):
            if detected_mask[t]:
                # find prev detected within 5 frames
                prev_idx = None
                for p in range(t-1, max(-1, t-6), -1):
                    if detected_mask[p]:
                        prev_idx = p
                        break
                # find next detected within 5 frames
                next_idx = None
                for n in range(t+1, min(len(df), t+6)):
                    if detected_mask[n]:
                        next_idx = n
                        break
                        
                if prev_idx is not None and next_idx is not None:
                    px, py = df.iloc[prev_idx]['x_center'], df.iloc[prev_idx]['y_center']
                    cx, cy = df.iloc[t]['x_center'], df.iloc[t]['y_center']
                    nx, ny = df.iloc[next_idx]['x_center'], df.iloc[next_idx]['y_center']
                    
                    d_prev = np.sqrt((cx - px)**2 + (cy - py)**2)
                    d_next = np.sqrt((nx - cx)**2 + (ny - cy)**2)
                    d_bridge = np.sqrt((nx - px)**2 + (ny - py)**2)
                    
                    speed_prev = d_prev / (t - prev_idx)
                    speed_next = d_next / (next_idx - t)
                    
                    if (speed_prev > 150 and speed_next > 150) or (d_prev > 150 and d_next > 150 and d_bridge < 120):
                        df.iloc[t] = [None, None, None, None, None, None]
                        detected_mask[t] = False

        # Pass 2: Remove two-frame consecutive spikes
        for t in range(1, len(df) - 2):
            if detected_mask[t] and detected_mask[t+1]:
                # Find prev detected before t
                prev_idx = None
                for p in range(t-1, max(-1, t-6), -1):
                    if detected_mask[p]:
                        prev_idx = p
                        break
                # Find next detected after t+1
                next_idx = None
                for n in range(t+2, min(len(df), t+7)):
                    if detected_mask[n]:
                        next_idx = n
                        break
                        
                if prev_idx is not None and next_idx is not None:
                    px, py = df.iloc[prev_idx]['x_center'], df.iloc[prev_idx]['y_center']
                    c1x, c1y = df.iloc[t]['x_center'], df.iloc[t]['y_center']
                    c2x, c2y = df.iloc[t+1]['x_center'], df.iloc[t+1]['y_center']
                    nx, ny = df.iloc[next_idx]['x_center'], df.iloc[next_idx]['y_center']
                    
                    d_prev = np.sqrt((c1x - px)**2 + (c1y - py)**2)
                    d_next = np.sqrt((nx - c2x)**2 + (ny - c2y)**2)
                    d_bridge = np.sqrt((nx - px)**2 + (ny - py)**2)
                    
                    speed_prev = d_prev / (t - prev_idx)
                    speed_next = d_next / (next_idx - (t+1))
                    
                    if (speed_prev > 150 and speed_next > 150) or (d_prev > 150 and d_next > 150 and d_bridge < 120):
                        df.iloc[t] = [None, None, None, None, None, None]
                        df.iloc[t+1] = [None, None, None, None, None, None]
                        detected_mask[t] = False
                        detected_mask[t+1] = False
                        
        # Drop temporary helper columns
        df = df[['x1', 'y1', 'x2', 'y2']]
        
        # Linearly interpolate missing bounding boxes
        df = df.interpolate(method='linear')
        
        # Backward-fill any missing detections at the start of the video
        df = df.bfill()
        
        # Re-format back to list of dictionaries with source tags
        interpolated_positions = []
        for idx, row in df.iterrows():
            if pd.isna(row['x1']):
                interpolated_positions.append({})
            else:
                source = 'detected' if detected_mask[idx] else 'interpolated'
                interpolated_positions.append({
                    1: {
                        'bbox': [row['x1'], row['y1'], row['x2'], row['y2']],
                        'source': source
                    }
                })
                
        return interpolated_positions

    def detect_frames(self, frames, read_from_stub=False, stub_path=None, verbose=True):
        # 1. Load from stub if requested and exists (with frame count validation)
        if read_from_stub and stub_path and os.path.exists(stub_path):
            if verbose:
                print(f"Loading ball tracking data from stub: {stub_path}")
            with open(stub_path, 'rb') as f:
                cached_data = pickle.load(f)
                if len(cached_data) == len(frames):
                    return cached_data
                else:
                    if verbose:
                        print(f"[WARNING] Cache mismatch detected. Stale tracking stub contains fewer frames than the active input video. Forcing full YOLO re-inference...")

        if verbose:
            print("Running ball detection on frames...")
        ball_detections = []
        last_ball_center = None
        lost_frames = 0

        # 2. Run prediction frame by frame
        for i, frame in enumerate(frames):
            # Run prediction with lower confidence threshold (0.08) to capture candidates
            results = self.model.predict(frame, conf=0.08, verbose=False)[0]
            
            frame_dict = {}
            if len(results.boxes) > 0:
                candidates = []
                for box in results.boxes:
                    coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    conf = float(box.conf[0])
                    cx = (coords[0] + coords[2]) / 2.0
                    cy = (coords[1] + coords[3]) / 2.0
                    candidates.append({'coords': coords, 'center': (cx, cy), 'conf': conf})
                
                best_candidate = None
                if last_ball_center is not None:
                    # Pick closest candidate
                    min_dist = float('inf')
                    for cand in candidates:
                        dist = np.sqrt((cand['center'][0] - last_ball_center[0])**2 + (cand['center'][1] - last_ball_center[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_candidate = cand
                            
                    # If the closest candidate is within 150 pixels, select it
                    if min_dist < 150:
                        last_ball_center = best_candidate['center']
                        frame_dict[1] = best_candidate['coords']
                        lost_frames = 0
                    else:
                        lost_frames += 1
                        if lost_frames > 5:
                            # Re-initialize to highest confidence candidate
                            sorted_cands = sorted(candidates, key=lambda x: x['conf'], reverse=True)
                            best_candidate = sorted_cands[0]
                            last_ball_center = best_candidate['center']
                            frame_dict[1] = best_candidate['coords']
                            lost_frames = 0
                else:
                    # Re-initialize to highest confidence candidate
                    sorted_cands = sorted(candidates, key=lambda x: x['conf'], reverse=True)
                    best_candidate = sorted_cands[0]
                    last_ball_center = best_candidate['center']
                    frame_dict[1] = best_candidate['coords']
                    lost_frames = 0
            else:
                lost_frames += 1
                if lost_frames > 5:
                    last_ball_center = None
            
            ball_detections.append(frame_dict)
            if verbose and ((i + 1) % 100 == 0 or (i + 1) == len(frames)):
                print(f"Processed {i + 1}/{len(frames)} frames for ball detection")

        # 3. Save to stub if path is provided
        if stub_path:
            stub_dir = os.path.dirname(stub_path)
            if stub_dir and not os.path.exists(stub_dir):
                os.makedirs(stub_dir)
            
            if verbose:
                print(f"Saving ball tracking data to stub: {stub_path}")
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def draw_bboxes(self, video_frames, ball_detections):
        annotated_frames = []
        for i, frame in enumerate(video_frames):
            # Create a copy to avoid modifying original frames in-place
            annotated_frame = frame.copy()
            
            # Get detections for current frame (if any)
            detections = ball_detections[i] if i < len(ball_detections) else {}
            
            if 1 in detections:
                data = detections[1]
                # Support both old format (plain list) and new tagged format (dict)
                if isinstance(data, dict):
                    bbox = data['bbox']
                    source = data.get('source', 'detected')
                else:
                    bbox = data
                    source = 'detected'
                
                x1, y1, x2, y2 = map(int, bbox)
                
                # Green box for YOLO-detected, pink box for Pandas-interpolated
                if source == 'detected':
                    box_color = (0, 255, 0)       # green (BGR)
                    circle_color = (0, 255, 0)
                    label = "Ball"
                else:
                    box_color = (180, 105, 255)   # pink (BGR)
                    circle_color = (180, 105, 255)
                    label = "Ball (pred)"
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Draw center dot
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(annotated_frame, (center_x, center_y), 4, circle_color, -1)
                
                # Draw label above the box
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), box_color, -1)
                cv2.putText(annotated_frame, label, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            
            annotated_frames.append(annotated_frame)
            
        return annotated_frames
