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
        print("Interpolating missing ball positions using Pandas...")
        # Extract coordinates to a list, using None for missing frames
        extracted_coords = []
        # Track which frames had an actual YOLO detection
        detected_mask = []
        for pos in ball_positions:
            if 1 in pos:
                extracted_coords.append(pos[1])
                detected_mask.append(True)
            else:
                extracted_coords.append([None, None, None, None])
                detected_mask.append(False)
        
        # Create DataFrame
        df = pd.DataFrame(extracted_coords, columns=['x1', 'y1', 'x2', 'y2'])
        
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

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        # 1. Load from stub if requested and exists (with frame count validation)
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"Loading ball tracking data from stub: {stub_path}")
            with open(stub_path, 'rb') as f:
                cached_data = pickle.load(f)
                if len(cached_data) == len(frames):
                    return cached_data
                else:
                    print(f"[WARNING] Cache mismatch detected. Stale tracking stub contains fewer frames than the active input video. Forcing full YOLO re-inference...")

        print("Running ball detection on frames...")
        ball_detections = []

        # 2. Run prediction frame by frame
        for i, frame in enumerate(frames):
            # Run prediction with conf=0.15
            results = self.model.predict(frame, conf=0.15)[0]
            
            frame_dict = {}
            # If a ball is detected, store the first detection (usually highest confidence) under key 1
            if len(results.boxes) > 0:
                box = results.boxes[0]
                coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                frame_dict[1] = coords
            
            ball_detections.append(frame_dict)
            if (i + 1) % 10 == 0 or (i + 1) == len(frames):
                print(f"Processed {i + 1}/{len(frames)} frames for ball detection")

        # 3. Save to stub if path is provided
        if stub_path:
            stub_dir = os.path.dirname(stub_path)
            if stub_dir and not os.path.exists(stub_dir):
                os.makedirs(stub_dir)
            
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
