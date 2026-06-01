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
        for pos in ball_positions:
            if 1 in pos:
                extracted_coords.append(pos[1])
            else:
                extracted_coords.append([None, None, None, None])
        
        # Create DataFrame
        df = pd.DataFrame(extracted_coords, columns=['x1', 'y1', 'x2', 'y2'])
        
        # Linearly interpolate missing bounding boxes
        df = df.interpolate(method='linear')
        
        # Backward-fill any missing detections at the start of the video
        df = df.bfill()
        
        # Re-format back to list of dictionaries
        interpolated_positions = []
        for _, row in df.iterrows():
            if pd.isna(row['x1']):
                interpolated_positions.append({})
            else:
                interpolated_positions.append({1: [row['x1'], row['y1'], row['x2'], row['y2']]})
                
        return interpolated_positions

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        # 1. Load from stub if requested and exists
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"Loading ball tracking data from stub: {stub_path}")
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

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
                bbox = detections[1]
                x1, y1, x2, y2 = map(int, bbox)
                
                # Calculate the center of the bounding box to draw a circle marker
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Draw a filled yellow circle (BGR: (0, 255, 255))
                cv2.circle(annotated_frame, (center_x, center_y), 6, (0, 255, 255), -1)
                # Draw a thin black border around the yellow circle for better visibility
                cv2.circle(annotated_frame, (center_x, center_y), 6, (0, 0, 0), 1)
            
            annotated_frames.append(annotated_frame)
            
        return annotated_frames
