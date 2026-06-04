import os
import pickle
import cv2
import numpy as np
from ultralytics import YOLO

class PlayerTracker:
    def __init__(self, model_path='yolov8x.pt'):
        print(f"Initializing PlayerTracker with model: {model_path}")
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None, court_keypoints_path="tracker_stubs/court_keypoints.pkl"):
        # 1. Load from stub if requested and exists (with frame count validation)
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"Loading player tracking data from stub: {stub_path}")
            with open(stub_path, 'rb') as f:
                cached_data = pickle.load(f)
                if len(cached_data) == len(frames):
                    return cached_data
                else:
                    print(f"[WARNING] Cache mismatch detected. Stale tracking stub contains fewer frames than the active input video. Forcing full YOLO re-inference...")

        # Load court keypoints to define boundaries
        polygon = None
        if court_keypoints_path and os.path.exists(court_keypoints_path):
            print(f"Loading court keypoints for boundary filtering: {court_keypoints_path}")
            with open(court_keypoints_path, 'rb') as f:
                court_keypoints = pickle.load(f)
            if len(court_keypoints) >= 12:
                corner1 = court_keypoints[0]   # Top-Left Far Baseline
                corner2 = court_keypoints[2]   # Top-Right Far Baseline
                corner3 = court_keypoints[11]  # Bottom-Right Near Baseline
                corner4 = court_keypoints[9]   # Bottom-Left Near Baseline
                polygon = np.array([corner1, corner2, corner3, corner4], dtype=np.int32)
            else:
                print(f"[WARNING] Insufficient court keypoints ({len(court_keypoints)}) to define polygon. Skipping filtering.")
        else:
            print(f"[WARNING] No court keypoints found at {court_keypoints_path}. Skipping boundary filtering.")

        print("Running object tracking on frames...")
        player_detections = []

        # 2. Run tracking frame by frame
        for i, frame in enumerate(frames):
            # Run tracking with human class only (class 0)
            results = self.model.track(frame, persist=True, classes=[0])[0]
            
            frame_dict = {}
            for box in results.boxes:
                # check if track id exists
                if box.id is not None:
                    track_id = int(box.id[0].item())
                    coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    
                    is_on_court = True
                    if polygon is not None:
                        # Calculate ground contact anchor point (feet)
                        x_foot = (coords[0] + coords[2]) / 2.0
                        y_foot = coords[3]
                        
                        # cv2.pointPolygonTest returns >= 0 if inside or on boundary
                        is_inside = cv2.pointPolygonTest(polygon, (x_foot, y_foot), False)
                        is_on_court = True if is_inside >= 0 else False
                    
                    frame_dict[track_id] = {
                        'bbox': coords,
                        'is_on_court': is_on_court
                    }
            
            player_detections.append(frame_dict)
            if (i + 1) % 10 == 0 or (i + 1) == len(frames):
                print(f"Tracked {i + 1}/{len(frames)} frames")

        # 3. Save to stub if path is provided
        if stub_path:
            stub_dir = os.path.dirname(stub_path)
            if stub_dir and not os.path.exists(stub_dir):
                os.makedirs(stub_dir)
            
            print(f"Saving player tracking data to stub: {stub_path}")
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def draw_bboxes(self, video_frames, player_detections):
        annotated_frames = []
        for i, frame in enumerate(video_frames):
            # Create a copy to avoid modifying original frames in-place
            annotated_frame = frame.copy()
            
            # Get detections for current frame (if any)
            detections = player_detections[i] if i < len(player_detections) else {}
            
            for track_id, data in detections.items():
                bbox = data['bbox']
                is_on_court = data.get('is_on_court', True)
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw the bounding box (neon green if on court, orange if out)
                color = (0, 255, 0) if is_on_court else (0, 165, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw the Tracking ID label
                label = f"Player {track_id}"
                if not is_on_court:
                    label += " (Out)"
                
                # Get text height and width for styling background box
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Draw label background box
                cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + w + 10, y1), color, -1)
                
                # Put label text (Black text on background color)
                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            
            annotated_frames.append(annotated_frame)
            
        return annotated_frames
