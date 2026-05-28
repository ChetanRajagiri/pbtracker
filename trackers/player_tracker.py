import os
import pickle
import cv2
from ultralytics import YOLO

class PlayerTracker:
    def __init__(self, model_path='yolov8x.pt'):
        print(f"Initializing PlayerTracker with model: {model_path}")
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        # 1. Load from stub if requested and exists
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"Loading player tracking data from stub: {stub_path}")
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

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
                    frame_dict[track_id] = coords
            
            player_detections.append(frame_dict)
            if (i + 1) % 10 == 0 or (i + 1) == len(frames):
                print(f"Tracked {i + 1}/{len(frames)} frames")

        # 3. Save to stub if path is provided
        if stub_path:
            # Create directories if they do not exist
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
            
            for track_id, bbox in detections.items():
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw the bounding box
                # Primary color: neon cyan/green for premium styling (BGR format: (0, 255, 0))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw the Tracking ID label
                label = f"Player {track_id}"
                
                # Get text height and width for styling background box
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Draw label background box
                cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + w + 10, y1), (0, 255, 0), -1)
                
                # Put label text (Black text on green background)
                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            
            annotated_frames.append(annotated_frame)
            
        return annotated_frames
