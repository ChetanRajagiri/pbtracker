import os
import pickle
import cv2
import numpy as np
from ultralytics import YOLO

class PlayerTracker:
    def __init__(self, model_path='yolov8x.pt'):
        print(f"Initializing PlayerTracker with model: {model_path}")
        self.model = YOLO(model_path)

    def choose_active_players(self, first_frame, player_detections, stub_path='tracker_stubs/active_players.pkl'):
        # Check if stub exists
        if stub_path and os.path.exists(stub_path):
            print(f"Loading active player Track IDs from stub: {stub_path}")
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        print("No active players stub found. Displaying Frame 0 to select Track IDs...")
        
        # Get frame 0 player detections
        frame_0_detections = player_detections[0] if len(player_detections) > 0 else {}
        
        # Create a copy of the first frame to draw annotations
        annotated_frame = first_frame.copy()
        
        for track_id, bbox in frame_0_detections.items():
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box (bright neon green BGR: (0, 255, 0))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Write Track ID above the box in bright neon red (BGR: (0, 0, 255))
            label = f"ID: {track_id}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the frame to the user
        cv2.imshow('Select Active Players', annotated_frame)
        cv2.waitKey(1)  # Refresh window rendering

        # Prompt user in terminal
        user_input = input('Enter the Track IDs of the active players you want to track, separated by commas (e.g. 1, 2): ')
        
        # Parse inputs
        active_track_ids = []
        try:
            active_track_ids = [int(x.strip()) for x in user_input.split(',') if x.strip()]
        except ValueError:
            print("Error parsing inputs. Defaulting to all detected Track IDs.")
            active_track_ids = list(frame_0_detections.keys())

        print(f"Tracking players with IDs: {active_track_ids}")
        
        # Close OpenCV window
        cv2.destroyWindow('Select Active Players')
        cv2.waitKey(1)

        # Save to stub
        if stub_path:
            stub_dir = os.path.dirname(stub_path)
            if stub_dir and not os.path.exists(stub_dir):
                os.makedirs(stub_dir)
            with open(stub_path, 'wb') as f:
                pickle.dump(active_track_ids, f)

        return active_track_ids

    def filter_by_track_ids(self, player_detections, active_track_ids):
        filtered_detections = []
        for frame_dict in player_detections:
            filtered_frame_dict = {}
            for track_id, bbox in frame_dict.items():
                if track_id in active_track_ids:
                    filtered_frame_dict[track_id] = bbox
            filtered_detections.append(filtered_frame_dict)
        return filtered_detections

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
