import os
import pickle
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from boxmot.trackers.tracker_zoo import create_tracker

def expand_polygon_asymmetric(polygon, frame_height=1080):
    if polygon is None:
        return None
    pts = polygon.reshape(-1, 2).astype(np.float32)
    centroid = np.mean(pts, axis=0)
    expanded_pts = []
    for pt in pts:
        y_val = pt[1]
        # Classify vertex: near baseline gets 80px, far baseline gets 30px, others get 40px
        if y_val > frame_height * 0.55:
            dist = 80.0
        else:
            dist = 30.0
        
        direction = pt - centroid
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            unit_direction = direction / norm
            new_pt = pt + dist * unit_direction
        else:
            new_pt = pt
        expanded_pts.append(new_pt)
    return np.array(expanded_pts, dtype=np.int32)


class PlayerTracker:
    def __init__(self, model_path='yolov8x.pt'):
        print(f"Initializing PlayerTracker with YOLO detector: {model_path}")
        self.model = YOLO(model_path)
        
        # Determine tracking acceleration device dynamically
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing BoxMOT BoT-SORT tracker with OSNet-x0_25 ReID weights on device: {device}")
        
        # Instantiate BoT-SORT with OSNet ReID weights and customized track buffer
        self.tracker = create_tracker(
            tracker_type='botsort',
            reid_weights='osnet_x0_25_msmt17.pt',
            device=device,
            half=False,
            tracker_backend='python',
            evolve_param_dict={'track_buffer': 90}
        )
        self.frame_idx = 0

    def detect_frames(self, frames, read_from_stub=False, stub_path=None, court_keypoints_path="tracker_stubs/court_keypoints.pkl", verbose=True):
        # 1. Load from stub if requested and exists (with frame count validation)
        if read_from_stub and stub_path and os.path.exists(stub_path):
            if verbose:
                print(f"Loading player tracking data from stub: {stub_path}")
            with open(stub_path, 'rb') as f:
                cached_data = pickle.load(f)
                if len(cached_data) == len(frames):
                    return cached_data
                else:
                    if verbose:
                        print(f"[WARNING] Cache mismatch detected. Stale tracking stub contains fewer frames than the active input video. Forcing full YOLO re-inference...")

        # Load court keypoints to define boundaries for initial gallery seeding
        polygon = None
        if court_keypoints_path and os.path.exists(court_keypoints_path):
            if verbose:
                print(f"Loading court keypoints for boundary filtering: {court_keypoints_path}")
            with open(court_keypoints_path, 'rb') as f:
                court_keypoints = pickle.load(f)
            if len(court_keypoints) >= 12:
                corner1 = court_keypoints[0]   # Top-Left Far Baseline
                corner2 = court_keypoints[2]   # Top-Right Far Baseline
                corner3 = court_keypoints[11]  # Bottom-Right Near Baseline
                corner4 = court_keypoints[9]   # Bottom-Left Near Baseline
                polygon = np.array([corner1, corner2, corner3, corner4], dtype=np.int32)
                
                # Determine frame_height for asymmetric polygon expansion
                frame_height = 1080
                if len(frames) > 0 and hasattr(frames[0], 'shape'):
                    frame_height = frames[0].shape[0]
                polygon = expand_polygon_asymmetric(polygon, frame_height)

            else:
                if verbose:
                    print(f"[WARNING] Insufficient court keypoints ({len(court_keypoints)}) to define polygon. Skipping filtering.")
        else:
            if verbose:
                print(f"[WARNING] No court keypoints found at {court_keypoints_path}. Skipping boundary filtering.")

        # Reset logic for frame-by-frame or batch tracking
        if len(frames) > 1:
            # Batch processing: reset tracker and frame index
            self.tracker.reset()
            self.frame_idx = 0
        elif self.frame_idx == 0:
            # First frame of streaming mode: reset tracker
            self.tracker.reset()

        player_detections = []

        for frame in frames:
            # Run YOLO prediction for person class (class 0)
            results = self.model.predict(frame, classes=[0], verbose=False)[0]
            
            # Format detections to NumPy array for BoxMOT: [x1, y1, x2, y2, confidence, class]
            dets = []
            for box in results.boxes:
                coords = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                dets.append(coords + [conf, cls])
            dets = np.array(dets) if dets else np.empty((0, 6))
            
            # Update BoT-SORT tracker
            tracks = self.tracker.update(dets, frame)
            
            frame_dict = {}
            # Extract tracking output: [x1, y1, x2, y2, track_id, conf, cls, ind]
            for t in tracks:
                bbox = t[:4].tolist()
                tid = int(t[4])
                
                # Check court polygon boundary filter using 3-point feet check
                is_on_court = True
                if polygon is not None:
                    pt1 = (bbox[0], bbox[3])
                    pt2 = ((bbox[0] + bbox[2]) / 2.0, bbox[3])
                    pt3 = (bbox[2], bbox[3])
                    is_inside = (
                        cv2.pointPolygonTest(polygon, pt1, False) >= 0 or
                        cv2.pointPolygonTest(polygon, pt2, False) >= 0 or
                        cv2.pointPolygonTest(polygon, pt3, False) >= 0
                    )
                    is_on_court = True if is_inside else False

                
                if not is_on_court:
                    # Bug 3: assign track ID 0, skip embedding extraction, do not match
                    # We map out-of-court detections to unique negative raw keys to avoid collisions,
                    # which will be filtered out as track ID 0 by downstream processors.
                    frame_dict[-tid] = {
                        'bbox': bbox,
                        'embedding': None,
                        'is_on_court': False
                    }
                else:
                    # Extract ReID embedding crop and run OSNet
                    emb = self.tracker.model.get_features(np.array([bbox]), frame)[0]
                    frame_dict[tid] = {
                        'bbox': bbox,
                        'embedding': emb,
                        'is_on_court': True
                    }
            
            player_detections.append(frame_dict)
            self.frame_idx += 1

        # Save to stub only if batch processing (if a stub path is provided and we processed all frames)
        if len(frames) > 1 and stub_path:
            stub_dir = os.path.dirname(stub_path)
            if stub_dir and not os.path.exists(stub_dir):
                os.makedirs(stub_dir)
            if verbose:
                print(f"Saving player tracking data to stub: {stub_path}")
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def draw_bboxes(self, video_frames, player_detections):
        annotated_frames = []
        for i, frame in enumerate(video_frames):
            annotated_frame = frame.copy()
            detections = player_detections[i] if i < len(player_detections) else {}
            
            for track_id, data in detections.items():
                if track_id not in [1, 2, 3, 4]:
                    continue
                    
                bbox = data['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                # Assign premium custom color maps matching the shared design:
                color_map = {
                    1: (0, 95, 255),    # Player 1: Orange-Red
                    2: (0, 215, 255),   # Player 2: Gold-Yellow
                    3: (255, 191, 0),   # Player 3: Light Cyan/Blue
                    4: (255, 50, 50)    # Player 4: Royal Blue
                }
                color = color_map.get(track_id, (0, 255, 0))
                
                # 1. Draw perspective-aligned ground ellipse around the player's feet
                center_x = int((x1 + x2) / 2)
                center_y = int(y2)
                # Scale width based on bounding box width, compress height for perspective
                axes_width = int((x2 - x1) * 0.7)
                axes_height = int(axes_width * 0.3)
                
                # Transparent filled shadow for the ring
                overlay = annotated_frame.copy()
                cv2.ellipse(overlay, (center_x, center_y), (axes_width, axes_height), 0, 0, 360, color, -1, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.25, annotated_frame, 0.75, 0, annotated_frame)
                
                # Draw the outline of the ring
                cv2.ellipse(annotated_frame, (center_x, center_y), (axes_width, axes_height), 0, 0, 360, color, 3, cv2.LINE_AA)
                
                # 2. Draw a clean, premium name tag pill centered above the player's head
                label = f"Player {track_id}"
                font_scale = 0.45
                font_thickness = 1
                (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                
                # Determine tag positioning above head
                tag_y = max(y1 - 12, h + 15)
                # Outer pill rectangle
                cv2.rectangle(annotated_frame, 
                              (center_x - w//2 - 8, tag_y - h - 6), 
                              (center_x + w//2 + 8, tag_y + 4), 
                              color, -1, cv2.LINE_AA)
                # Text overlay inside pill
                cv2.putText(annotated_frame, label, 
                            (center_x - w//2, tag_y - 1), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
            
            annotated_frames.append(annotated_frame)
            
        return annotated_frames
