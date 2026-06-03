import cv2
import numpy as np
import pickle
import os

class MiniCourt:
    def __init__(self, canvas_width=400, canvas_height=800, padding=40, court_keypoints_path='tracker_stubs/court_keypoints.pkl'):
        # Canvas dimensions and padding
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.padding = padding
        
        # Official Pickleball dimensions in feet
        self.court_length_ft = 44.0
        self.court_width_ft = 20.0
        self.kitchen_length_ft = 7.0
        self.half_court_length_ft = 22.0
        
        # Dimensions inside padding for top-down representation
        self.drawing_width = canvas_width - 2 * padding
        self.drawing_height = canvas_height - 2 * padding
        
        # Transformation matrix initialization
        self.homography_matrix = None
        self.load_homography_matrix(court_keypoints_path)
        
    def load_homography_matrix(self, court_keypoints_path):
        if not os.path.exists(court_keypoints_path):
            print(f"Warning: Court keypoint stub {court_keypoints_path} not found. Cannot calculate homography.")
            return

        with open(court_keypoints_path, 'rb') as f:
            src_pts = pickle.load(f)
            
        if len(src_pts) != 12:
            print(f"Warning: Expected 12 court keypoints, found {len(src_pts)}. Cannot calculate homography.")
            return
            
        # Convert source points to float32 NumPy array
        src_pts = np.array(src_pts, dtype=np.float32)
        
        # Destination real-world coordinates in feet, matching 12-point keypoint layout
        # Manual selection order sequence (Far to Near baseline intersections):
        # 1-3: Far Baseline (0, 44), (10, 44), (20, 44)
        # 4-6: Far Kitchen line (0, 29), (10, 29), (20, 29)
        # 7-9: Near Kitchen line (0, 15), (10, 15), (20, 15)
        # 10-12: Near Baseline (0, 0), (10, 0), (20, 0)
        dst_pts = np.array([
            [0, 44],   # 1
            [10, 44],  # 2
            [20, 44],  # 3
            [0, 29],   # 4
            [10, 29],  # 5
            [20, 29],  # 6
            [0, 15],   # 7
            [10, 15],  # 8
            [20, 15],  # 9
            [0, 0],    # 10
            [10, 0],   # 11
            [20, 0]    # 12
        ], dtype=np.float32)
        
        self.homography_matrix, _ = cv2.findHomography(src_pts, dst_pts)
        print("Successfully calculated homography matrix.")

    def ft_to_pixels(self, x_ft, y_ft):
        # Maps coordinates in feet (x: 0 to 20, y: 0 to 44) to drawing canvas pixels
        # Top baseline (y = 44) maps to canvas top padding
        # Bottom baseline (y = 0) maps to canvas bottom padding
        # Left sideline (x = 0) maps to canvas left padding
        # Right sideline (x = 20) maps to canvas right padding
        px_x = int(self.padding + (x_ft / self.court_width_ft) * self.drawing_width)
        px_y = int(self.canvas_height - self.padding - (y_ft / self.court_length_ft) * self.drawing_height)
        return (px_x, px_y)

    def draw_mini_court_base(self):
        # Create dark background canvas
        canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        # Dark gray/charcoal background
        canvas[:] = (30, 30, 30)
        
        # Color and line thickness configuration
        line_color = (255, 255, 255) # White
        thickness = 2
        
        # Draw perimeters
        cv2.rectangle(canvas, self.ft_to_pixels(0, 0), self.ft_to_pixels(20, 44), line_color, thickness)
        
        # Draw Net Line (at 22 ft)
        cv2.line(canvas, self.ft_to_pixels(0, 22), self.ft_to_pixels(20, 22), (0, 255, 255), thickness + 1) # Yellow for net
        
        # Draw Far Kitchen Line (at 29 ft)
        cv2.line(canvas, self.ft_to_pixels(0, 29), self.ft_to_pixels(20, 29), line_color, thickness)
        
        # Draw Near Kitchen Line (at 15 ft)
        cv2.line(canvas, self.ft_to_pixels(0, 15), self.ft_to_pixels(20, 15), line_color, thickness)
        
        # Draw Center Line for Far court (y: 29 to 44)
        cv2.line(canvas, self.ft_to_pixels(10, 29), self.ft_to_pixels(10, 44), line_color, thickness)
        
        # Draw Center Line for Near court (y: 0 to 15)
        cv2.line(canvas, self.ft_to_pixels(10, 0), self.ft_to_pixels(10, 15), line_color, thickness)
        
        return canvas

    def transform_point(self, point):
        # Point is (x, y) in frame coordinates
        if self.homography_matrix is None:
            return None
            
        pt_array = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt_array, self.homography_matrix)
        
        # Transformed coordinates in real-world feet (x, y)
        x_ft, y_ft = transformed[0][0][0], transformed[0][0][1]
        return x_ft, y_ft

    def draw_points_on_mini_court(self, player_detections, ball_detections_frame, line_calls=None, current_frame_idx=None):
        # Initialize standard mini-court panel layout
        canvas = self.draw_mini_court_base()
        
        # Player indicators (represented as colored filled circles)
        player_colors = {
            1: (255, 0, 0),    # Player 1: Blue
            2: (0, 0, 255),    # Player 2: Red
            3: (0, 255, 0),    # Player 3: Green
            4: (0, 165, 255)   # Player 4: Orange
        }
        
        # Draw players
        for track_id, player_data in player_detections.items():
            if track_id not in player_colors:
                continue
            
            bbox = player_data.get('bbox')
            if bbox is not None:
                # Find bottom-center of the player's bounding box (foot placement)
                foot_x = (bbox[0] + bbox[2]) / 2.0
                foot_y = bbox[3]
                
                transformed = self.transform_point((foot_x, foot_y))
                if transformed is not None:
                    x_ft, y_ft = transformed
                    # Clip coordinates slightly if they are slightly out of bounds to keep visual clean
                    x_ft = np.clip(x_ft, 0, self.court_width_ft)
                    y_ft = np.clip(y_ft, 0, self.court_length_ft)
                    
                    pixel_coords = self.ft_to_pixels(x_ft, y_ft)
                    
                    # Draw player circle marker
                    cv2.circle(canvas, pixel_coords, 8, player_colors[track_id], -1)
                    # Text/ID label overlay
                    cv2.putText(canvas, f"P{track_id}", (pixel_coords[0] - 12, pixel_coords[1] - 12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw ball if detected
        if ball_detections_frame:
            bbox = ball_detections_frame.get('bbox')
            if bbox:
                # Bottom center or center of the ball bounding box
                ball_x = (bbox[0] + bbox[2]) / 2.0
                ball_y = (bbox[1] + bbox[3]) / 2.0
                
                transformed = self.transform_point((ball_x, ball_y))
                if transformed is not None:
                    x_ft, y_ft = transformed
                    x_ft = np.clip(x_ft, 0, self.court_width_ft)
                    y_ft = np.clip(y_ft, 0, self.court_length_ft)
                    
                    pixel_coords = self.ft_to_pixels(x_ft, y_ft)
                    
                    # Determine color: green for detected, pink for interpolated
                    source = ball_detections_frame.get('source', 'detected')
                    ball_color = (0, 255, 0) if source == 'detected' else (203, 192, 255) # Pink (BGR: (255, 192, 203))
                    
                    cv2.circle(canvas, pixel_coords, 6, ball_color, -1)

        # Draw recent officiating bounce locations on the mini-court radar
        if line_calls is not None and current_frame_idx is not None:
            # Lookback window: check last 30 frames for any bounce event
            for offset in range(30):
                target_frame = current_frame_idx - offset
                if target_frame in line_calls:
                    bounce_info = line_calls[target_frame]
                    call_type = bounce_info["call"]
                    x_ft, y_ft = bounce_info["coords"]
                    
                    # Map real-world coordinates to local minimap pixels
                    pixel_coords = self.ft_to_pixels(x_ft, y_ft)
                    
                    # Color check: bright Green for IN, bright Red for OUT
                    bounce_color = (0, 255, 0) if call_type == "IN" else (0, 0, 255)
                    
                    # Render bounce spot indicator dot
                    cv2.circle(canvas, pixel_coords, 7, bounce_color, -1)
                    cv2.circle(canvas, pixel_coords, 9, (255, 255, 255), 1) # White outline
                    
                    # Render small offset call type text
                    text_offset_x = 10 if pixel_coords[0] < self.canvas_width - 40 else -35
                    text_offset_y = 5
                    cv2.putText(canvas, call_type, 
                                (pixel_coords[0] + text_offset_x, pixel_coords[1] + text_offset_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, bounce_color, 2, cv2.LINE_AA)
                    
                    # Only draw the most recent active bounce to avoid overlapping clutter
                    break
                    
        return canvas
