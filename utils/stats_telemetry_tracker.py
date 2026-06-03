import os
import pickle
import numpy as np
import cv2
from court_line_detector.mini_court import MiniCourt

class StatsTelemetryTracker:
    def __init__(self, fps=30.0, court_keypoints_path='tracker_stubs/court_keypoints.pkl'):
        self.fps = fps
        self.mini_court = MiniCourt(court_keypoints_path=court_keypoints_path)
        
        # Player performance database initialization for core synchronized IDs
        self.player_stats = {
            1: {'total_shots': 0, 'total_distance_feet': 0.0, 'max_speed_mph': 0.0, 'prev_coords': None},
            2: {'total_shots': 0, 'total_distance_feet': 0.0, 'max_speed_mph': 0.0, 'prev_coords': None},
            3: {'total_shots': 0, 'total_distance_feet': 0.0, 'max_speed_mph': 0.0, 'prev_coords': None},
            4: {'total_shots': 0, 'total_distance_feet': 0.0, 'max_speed_mph': 0.0, 'prev_coords': None}
        }
        self.frame_stats_history = []  # Chronological snap-shot of stats for drawing back-references

    def process_match_telemetry(self, player_detections, ball_detections, ball_events):
        self.frame_stats_history = []
        
        # Reset stats
        for pid in self.player_stats:
            self.player_stats[pid] = {
                'total_shots': 0,
                'total_distance_feet': 0.0,
                'max_speed_mph': 0.0,
                'prev_coords': None
            }

        # Conversion constant: ft/s to mph is (3600 / 5280) = 0.681818
        mps_to_mph = 3600.0 / 5280.0
        time_per_frame = 1.0 / self.fps

        for idx in range(len(player_detections)):
            frame_players = player_detections[idx]
            frame_ball = ball_detections[idx] if idx < len(ball_detections) else {}
            
            # --- 1. Locomotion Analytics (Distance + Speed) ---
            for pid in self.player_stats:
                if pid in frame_players:
                    bbox = frame_players[pid].get('bbox')
                    if bbox:
                        # Ground contact point
                        foot_x = (bbox[0] + bbox[2]) / 2.0
                        foot_y = bbox[3]
                        
                        current_coords = self.mini_court.transform_point((foot_x, foot_y))
                        
                        if current_coords:
                            prev_coords = self.player_stats[pid]['prev_coords']
                            if prev_coords:
                                # Distance delta in feet
                                dist_delta = np.sqrt(
                                    (current_coords[0] - prev_coords[0])**2 + 
                                    (current_coords[1] - prev_coords[1])**2
                                )
                                
                                # Ignore unrealistic camera snaps/tracking jumps (sprinting threshold = 30 ft/s (~20mph))
                                if dist_delta / time_per_frame < 30.0:
                                    self.player_stats[pid]['total_distance_feet'] += dist_delta
                                    
                                    # Instantaneous Speed in MPH
                                    speed_fps = dist_delta / time_per_frame
                                    speed_mph = speed_fps * mps_to_mph
                                    
                                    if speed_mph > self.player_stats[pid]['max_speed_mph']:
                                        self.player_stats[pid]['max_speed_mph'] = speed_mph
                                        
                            self.player_stats[pid]['prev_coords'] = current_coords
                else:
                    # Player not detected in this frame, reset path anchor
                    self.player_stats[pid]['prev_coords'] = None

            # --- 2. Shot Attribution (Proximity-based check at HIT frames) ---
            if idx in ball_events and ball_events[idx] == "HIT":
                if 1 in frame_ball:
                    ball_bbox = frame_ball[1].get('bbox') if isinstance(frame_ball[1], dict) else frame_ball[1]
                    if ball_bbox:
                        ball_x = (ball_bbox[0] + ball_bbox[2]) / 2.0
                        ball_y = (ball_bbox[1] + ball_bbox[3]) / 2.0
                        
                        closest_pid = None
                        min_dist = float('inf')
                        
                        for pid in self.player_stats:
                            if pid in frame_players:
                                p_bbox = frame_players[pid].get('bbox')
                                if p_bbox:
                                    # Center coordinates of the player bounding box
                                    p_x = (p_bbox[0] + p_bbox[2]) / 2.0
                                    p_y = (p_bbox[1] + p_bbox[3]) / 2.0
                                    
                                    dist = np.sqrt((ball_x - p_x)**2 + (ball_y - p_y)**2)
                                    if dist < min_dist:
                                        min_dist = dist
                                        closest_pid = pid
                        
                        # Attribute to closest player if within reasonable proximity
                        if closest_pid is not None:
                            self.player_stats[closest_pid]['total_shots'] += 1

            # Append snapshot copy for historical frame reference drawing
            frame_snapshot = {
                pid: {
                    'total_shots': self.player_stats[pid]['total_shots'],
                    'total_distance_feet': self.player_stats[pid]['total_distance_feet'],
                    'max_speed_mph': self.player_stats[pid]['max_speed_mph']
                } for pid in self.player_stats
            }
            self.frame_stats_history.append(frame_snapshot)

    def draw_stats_hud(self, frame, current_frame_idx):
        if not self.frame_stats_history or current_frame_idx >= len(self.frame_stats_history):
            return frame
            
        stats_snapshot = self.frame_stats_history[current_frame_idx]
        
        # Dimensions and properties of the translucent HUD panel overlay (reduced size)
        hud_w, hud_h = 280, 125
        margin = 15
        
        # Position HUD at the bottom-left corner of the video frame
        h_frame, w_frame = frame.shape[:2]
        x1 = margin
        y1 = h_frame - hud_h - margin
        x2 = x1 + hud_w
        y2 = y1 + hud_h
        
        # Create separate overlay layer
        overlay = frame.copy()
        
        # Rounded or standard dark semi-transparent rectangle box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 15, 15), -1)
        
        # Alpha blend overlays: frame * (1 - alpha) + overlay * alpha
        alpha = 0.65
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Panel outline stroke
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        
        # Draw header text
        cv2.putText(frame, "PERFORMANCE TELEMETRY", (x1 + 12, y1 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)
        
        # Draw stats for each player (IDs 1-4)
        player_colors = {
            1: (255, 100, 100),  # Light Blue
            2: (100, 100, 255),  # Light Red
            3: (100, 255, 100),  # Light Green
            4: (100, 200, 255)   # Light Orange
        }
        
        y_offset = y1 + 42
        for pid in [1, 2, 3, 4]:
            p_data = stats_snapshot[pid]
            label = f"P{pid} | S: {p_data['total_shots']:<2} | D: {p_data['total_distance_feet']:>5.1f} ft | Max: {p_data['max_speed_mph']:>4.1f} mph"
            
            # Label outline dot key matching player circle colors
            color = player_colors.get(pid, (255, 255, 255))
            cv2.circle(frame, (x1 + 18, y_offset - 4), 3, color, -1)
            
            cv2.putText(frame, label, (x1 + 30, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (240, 240, 240), 1, cv2.LINE_AA)
            y_offset += 20
            
        return frame
