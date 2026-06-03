import os
import cv2
import numpy as np
from court_line_detector.mini_court import MiniCourt

class OfficiatingEngine:
    def __init__(self):
        self.line_calls = {}

    def get_line_calls(self, ball_events, ball_detections, mini_court):
        """
        Calculate automated IN/OUT line calls for each BOUNCE event.
        
        Parameters:
        -----------
        ball_events: dict {frame_idx: "BOUNCE" or "HIT"}
        ball_detections: list of dicts (each containing coordinate bbox details)
        mini_court: MiniCourt instance with loaded homography matrix
        """
        self.line_calls = {}
        
        for frame_idx, event_type in ball_events.items():
            if event_type == "BOUNCE":
                if frame_idx < len(ball_detections):
                    frame_ball = ball_detections[frame_idx]
                    if 1 in frame_ball:
                        bbox = frame_ball[1].get('bbox') if isinstance(frame_ball[1], dict) else frame_ball[1]
                        if bbox:
                            # Bottom-center coordinate (footprint contact) of the ball bounding box
                            ball_x = (bbox[0] + bbox[2]) / 2.0
                            ball_y = bbox[3] # Bottom-most point representing court contact
                            
                            # Perspective projection to real-world feet coordinates
                            transformed = mini_court.transform_point((ball_x, ball_y))
                            if transformed:
                                x_ft, y_ft = transformed
                                
                                # Line Call boundary conditions:
                                # Court size is total length = 44 feet, total width = 20 feet
                                # Coordinate origin (0, 0) at the bottom-left baseline corner
                                is_in = (0.0 <= x_ft <= 20.0) and (0.0 <= y_ft <= 44.0)
                                call = "IN" if is_in else "OUT"
                                
                                self.line_calls[frame_idx] = {
                                    "call": call,
                                    "coords": (x_ft, y_ft)
                                }
                                print(f"[OFFICIATING] Bounce at frame {frame_idx} -> Projected: ({x_ft:.2f} ft, {y_ft:.2f} ft) -> Call: {call}")
        return self.line_calls

    def draw_calls_on_video(self, video_frames, duration_frames=25):
        """
        Draw a broadcast-style officiating line call banner overlay on the video frames.
        
        Parameters:
        -----------
        video_frames: list of numpy arrays (BGR frames)
        duration_frames: int (frames to retain the IN/OUT message on the screen)
        """
        annotated_frames = []
        n_frames = len(video_frames)
        
        # Loop through each frame and check if a call banner is active
        for i, frame in enumerate(video_frames):
            annotated_frame = frame.copy()
            
            # Find the most recent line call that is still within the display duration
            active_call = None
            active_age = 0
            
            # Search backward from current frame index up to the duration limit
            for offset in range(duration_frames):
                target_frame = i - offset
                if target_frame in self.line_calls:
                    active_call = self.line_calls[target_frame]
                    active_age = offset
                    break
            
            if active_call:
                call_text = active_call["call"]
                color = (0, 255, 0) if call_text == "IN" else (0, 0, 255) # Green for IN, Red for OUT
                
                # Render broadcast-style display center overlay banner
                height, width = annotated_frame.shape[:2]
                
                # Draw a dark backing banner strip across center screen
                banner_y1 = int(height * 0.42)
                banner_y2 = int(height * 0.58)
                
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (0, banner_y1), (width, banner_y2), (15, 15, 15), -1)
                
                # Blend banner strip background
                alpha = 0.70
                cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
                
                # Write "IN" or "OUT" text centered
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 3.0
                thickness = 6
                (text_w, text_h), _ = cv2.getTextSize(call_text, font, font_scale, thickness)
                
                text_x = (width - text_w) // 2
                text_y = (height + text_h) // 2
                
                # Draw text dropshadow
                cv2.putText(annotated_frame, call_text, (text_x + 3, text_y + 3), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, call_text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
                
                # Draw helper sublabel with details
                x_ft, y_ft = active_call["coords"]
                detail_text = f"Bounce Location: {x_ft:.1f} ft, {y_ft:.1f} ft"
                (det_w, det_h), _ = cv2.getTextSize(detail_text, font, 0.6, 2)
                cv2.putText(annotated_frame, detail_text, ((width - det_w) // 2, text_y + 40), font, 0.6, (240, 240, 240), 2, cv2.LINE_AA)
            
            annotated_frames.append(annotated_frame)
            
        return annotated_frames
