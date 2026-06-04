import os
import glob
import cv2
import pickle
import argparse
from trackers.player_tracker import PlayerTracker
from trackers.ball_tracker import BallTracker
from court_line_detector.manual_court_selector import ManualCourtDetector
from court_line_detector.mini_court import MiniCourt
from utils.stats_telemetry_tracker import StatsTelemetryTracker
from utils.officiating_engine import OfficiatingEngine
from utils.auto_player_filter import AutoPlayerFilter
from utils.ball_physics_analyzer import BallPhysicsAnalyzer

def main():
    # Set up argument parsing for dynamic video input
    parser = argparse.ArgumentParser(description="Dynamic Pickleball Tracking and Telemetry Pipeline")
    parser.add_argument("--video", default="input_videos/newclip.mp4", help="Path to the input video file")
    args = parser.parse_args()
    
    video_path = args.video
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")
        return
        
    print(f"Processing video: {video_path}")
    
    # Generate base name for dynamic cache stub file naming
    video_base = os.path.splitext(os.path.basename(video_path))[0]
    
    # Stub paths
    player_stub_path = "tracker_stubs/player_detections.pkl"
    ball_stub_path = "tracker_stubs/ball_detections.pkl"
    court_stub_path = "tracker_stubs/court_keypoints.pkl"
    ball_events_stub_path = "tracker_stubs/ball_events.pkl"

    # Open Video Capture to get dimensions & FPS
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read the first frame for manual court selector keypoint collection
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read first frame of video.")
        return

    # Initialize Trackers, Court Detector
    player_tracker = PlayerTracker(model_path='yolov8x.pt')
    ball_tracker = BallTracker(model_path='models/yolo5_last.pt')
    court_detector = ManualCourtDetector()
    
    # Get court keypoints first (so it exists for the tracking loop boundary filtering)
    court_keypoints = court_detector.get_keypoints(first_frame, stub_path=court_stub_path)

    # Initialize MiniCourt Radar
    mini_court = MiniCourt(court_keypoints_path=court_stub_path)

    # 1. Load or run player detections in memory-efficient stream loop
    need_filtration = False
    if os.path.exists(player_stub_path):
        print(f"Loading player tracking data from stub: {player_stub_path}")
        with open(player_stub_path, 'rb') as f:
            player_detections = pickle.load(f)
        
        # Check if loaded data is raw (contains track IDs other than 1, 2, 3, 4)
        unique_ids = set()
        for frame_dict in player_detections:
            unique_ids.update(frame_dict.keys())
        if any(uid > 4 for uid in unique_ids):
            print("[PIPELINE] Loaded player detections stub contains raw tracks. Forcing player filtration...")
            need_filtration = True
    else:
        print("Player detections cache not found. Running tracking over video stream...")
        player_detections = []
        cap = cv2.VideoCapture(video_path)
        from tqdm import tqdm
        pbar = tqdm(total=total_frames, desc="Tracking players", unit="frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # detect_frames on a single-frame list
            detection_dict = player_tracker.detect_frames([frame], read_from_stub=False, court_keypoints_path=court_stub_path, verbose=False)[0]
            player_detections.append(detection_dict)
            pbar.update(1)
        pbar.close()
        cap.release()
        
        # Save to stub
        os.makedirs(os.path.dirname(player_stub_path), exist_ok=True)
        with open(player_stub_path, 'wb') as f:
            pickle.dump(player_detections, f)
        print(f"Successfully saved raw player tracking to stub: {player_stub_path}")
        need_filtration = True
        
    if need_filtration:
        # Run AutoPlayerFilter to clean the detections
        print("Applying automatic player isolation and spectator purging...")
        player_filter = AutoPlayerFilter(detections_pkl=player_stub_path, court_pkl=court_stub_path)
        player_detections = player_filter.run_filtration()

    # 2. Load or run ball detections in memory-efficient stream loop
    if os.path.exists(ball_stub_path):
        print(f"Loading ball tracking data from stub: {ball_stub_path}")
        with open(ball_stub_path, 'rb') as f:
            ball_detections = pickle.load(f)
    else:
        print("Ball detections cache not found. Running ball detection over video stream...")
        ball_detections = []
        cap = cv2.VideoCapture(video_path)
        from tqdm import tqdm
        pbar = tqdm(total=total_frames, desc="Tracking ball", unit="frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # detect_frames on a single-frame list
            detection_dict = ball_tracker.detect_frames([frame], read_from_stub=False, verbose=False)[0]
            ball_detections.append(detection_dict)
            pbar.update(1)
        pbar.close()
        cap.release()
        
        # Save to stub
        os.makedirs(os.path.dirname(ball_stub_path), exist_ok=True)
        with open(ball_stub_path, 'wb') as f:
            pickle.dump(ball_detections, f)
        print(f"Successfully saved ball tracking to stub: {ball_stub_path}")

    # Interpolate ball positions to resolve flickering
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # 3. Load or compute ball physics events
    if os.path.exists(ball_events_stub_path):
        print(f"Loading ball physics events from: {ball_events_stub_path}")
        with open(ball_events_stub_path, 'rb') as f:
            ball_events = pickle.load(f)
    else:
        print("Ball physics events cache not found. Running BallPhysicsAnalyzer...")
        physics_analyzer = BallPhysicsAnalyzer(pkl_path=ball_stub_path)
        physics_analyzer.detect_events()
        physics_analyzer.save_events(output_path=ball_events_stub_path)
        ball_events = physics_analyzer.events

    # Initialize and process Stats Telemetry
    telemetry_tracker = StatsTelemetryTracker(fps=fps, court_keypoints_path=court_stub_path)
    print("Processing match performance telemetry and running shot attributions...")
    telemetry_tracker.process_match_telemetry(player_detections, ball_detections, ball_events)

    # Initialize and compute Officiating Line Calls
    officiating_engine = OfficiatingEngine()
    print("Calculating automated officiating line calls...")
    officiating_engine.get_line_calls(ball_events, ball_detections, mini_court)

    # Define VideoWriter to write frames sequentially as they are annotated (memory footprint constant)
    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "player_tracking_test.mp4")
    print(f"Streaming final annotations directly to output video: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    from tqdm import tqdm
    pbar = tqdm(total=total_frames, desc="Rendering annotated video", unit="frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw bounding boxes (players first, then ball, then court lines)
        frame_players_det = [player_detections[frame_idx]] if frame_idx < len(player_detections) else [{}]
        frame_ball_det = [ball_detections[frame_idx]] if frame_idx < len(ball_detections) else [{}]
        
        annotated_frame = player_tracker.draw_bboxes([frame], frame_players_det)[0]
        annotated_frame = ball_tracker.draw_bboxes([annotated_frame], frame_ball_det)[0]
        
        if court_keypoints:
            annotated_frame = court_detector.draw_keypoints_on_video([annotated_frame], court_keypoints)[0]
            
        # Overlay performance HUD dashboard
        annotated_frame = telemetry_tracker.draw_stats_hud(annotated_frame, frame_idx)

        # Render and overlay the top-down radar canvas
        frame_players = player_detections[frame_idx] if frame_idx < len(player_detections) else {}
        frame_ball = ball_detections[frame_idx] if frame_idx < len(ball_detections) else {}
        
        line_calls = officiating_engine.line_calls
        radar_canvas = mini_court.draw_points_on_mini_court(frame_players, frame_ball, line_calls, frame_idx)
        
        # Dynamically scale radar to fit frame height (e.g., 45% of video height)
        target_height = int(annotated_frame.shape[0] * 0.45)
        aspect_ratio = radar_canvas.shape[1] / radar_canvas.shape[0]
        target_width = int(target_height * aspect_ratio)
        
        radar_canvas_resized = cv2.resize(radar_canvas, (target_width, target_height))
        radar_h, radar_w = radar_canvas_resized.shape[:2]
        
        margin = 15
        start_y = annotated_frame.shape[0] - radar_h - margin
        start_x = annotated_frame.shape[1] - radar_w - margin
        
        if start_y >= 0 and start_x >= 0:
            annotated_frame[start_y:start_y+radar_h, start_x:start_x+radar_w] = radar_canvas_resized

        # Draw frame counter
        label = f"Frame {frame_idx} / {total_frames}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.6, 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        x = annotated_frame.shape[1] - tw - 15
        y = 30
        cv2.rectangle(annotated_frame, (x - 8, y - th - 8), (x + tw + 8, y + 8), (0, 0, 0), -1)
        cv2.putText(annotated_frame, label, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Write frame to final video
        out.write(annotated_frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    print(f"Saved annotated video to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
