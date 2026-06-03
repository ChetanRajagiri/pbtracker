import os
import glob
import cv2
import pickle
from trackers.player_tracker import PlayerTracker
from trackers.ball_tracker import BallTracker
from court_line_detector.manual_court_selector import ManualCourtDetector
from court_line_detector.mini_court import MiniCourt
from utils.stats_telemetry_tracker import StatsTelemetryTracker
from utils.officiating_engine import OfficiatingEngine

def main():
    # Find sample video
    input_dir = "input_videos"
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not video_files:
        print(f"No videos found in '{input_dir}'. Please add a tennis video clip first.")
        return

    video_path = "input_videos/newclip.mp4"
    print(f"Reading video: {video_path}")

    # 1. Open Video Capture to get dimensions & FPS
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

    # Initialize Trackers, Court Detector, and Mini Court Radar
    player_tracker = PlayerTracker(model_path='yolov8x.pt')
    ball_tracker = BallTracker(model_path='models/yolo5_last.pt')
    court_detector = ManualCourtDetector()

    # Stub paths
    player_stub_path = "tracker_stubs/player_detections.pkl"
    ball_stub_path = "tracker_stubs/ball_detections.pkl"
    court_stub_path = "tracker_stubs/court_keypoints.pkl"
    ball_events_stub_path = "tracker_stubs/ball_events.pkl"
    
    # Get court keypoints first (so it exists for the tracking loop boundary filtering)
    court_keypoints = court_detector.get_keypoints(first_frame, stub_path=court_stub_path)

    # Initialize MiniCourt Radar
    mini_court = MiniCourt(court_keypoints_path=court_stub_path)

    # Load or run player detections in memory-efficient stream loop if cache does not exist
    if os.path.exists(player_stub_path):
        print(f"Loading player tracking data from stub: {player_stub_path}")
        with open(player_stub_path, 'rb') as f:
            player_detections = pickle.load(f)
    else:
        print("Player detections cache not found. Running tracking over video stream...")
        player_detections = []
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # detect_frames on a single-frame list
            detection_dict = player_tracker.detect_frames([frame], read_from_stub=False)[0]
            player_detections.append(detection_dict)
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Tracked players on {frame_idx}/{total_frames} frames...")
        cap.release()
        
        # Save to stub
        os.makedirs(os.path.dirname(player_stub_path), exist_ok=True)
        with open(player_stub_path, 'wb') as f:
            pickle.dump(player_detections, f)
        print(f"Successfully saved player tracking to stub: {player_stub_path}")

    # Load or run ball detections in memory-efficient stream loop if cache does not exist
    if os.path.exists(ball_stub_path):
        print(f"Loading ball tracking data from stub: {ball_stub_path}")
        with open(ball_stub_path, 'rb') as f:
            ball_detections = pickle.load(f)
    else:
        print("Ball detections cache not found. Running ball detection over video stream...")
        ball_detections = []
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # detect_frames on a single-frame list
            detection_dict = ball_tracker.detect_frames([frame], read_from_stub=False)[0]
            ball_detections.append(detection_dict)
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Detected ball on {frame_idx}/{total_frames} frames...")
        cap.release()
        
        # Save to stub
        os.makedirs(os.path.dirname(ball_stub_path), exist_ok=True)
        with open(ball_stub_path, 'wb') as f:
            pickle.dump(ball_detections, f)
        print(f"Successfully saved ball tracking to stub: {ball_stub_path}")

    # Interpolate ball positions to resolve flickering
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Load ball events for telemetry processing (default to empty dict if not found)
    ball_events = {}
    if os.path.exists(ball_events_stub_path):
        print(f"Loading ball physics events from: {ball_events_stub_path}")
        with open(ball_events_stub_path, 'rb') as f:
            ball_events = pickle.load(f)
    else:
        print("[WARNING] No ball events stub found. Skipping shot attribution. Run utils/ball_physics_analyzer.py first.")

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
        
        if frame_idx % 100 == 0:
            print(f"Rendered and saved annotated frames: {frame_idx}/{total_frames}...")

    cap.release()
    out.release()
    print(f"Saved annotated video to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
