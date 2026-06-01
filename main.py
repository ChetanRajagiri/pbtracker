import os
import glob
import cv2
from trackers.player_tracker import PlayerTracker
from trackers.ball_tracker import BallTracker
from court_line_detector.manual_court_selector import ManualCourtDetector

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

    video_path = video_files[0]
    print(f"Reading video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()

    print(f"Successfully read {len(frames)} frames from the video.")

    # Initialize Trackers and Court Detector
    player_tracker = PlayerTracker(model_path='yolov8x.pt')
    ball_tracker = BallTracker(model_path='models/yolo5_last.pt')
    court_detector = ManualCourtDetector()

    # Detect frames (and save to stub or read if already exists)
    player_stub_path = "tracker_stubs/player_detections.pkl"
    ball_stub_path = "tracker_stubs/ball_detections.pkl"
    court_stub_path = "tracker_stubs/court_keypoints.pkl"
    
    # Read player tracking results from stub if present
    player_detections = player_tracker.detect_frames(
        frames, 
        read_from_stub=False, 
        stub_path=player_stub_path
    )

    # Read ball tracking results from stub if present
    ball_detections = ball_tracker.detect_frames(
        frames,
        read_from_stub=False,
        stub_path=ball_stub_path
    )

    # Interpolate ball positions to resolve flickering
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Get court keypoints (GUI click if stub not present)
    court_keypoints = court_detector.get_keypoints(frames[0], stub_path=court_stub_path)

    # Choose active players manually or load from stub
    active_track_ids = player_tracker.choose_active_players(frames[0], player_detections)

    # Filter detections to keep only active players
    player_detections = player_tracker.filter_by_track_ids(player_detections, active_track_ids)

    # Draw annotations on frames (players first, then ball, then court lines)
    print("Drawing bounding boxes, track IDs, and court keypoints...")
    annotated_frames = player_tracker.draw_bboxes(frames, player_detections)
    annotated_frames = ball_tracker.draw_bboxes(annotated_frames, ball_detections)
    if court_keypoints:
        annotated_frames = court_detector.draw_keypoints_on_video(annotated_frames, court_keypoints)

    # Save to output_videos/player_tracking_test.mp4
    output_dir = "output_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "player_tracking_test.mp4")
    print(f"Saving output video to {output_path}...")
    
    # Define Codec and VideoWriter (mp4v is widely supported)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in annotated_frames:
        out.write(frame)
    out.release()
    print(f"Saved annotated video to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
