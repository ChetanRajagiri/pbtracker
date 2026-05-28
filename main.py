import os
import glob
import cv2
from trackers.player_tracker import PlayerTracker
from trackers.ball_tracker import BallTracker

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

    # Initialize PlayerTracker and BallTracker
    player_tracker = PlayerTracker(model_path='yolov8x.pt')
    ball_tracker = BallTracker(model_path='models/yolo5_last.pt')

    # Detect frames (and save to stub or read if already exists)
    player_stub_path = "tracker_stubs/player_detections.pkl"
    ball_stub_path = "tracker_stubs/ball_detections.pkl"
    
    # Read player tracking results from stub if present
    player_detections = player_tracker.detect_frames(
        frames, 
        read_from_stub=True, 
        stub_path=player_stub_path
    )

    # Read ball tracking results from stub if present
    ball_detections = ball_tracker.detect_frames(
        frames,
        read_from_stub=True,
        stub_path=ball_stub_path
    )

    # Draw annotations on frames (players first, then overlay ball)
    print("Drawing bounding boxes and track IDs...")
    annotated_frames = player_tracker.draw_bboxes(frames, player_detections)
    annotated_frames = ball_tracker.draw_bboxes(annotated_frames, ball_detections)

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
