import os
import glob
import cv2
import pickle
import argparse

def main():
    # 1. Parse arguments for video selection
    parser = argparse.ArgumentParser(description="Generate raw ID diagnostic frame")
    parser.add_argument("--video", help="Path to the input video file")
    args = parser.parse_args()
    
    video_path = args.video
    
    # Fallback to scanning if video path not specified
    if not video_path:
        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join("input_videos", ext)))
            
        if not video_files:
            print("[ERROR] No video files found in 'input_videos/' directory. Please specify with --video <path>.")
            return
        video_path = video_files[0]
        
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file does not exist at: {video_path}")
        return
        
    print(f"[DIAGNOSTIC] Loading video from: {video_path}")
    
    # 2. Load the raw tracking data
    player_stub_path = "tracker_stubs/player_detections.pkl"
    if not os.path.exists(player_stub_path):
        print(f"[ERROR] Player detections stub not found at: {player_stub_path}")
        return
        
    with open(player_stub_path, 'rb') as f:
        player_detections = pickle.load(f)
        
    # 3. Find the first frame where at least 4 player detections exist (starting from frame 45)
    # Fallback progressively to 3, 2, or 1 if 4 players are never detected in a single frame
    first_detected_frame_idx = -1
    start_frame = 45 if len(player_detections) > 45 else 0
    for min_players in [4, 3, 2, 1]:
        for idx in range(start_frame, len(player_detections)):
            detections = player_detections[idx]
            if len(detections) >= min_players:
                first_detected_frame_idx = idx
                break
        if first_detected_frame_idx != -1:
            print(f"[DIAGNOSTIC] Found frame with at least {min_players} player detections (starting from frame {start_frame}) at index: {first_detected_frame_idx}")
            break
            
    if first_detected_frame_idx == -1:
        print("[ERROR] No tracking detections found in the stub file.")
        return
    
    # Open video capture and seek to target frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_detected_frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        print(f"[ERROR] Failed to read frame {first_detected_frame_idx} from video.")
        return
        
    # 4. Draw a bright bounding box around every detected person, overlay track ID in massive text
    annotated_frame = frame.copy()
    detections = player_detections[first_detected_frame_idx]
    
    for track_id, data in detections.items():
        bbox = data.get('bbox') if isinstance(data, dict) else data
        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bright bounding box (neon green BGR: 0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Overlay track ID number in massive text (fontScale=1.5, thickness=3, color bright red BGR: 0, 0, 255)
            label = f"ID: {track_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_x = x1
            text_y = max(y1 - 15, 50)  # Make sure it doesn't clip off the top of the frame
            
            cv2.putText(annotated_frame, label, (text_x, text_y), font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
            print(f"[DIAGNOSTIC] Annotated Track ID {track_id} at bbox [{x1}, {y1}, {x2}, {y2}]")
            
    # 5. Save single annotated image to root directory as TRACKER_ID_DIAGNOSTIC.jpg
    output_path = "TRACKER_ID_DIAGNOSTIC.jpg"
    cv2.imwrite(output_path, annotated_frame)
    
    # 6. Print terminal success message
    print(f"[SUCCESS] Diagnostic frame saved! Please open '{output_path}' to identify your player IDs.")

if __name__ == "__main__":
    main()
