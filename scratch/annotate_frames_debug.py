import pickle
import cv2
import os

def main():
    video_path = "input_videos/newclip.mp4"
    player_stub = "tracker_stubs/player_detections.pkl"
    court_stub = "tracker_stubs/court_keypoints.pkl"
    
    with open(player_stub, 'rb') as f:
        player_detections = pickle.load(f)
    with open(court_stub, 'rb') as f:
        court_keypoints = pickle.load(f)
        
    cap = cv2.VideoCapture(video_path)
    
    frames_to_save = [0, 100, 300, 500, 800]
    os.makedirs("debug_frames", exist_ok=True)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx in frames_to_save:
            annotated = frame.copy()
            
            # Draw keypoints in blue
            for i, kp in enumerate(court_keypoints):
                cv2.circle(annotated, kp, 4, (255, 0, 0), -1)
                cv2.putText(annotated, str(i+1), (kp[0]+5, kp[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
            # Draw player detections
            if frame_idx < len(player_detections):
                frame_dict = player_detections[frame_idx]
                for pid, data in frame_dict.items():
                    bbox = data['bbox']
                    is_on_court = data.get('is_on_court', True)
                    x1, y1, x2, y2 = map(int, bbox)
                    color = (0, 255, 0) if is_on_court else (0, 165, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, f"Player {pid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
            out_path = f"debug_frames/frame_{frame_idx}.jpg"
            cv2.imwrite(out_path, annotated)
            print(f"Saved debug frame to {out_path}")
            
        frame_idx += 1
        
    cap.release()

if __name__ == "__main__":
    main()
