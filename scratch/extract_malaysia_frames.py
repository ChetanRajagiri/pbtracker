import cv2
import os

def main():
    video_path = "output_videos/annotated_malaysia.mp4"
    if not os.path.exists(video_path):
        print("No output video found!")
        return
        
    cap = cv2.VideoCapture(video_path)
    
    frames_to_save = [903]
    os.makedirs("debug_malaysia_final", exist_ok=True)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx in frames_to_save:
            out_path = f"debug_malaysia_final/final_frame_{frame_idx}.jpg"
            cv2.imwrite(out_path, frame)
            print(f"Saved final annotated frame to {out_path}")
            
        frame_idx += 1
        
    cap.release()

if __name__ == "__main__":
    main()
