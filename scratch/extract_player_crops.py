import pickle
import cv2
import os

def main():
    video_path = "input_videos/newclip.mp4"
    player_stub = "tracker_stubs/player_detections.pkl"
    
    if not os.path.exists(player_stub):
        print("No player stub found!")
        return
        
    with open(player_stub, 'rb') as f:
        player_detections = pickle.load(f)
        
    cap = cv2.VideoCapture(video_path)
    
    crops_to_save = {1: None, 2: None, 3: None, 4: None}
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx < len(player_detections):
            frame_dict = player_detections[frame_idx]
            for pid in [1, 2, 3, 4]:
                if pid in frame_dict and crops_to_save[pid] is None:
                    bbox = frame_dict[pid].get('bbox')
                    if bbox:
                        x1, y1, x2, y2 = map(int, bbox)
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            crops_to_save[pid] = (frame_idx, crop)
                            
        # Stop if we found a crop for all players
        if all(val is not None for val in crops_to_save.values()):
            break
        frame_idx += 1
        
    cap.release()
    
    os.makedirs("debug", exist_ok=True)
    for pid, data in crops_to_save.items():
        if data:
            frame_num, crop = data
            out_path = f"debug/player_{pid}_frame_{frame_num}.jpg"
            cv2.imwrite(out_path, crop)
            print(f"Saved player {pid} crop from frame {frame_num} to {out_path}")
        else:
            print(f"No crop found for player {pid}")

if __name__ == "__main__":
    main()
