import pickle
import cv2
import os

def main():
    video_path = "input_videos/newclip.mp4"
    raw_stub = "tracker_stubs/player_detections_raw.pkl"
    
    if not os.path.exists(raw_stub):
        print("No raw player stub found!")
        return
        
    with open(raw_stub, 'rb') as f:
        player_detections = pickle.load(f)
        
    cap = cv2.VideoCapture(video_path)
    
    # We want to find the first frame where each raw ID appears and save its crop
    raw_id_crops = {}
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx < len(player_detections):
            frame_dict = player_detections[frame_idx]
            for track_id, data in frame_dict.items():
                bbox = data.get('bbox')
                if bbox and track_id not in raw_id_crops:
                    x1, y1, x2, y2 = map(int, bbox)
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        raw_id_crops[track_id] = (frame_idx, crop)
                        
        frame_idx += 1
        
    cap.release()
    
    os.makedirs("debug_raw", exist_ok=True)
    # Filter for track IDs that have lifespan > 100
    # Let's count lifespans first
    id_lifespans = {}
    for f_dict in player_detections:
        for tid in f_dict:
            id_lifespans[tid] = id_lifespans.get(tid, 0) + 1
            
    for tid, (f_idx, crop) in raw_id_crops.items():
        lifespan = id_lifespans.get(tid, 0)
        if lifespan > 100:
            out_path = f"debug_raw/raw_{tid}_lifespan_{lifespan}_frame_{f_idx}.jpg"
            cv2.imwrite(out_path, crop)
            print(f"Saved raw ID {tid} (lifespan {lifespan}) crop to {out_path}")

if __name__ == "__main__":
    main()
