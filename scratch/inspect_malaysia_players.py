import pickle
import numpy as np

def main():
    player_stub = "tracker_stubs/player_detections_malaysia.pkl"
    with open(player_stub, 'rb') as f:
        player_detections = pickle.load(f)
        
    print(f"Total frames: {len(player_detections)}")
    
    # Check detections for each player ID (1, 2, 3, 4)
    for pid in [1, 2, 3, 4]:
        detected_frames = []
        bboxes = []
        for idx, frame_dict in enumerate(player_detections):
            if pid in frame_dict:
                detected_frames.append(idx)
                bboxes.append(frame_dict[pid].get('bbox'))
                
        if detected_frames:
            mean_bbox = np.mean(bboxes, axis=0)
            print(f"Player {pid}: Present in {len(detected_frames)} frames | First frame: {detected_frames[0]}, Last frame: {detected_frames[-1]} | Mean bbox: {mean_bbox.tolist()}")
        else:
            print(f"Player {pid}: Never present in any frame!")

if __name__ == "__main__":
    main()
