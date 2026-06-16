import os
import pickle
import json
import numpy as np

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(x) for x in obj]
    return obj

def main():
    out_dir = "dashboard/public/data"
    os.makedirs(out_dir, exist_ok=True)
    
    stubs = {
        "court_keypoints": "tracker_stubs/court_keypoints.pkl",
        "player_detections": "tracker_stubs/player_detections.pkl",
        "ball_detections": "tracker_stubs/ball_detections.pkl",
        "ball_events": "tracker_stubs/ball_events.pkl"
    }
    
    for name, filepath in stubs.items():
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            serializable_data = convert_to_serializable(data)
            
            out_file = os.path.join(out_dir, f"{name}.json")
            with open(out_file, 'w') as f_out:
                json.dump(serializable_data, f_out, indent=2)
            print(f"[EXPORT] Saved {name} to {out_file}")
        else:
            print(f"[WARNING] Stub file not found: {filepath}")

if __name__ == "__main__":
    main()
