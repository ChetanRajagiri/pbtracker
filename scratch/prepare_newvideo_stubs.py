import pickle
import os
import shutil

def main():
    stubs = ["court_keypoints.pkl", "player_detections.pkl", "ball_detections.pkl"]
    
    for stub in stubs:
        src = f"tracker_stubs/{stub}"
        if os.path.exists(src):
            # Check length or data details
            with open(src, 'rb') as f:
                data = pickle.load(f)
            
            # Print info
            if stub == "court_keypoints.pkl":
                print(f"court_keypoints.pkl: {len(data)} keypoints")
            else:
                print(f"{stub}: {len(data)} frames")
                
            # Copy to newvideo dynamic path
            base, ext = os.path.splitext(stub)
            dst = f"tracker_stubs/{base}_newvideo{ext}"
            shutil.copy(src, dst)
            print(f"Copied {src} to {dst}")
        else:
            print(f"Source stub not found: {src}")

if __name__ == "__main__":
    main()
