import os
import shutil

def main():
    suffixes = ["court_keypoints", "player_detections", "ball_detections", "ball_events"]
    for suffix in suffixes:
        src = f"tracker_stubs/{suffix}_newvideo.pkl"
        dst = f"tracker_stubs/{suffix}_malaysia.pkl"
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Copied {src} to {dst}")
            # Optional: delete the newvideo version to keep it clean
            os.remove(src)
            print(f"Removed temporary {src}")

if __name__ == "__main__":
    main()
