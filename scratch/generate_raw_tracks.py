import os
import cv2
import pickle
from trackers.player_tracker import PlayerTracker

def main():
    video_path = "input_videos/newclip.mp4"
    player_stub_path = "tracker_stubs/player_detections_raw.pkl"
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames to track: {total_frames}")
    
    player_tracker = PlayerTracker(model_path='yolov8x.pt')
    
    player_detections = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Run detection on single frame
        detection_dict = player_tracker.detect_frames([frame], read_from_stub=False)[0]
        player_detections.append(detection_dict)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Tracked {frame_idx}/{total_frames} frames...")
    cap.release()
    
    with open(player_stub_path, 'wb') as f:
        pickle.dump(player_detections, f)
    print(f"Saved raw detections to {player_stub_path}")

    # Analyze raw tracks
    id_lifespans = {}
    id_bboxes = {}
    for frame_dict in player_detections:
        for track_id, data in frame_dict.items():
            bbox = data.get('bbox')
            if track_id not in id_lifespans:
                id_lifespans[track_id] = 0
                id_bboxes[track_id] = []
            id_lifespans[track_id] += 1
            id_bboxes[track_id].append(bbox)
            
    print("\n--- Raw Track ID Statistics ---")
    for tid in sorted(id_lifespans.keys()):
        count = id_lifespans[tid]
        bboxes = id_bboxes[tid]
        xs = [(b[0] + b[2])/2.0 for b in bboxes]
        ys = [(b[1] + b[3])/2.0 for b in bboxes]
        x_range = max(xs) - min(xs) if xs else 0
        y_range = max(ys) - min(ys) if ys else 0
        print(f"Raw ID {tid}: Lifespan = {count} frames, X-Range = {x_range:.1f}px, Y-Range = {y_range:.1f}px")

if __name__ == "__main__":
    main()
