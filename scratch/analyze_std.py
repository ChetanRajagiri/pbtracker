import pickle
import numpy as np

def main():
    with open("tracker_stubs/player_detections_raw.pkl", 'rb') as f:
        player_detections = pickle.load(f)
        
    track_box_centers = {}
    for frame_dict in player_detections:
        for track_id, data in frame_dict.items():
            bbox = data.get('bbox')
            if bbox:
                x_center = (bbox[0] + bbox[2]) / 2.0
                y_center = (bbox[1] + bbox[3]) / 2.0
                if track_id not in track_box_centers:
                    track_box_centers[track_id] = []
                track_box_centers[track_id].append((x_center, y_center))
                
    print("Track ID | Lifespan | Std X | Std Y | Max Range")
    for tid, centers in sorted(track_box_centers.items()):
        if len(centers) < 50:
            continue
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        std_x = np.std(xs)
        std_y = np.std(ys)
        range_x = max(xs) - min(xs)
        range_y = max(ys) - min(ys)
        max_range = max(range_x, range_y)
        print(f"ID {tid:<3} | {len(centers):<8} | {std_x:<5.1f} | {std_y:<5.1f} | {max_range:<5.1f}")

if __name__ == "__main__":
    main()
