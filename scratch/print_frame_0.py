import pickle

with open("tracker_stubs/player_detections.pkl", 'rb') as f:
    player_detections = pickle.load(f)
    
print("Frame 0 Player Detections:")
for pid, data in player_detections[0].items():
    print(f"Player {pid}: {data['bbox']}")
