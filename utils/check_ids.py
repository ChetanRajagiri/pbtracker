import pickle

with open("tracker_stubs/player_detections.pkl", 'rb') as f:
    player_detections = pickle.load(f)

id_summary = {}
for frame_idx, frame_players in enumerate(player_detections):
    for pid in frame_players.keys():
        if pid not in id_summary:
            id_summary[pid] = {'first_seen': frame_idx, 'total_frames': 0}
        id_summary[pid]['total_frames'] += 1

print("\n=== DETECTED TRACK IDS SUMMARY ===")
for pid, stats in sorted(id_summary.items(), key=lambda x: x[1]['total_frames'], reverse=True):
    print(f"Track ID: {pid:3d} | Total Frames Present: {stats['total_frames']:4d} | First Appeared on Frame: {stats['first_seen']}")