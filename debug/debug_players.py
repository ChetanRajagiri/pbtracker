import os
import pickle
import cv2
import numpy as np

def main():
    # Paths
    keypoints_path = "tracker_stubs/court_keypoints.pkl"
    player_detections_path = "tracker_stubs/player_detections.pkl"
    video_path = "input_videos/input.mp4"

    # Verify files exist
    if not os.path.exists(keypoints_path):
        print(f"Error: {keypoints_path} not found.")
        return
    if not os.path.exists(player_detections_path):
        print(f"Error: {player_detections_path} not found.")
        return
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    # Load keypoints
    print(f"Loading court keypoints from: {keypoints_path}")
    with open(keypoints_path, 'rb') as f:
        keypoints = pickle.load(f)

    # Load raw player detections
    print(f"Loading player detections from: {player_detections_path}")
    with open(player_detections_path, 'rb') as f:
        player_detections = pickle.load(f)

    # Read Frame 0 of the video
    print(f"Reading first frame (Frame 0) of: {video_path}")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Error: Could not read Frame 0 from video.")
        return

    # Reconstruct the court boundary polygon
    if len(keypoints) < 4:
        print("Error: Fewer than 4 keypoints loaded. Cannot define court boundary.")
        return

    corner1 = keypoints[0]
    corner2 = keypoints[1]
    corner3 = keypoints[2]
    corner4 = keypoints[3]

    polygon = np.array([corner1, corner2, corner3, corner4], dtype=np.int32)

    # Draw semi-transparent blue boundary polygon on the frame
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], (255, 0, 0))  # Blue in BGR
    alpha = 0.35
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # Also draw a solid blue outline around the boundary polygon
    cv2.polylines(frame, [polygon], True, (255, 0, 0), 2)

    # Extract all player bounding boxes present in the raw detections for Frame 0
    # Note: player_detections is a list of dictionaries (one per frame)
    frame_0_detections = player_detections[0] if len(player_detections) > 0 else {}

    print("\n--- Frame 0 Player Detection & Polygon Test Results ---")
    
    for track_id, bbox in frame_0_detections.items():
        x1, y1, x2, y2 = bbox
        
        # Calculate the foot coordinate (bottom center of their box)
        x_foot = int((x1 + x2) / 2)
        y_foot = int(y2)
        
        # Run point polygon test (returns >= 0 if inside or on boundary)
        dist = cv2.pointPolygonTest(polygon, (x_foot, y_foot), False)
        is_inside = dist >= 0
        
        # Print status log
        print(f"Track ID: {track_id:2d} | BBox: [{x1:7.2f}, {y1:7.2f}, {x2:7.2f}, {y2:7.2f}] | "
              f"Foot: ({x_foot:4d}, {y_foot:4d}) | Inside Polygon: {is_inside}")
        
        # Draw the bounding box on the frame in bright red
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        
        # Draw the foot coordinate as a small solid red circle
        cv2.circle(frame, (x_foot, y_foot), 6, (0, 0, 255), -1)
        
        # Label the box with the Track ID (Black text on white label box)
        label = f"ID: {track_id} (Inside: {is_inside})"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (int(x1), int(y1) - 20), (int(x1) + w + 10, int(y1)), (255, 255, 255), -1)
        cv2.putText(frame, label, (int(x1) + 5, int(y1) - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Save the resulting annotated frame
    output_path = "debug_frame_0.jpg"
    cv2.imwrite(output_path, frame)
    print(f"\nSaved annotated debugging frame to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
