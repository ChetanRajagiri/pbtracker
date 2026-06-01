import os
import pickle
import cv2

class ManualCourtDetector:
    def __init__(self):
        pass
        
    def get_keypoints(self, first_frame, stub_path='tracker_stubs/court_keypoints.pkl'):
        # Check if stub exists
        if stub_path and os.path.exists(stub_path):
            print(f"Loading court keypoints from stub: {stub_path}")
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
                
        keypoints = []
        display_frame = first_frame.copy()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(keypoints) < 12:
                    keypoints.append((x, y))
                    print(f"Collected keypoint {len(keypoints)}/12 at: ({x}, {y})")
                    # Draw a red dot at selection point
                    cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
                    # Label the dot
                    cv2.putText(display_frame, str(len(keypoints)), (x + 8, y - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow("Select 12 Court Keypoints", display_frame)

        print("\n=== Court Keypoint Selection CLI ===")
        print("Please click exactly 12 points representing the court bounds in the popup window.")
        
        cv2.namedWindow("Select 12 Court Keypoints")
        cv2.setMouseCallback("Select 12 Court Keypoints", mouse_callback)
        cv2.imshow("Select 12 Court Keypoints", display_frame)
        
        while len(keypoints) < 12:
            key = cv2.waitKey(10) & 0xFF
            if key == 27:  # Escape key
                print("ESC pressed. Exiting point collection early.")
                break
                
        cv2.destroyAllWindows()
        
        if len(keypoints) == 12:
            if stub_path:
                stub_dir = os.path.dirname(stub_path)
                if stub_dir and not os.path.exists(stub_dir):
                    os.makedirs(stub_dir)
                print(f"Saving court keypoints to stub: {stub_path}")
                with open(stub_path, 'wb') as f:
                    pickle.dump(keypoints, f)
        else:
            print("Warning: Collected fewer than 12 points. Not saving stub.")
            
        return keypoints

    def draw_keypoints_on_video(self, video_frames, keypoints):
        annotated_frames = []
        for frame in video_frames:
            annotated_frame = frame.copy()
            for i, kp in enumerate(keypoints):
                x, y = kp
                # Draw a blue circle (BGR: (255, 0, 0)) for visual confirmation
                cv2.circle(annotated_frame, (x, y), 5, (255, 0, 0), -1)
                # Label the point
                cv2.putText(annotated_frame, str(i + 1), (x + 8, y - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
            annotated_frames.append(annotated_frame)
        return annotated_frames
