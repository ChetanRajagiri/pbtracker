import cv2
cap = cv2.VideoCapture("input_videos/malaysia.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("malaysia.mp4 frames:", total_frames)
cap.release()
