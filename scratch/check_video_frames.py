import cv2

for name in ["newclip.mp4", "newvideo.mp4", "input1.mp4", "final.mp4"]:
    cap = cv2.VideoCapture(f"input_videos/{name}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{name}: {total_frames} frames, {width}x{height}, {fps} fps")
    cap.release()
