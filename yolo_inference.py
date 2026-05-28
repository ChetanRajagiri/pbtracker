import os
import glob
import cv2
from ultralytics import YOLO

def main():
    # Load the pre-trained YOLOv8x model
    print("Loading YOLOv8x model...")
    model = YOLO("yolov8x.pt")

    # Locate sample video in input_videos/ directory
    input_dir = "input_videos"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created directory '{input_dir}'. Please place a tennis video inside it.")
        return

    # Find the first video in the directory
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not video_files:
        print(f"No video files found in '{input_dir}'. Supported formats: mp4, avi, mov, mkv.")
        print("Please place a video file in 'input_videos/' and run again.")
        return

    sample_video_path = video_files[0]
    print(f"Found sample video: {sample_video_path}")

    # Open the video file
    cap = cv2.VideoCapture(sample_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {sample_video_path}")
        return

    # Read the first frame
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Error: Could not read the first frame from the video.")
        return

    print("Successfully read the first frame. Running YOLOv8x inference...")

    # Run inference on the first frame
    results = model(frame)[0]

    # Print the detected bounding box coordinates for humans (class 0)
    print("\n--- Detection Results (Class 0: Person) ---")
    detected_persons = 0
    for box in results.boxes:
        class_id = int(box.cls[0].item())
        confidence = box.conf[0].item()
        
        # Class 0 corresponds to 'person' in the COCO dataset
        if class_id == 0:
            detected_persons += 1
            # Coordinates are in format [xmin, ymin, xmax, ymax]
            coords = box.xyxy[0].tolist()
            print(f"Person {detected_persons}: Bounding Box = [x1={coords[0]:.2f}, y1={coords[1]:.2f}, x2={coords[2]:.2f}, y2={coords[3]:.2f}], Confidence = {confidence:.2%}")

    if detected_persons == 0:
        print("No persons detected in the first frame.")
    else:
        print(f"\nDetection complete. Detected {detected_persons} person(s) in the first frame.")

    # Save the annotated frame to the project root directory
    print("Generating and saving annotated image...")
    annotated_frame = results.plot()
    output_path = "first_frame_detection.jpg"
    cv2.imwrite(output_path, annotated_frame)
    print(f"Saved annotated first frame to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
