from ultralytics import YOLO
import cv2
import json
from pathlib import Path

class PoseEstimator:
    def __init__(self):
        # Initialize your YOLO model here
        self.model = YOLO('yolov8n-pose.pt')  # Example model, adjust as needed

    def process_video(self, video_path, output_path):
        # Ensure output_path is a Path object
        output_path = Path(output_path)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        poses = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run pose estimation on the frame
            results = self.model(frame)
            poses.append(results)  # Adjust this to extract the necessary pose data

        cap.release()

        # Save poses to the output path
        with open(output_path, 'w') as f:
            json.dump(poses, f)

        return poses