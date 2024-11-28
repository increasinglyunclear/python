"""
CCAI Pose Estimation Script
Kevin Walker
28 Nov 2024
Detects poses from people in video, writes the results to a CSV file.
This version run locally.
----------------------

Setup Steps:
1. Install Homebrew (if not already installed):
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

2. Install FFmpeg:
   brew install ffmpeg

3. Install Anaconda or Miniconda (if not already installed):
   Visit: https://docs.conda.io/en/latest/miniconda.html

4. Create and activate conda environment:
   conda create -n poseestimation python=3.10
   conda activate poseestimation

5. Install required packages:
   pip install ultralytics opencv-python

6. Create directories:
   mkdir -p input output

7. Place video files in the 'input' directory.

8. Run script:
   python3 pose_estimation.py

Usage:
- The script will process video files in the 'input' directory.
- Pose estimation results will be saved as CSVs in the 'output' directory.
- Press Ctrl+C to stop the script.

Supported video formats: .mp4, .avi, .mov, .MOV
"""

from ultralytics import YOLO
import cv2
import csv
from pathlib import Path
import time
import logging

class PoseEstimator:
    def __init__(self, input_dir="input", output_dir="output"):
        print("Initializing Pose Estimator...")  # Debugging output
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        # Create directories if they don't exist
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        # Load pose model
        self.pose_model = YOLO('yolov8n-pose.pt')  # Load the pose estimation model

        logging.basicConfig(level=logging.INFO)

    def process_video(self, video_path):
        video_name = video_path.stem
        pose_file = self.output_dir / f"{video_name}_poses.csv"
        
        print(f"\nProcessing video: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.error(f"Could not open video file: {video_path}")
            return
        
        frame_count = 0
        
        # Open pose results file
        with open(pose_file, 'w', newline='') as pose_csv:
            pose_writer = csv.writer(pose_csv)
            # Write headers
            pose_writer.writerow([
                'frame_number',
                'person_id',
                'keypoint_name',
                'x',
                'y',
                'confidence'
            ])
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process pose estimation
                try:
                    pose_results = self.pose_model(frame)
                    print(f"Pose results: {pose_results}")  # Debugging output
                    
                    if pose_results is not None and len(pose_results) > 0:
                        for r in pose_results:
                            if r.keypoints is not None and len(r.keypoints.data) > 0:
                                keypoints = r.keypoints.data[0].cpu().numpy()
                                for person_id in range(len(r.boxes)):
                                    for kp_id, keypoint in enumerate(keypoints):
                                        x, y, conf = keypoint
                                        kp_name = self.pose_model.names[kp_id]
                                        
                                        # Debugging output for pose estimation
                                        print(f"Detected keypoint: {kp_name} for person {person_id} with confidence {conf} at [{x}, {y}]")
                                        
                                        pose_writer.writerow([
                                            frame_count,
                                            person_id,
                                            kp_name,
                                            x, y, conf
                                        ])
                            else:
                                logging.warning(f"No keypoints detected for frame {frame_count}.")
                                print(f"No keypoints detected for frame {frame_count}.")  # Debugging output
                    else:
                        logging.warning(f"No pose results for frame {frame_count}.")
                        print(f"No pose results for frame {frame_count}.")  # Debugging output
                
                except Exception as e:
                    logging.error(f"Error during pose estimation for frame {frame_count}: {str(e)}")
                    print(f"Error during pose estimation for frame {frame_count}: {str(e)}")  # Print error details
            
                frame_count += 1
            
        cap.release()  # Ensure the video capture is released
        print(f"\nCompleted processing {video_path}")
        print(f"Pose estimation results saved to: {pose_file}")

    def watch_directory(self):
        """Watch input directory for new video files"""
        print(f"Watching directory: {self.input_dir.absolute()}")
        print(f"Results will be saved to: {self.output_dir.absolute()}")
        print("Waiting for video files... (Press Ctrl+C to stop)")
        
        # Define supported video formats
        supported_formats = ['*.mp4', '*.avi', '*.mov', '*.MOV']
        
        while True:
            try:
                # Look for video files
                video_files = []
                for format in supported_formats:
                    video_files.extend(self.input_dir.glob(format))
                
                if video_files:
                    for video_path in video_files:
                        try:
                            logging.info(f"Processing: {video_path}")
                            self.process_video(video_path)
                            
                            # Move processed file to output directory
                            processed_path = self.output_dir / f"processed_{video_path.name}"
                            video_path.rename(processed_path)
                            logging.info(f"Moved processed video to: {processed_path}")
                            
                        except Exception as e:
                            logging.error(f"Error processing {video_path}: {str(e)}")
            
                # Wait before checking again
                print("\nChecking for new files...")
                time.sleep(5)  # Check every 5 seconds
                
            except KeyboardInterrupt:
                print("\nStopping pose estimator...")
                break
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                time.sleep(5)

def main():
    print("Starting Pose Estimator...")  # Debugging output
    estimator = PoseEstimator()
    estimator.watch_directory()

if __name__ == "__main__":
    main()
