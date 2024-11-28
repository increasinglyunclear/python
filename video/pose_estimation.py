"""
CCAI Pose Estimation Script using YOLO model
Kevin Walker
28 Nov 2024
This estimates human poses in video, writes the results to CSV file
--------------------------

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
- Place video files in the 'input' directory
- Script will automatically process videos and save results to 'output' directory
- Results are saved as CSV files with keypoint data
- Processed videos are moved to 'output' directory with 'processed_' prefix
- Press Ctrl+C to stop the script

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
        """
        Initialize the PoseEstimator with input and output directories.
        
        Args:
            input_dir (str): Path to input directory containing video files
            output_dir (str): Path to output directory for results and processed videos
        """
        print("Initializing Pose Estimator...")
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        # Create directories if they don't exist
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        # Load YOLO pose estimation model
        self.pose_model = YOLO('yolov8n-pose.pt')

        # Configure logging
        logging.basicConfig(level=logging.INFO)

    def process_video(self, video_path):
        """
        Process a video file and extract pose estimation data.
        
        Args:
            video_path (Path): Path to the input video file
        """
        video_name = video_path.stem
        pose_file = self.output_dir / f"{video_name}_poses.csv"
        
        print(f"\nProcessing video: {video_path}")
        
        # Open video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.error(f"Could not open video file: {video_path}")
            return
        
        # Get video properties
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Open CSV file for writing results
        with open(pose_file, 'w', newline='') as pose_csv:
            pose_writer = csv.writer(pose_csv)
            # Write CSV headers
            pose_writer.writerow([
                'frame_number',
                'person_id',
                'keypoint_name',
                'x',
                'y',
                'confidence'
            ])
            
            # Process each frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process pose estimation
                try:
                    # Print progress every 100 frames
                    if frame_count % 100 == 0:
                        print(f"Processing frame {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)")
                    
                    # Run pose estimation on current frame
                    results = self.pose_model(frame)
                    
                    # Process each detection
                    for result in results:
                        # Check if keypoints were detected
                        if hasattr(result, 'keypoints') and result.keypoints is not None:
                            # Get all keypoints for all detected persons
                            all_keypoints = result.keypoints.data
                            
                            # Process each person's keypoints
                            for person_idx in range(len(all_keypoints)):
                                person_keypoints = all_keypoints[person_idx]
                                
                                # Process each keypoint
                                for kp_idx, keypoint in enumerate(person_keypoints):
                                    # Extract coordinates and confidence
                                    x, y, conf = keypoint.cpu().numpy()
                                    
                                    # Get keypoint name from model's dictionary
                                    kp_name = self.pose_model.names.get(kp_idx, f"keypoint_{kp_idx}")
                                    
                                    # Write keypoint data to CSV
                                    pose_writer.writerow([
                                        frame_count,
                                        person_idx,
                                        kp_name,
                                        float(x),  # Convert to float for CSV compatibility
                                        float(y),
                                        float(conf)
                                    ])
                    
                except Exception as e:
                    logging.error(f"Error processing frame {frame_count}: {str(e)}")
                    print(f"Error details for frame {frame_count}: {type(e).__name__}: {str(e)}")
                    continue  # Continue with next frame even if current frame fails
                
                frame_count += 1
                
        # Clean up
        cap.release()
        print(f"\nCompleted processing {video_path}")
        print(f"Processed {frame_count} frames")
        print(f"Pose estimation results saved to: {pose_file}")

    def watch_directory(self):
        """
        Watch the input directory for new video files and process them automatically.
        Press Ctrl+C to stop watching.
        """
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
    """
    Main entry point for the script.
    """
    print("Starting Pose Estimator...")
    estimator = PoseEstimator()
    estimator.watch_directory()

if __name__ == "__main__":
    main()
