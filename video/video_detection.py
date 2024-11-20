"""
CCAI object detection in video
using YOLOv8 model

Kevin Walker
20 Nov 2024

Dependencies:
ultralytics
pytorch
torchvision
opencv
numpy

To run, first install packages:

# Make sure you're in your videoclassifier environment
conda activate videoclassifier

conda install pytorch torchvision ultralytics opencv numpy -c pytorch -c conda-forge

# Verify the installation
python -c "import torch; print(torch.__version__)"

Then run the script:
python video_detector.py

"""

import signal
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import csv
from datetime import datetime
from pathlib import Path
import time
import logging

def signal_handler(sig, frame):
    print('\nGracefully shutting down...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class VideoObjectDetector:
    def __init__(self, input_dir="input", output_dir="output"):
        # Setup directories
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create directories if they don't exist
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        logging.info("Loading YOLO model...")
        self.model = YOLO('yolov8n.pt')  # Using the nano model, can use 's', 'm', or 'l' for larger models
        logging.info("Model loaded successfully")
        
    def process_video(self, video_path):
        video_name = Path(video_path).stem
        results_file = self.output_dir / f"{video_name}_detections.csv"
        
        print(f"Processing video: {video_path}")
        print(f"Results will be saved to: {results_file}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.error(f"Could not open video file: {video_path}")
            return
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        with open(results_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'frame_number', 'object_class', 'confidence', 'x_min', 'y_min', 'x_max', 'y_max'])
            
            frame_count = 0
            sample_rate = 30  # Process every 30th frame
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % sample_rate == 0:
                    print(f"\rProcessing frame {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)", end='')
                    
                    # Run detection on frame
                    results = self.model(frame)
                    
                    timestamp = frame_count / fps
                    
                    # Process each detection
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            # Get box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            # Get class and confidence
                            class_id = int(box.cls)
                            class_name = self.model.names[class_id]
                            confidence = float(box.conf)
                            
                            # Write to CSV
                            writer.writerow([
                                timestamp,
                                frame_count,
                                class_name,
                                confidence,
                                x1, y1, x2, y2
                            ])
                
                frame_count += 1
            
        cap.release()
        print(f"\nCompleted processing {video_path}")

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
                    print(f"\nFound {len(video_files)} video files:")
                    for video_path in video_files:
                        print(f"- {video_path}")
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
                print("\nStopping video detector...")
                break
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                time.sleep(5)

def main():
    # You can customize these paths
    INPUT_DIR = "input"
    OUTPUT_DIR = "output"
    
    detector = VideoObjectDetector(INPUT_DIR, OUTPUT_DIR)
    detector.watch_directory()

if __name__ == "__main__":
    main()
