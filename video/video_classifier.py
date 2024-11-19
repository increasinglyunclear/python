"""
CCAI video classifier 
19 Nov 2024
Kevin Walker

TO RUN:

Navigate to the directory where the script is.

Create folders named 'input' and 'output'
(in Terminal: mkdir input output)

In Terminal:

# First, make sure you're in the right environment
conda activate videoclassifier

# Install opencv
conda install opencv

# Verify it's installed
python -c "import cv2; print(cv2.__version__)"

Then run the script:

python video_classifier.py

When running, it will wait for new files in the 'input' directory, 
create new CSV files
then move the video files to the 'output' directory.

"""

import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import csv
from datetime import datetime
from pathlib import Path
import time
import logging

class VideoClassifier:
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
        
        logging.info("Loading ResNet model...")
        self.model = ResNet50(weights='imagenet')
        logging.info("Model loaded successfully")
        
    def process_video(self, video_path):
        video_name = Path(video_path).stem
        results_file = self.output_dir / f"{video_name}_results.csv"
        
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
            writer.writerow(['timestamp', 'frame_number', 'class', 'confidence'])
            
            frame_count = 0
            sample_rate = 30  # Process every 30th frame
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % sample_rate == 0:
                    print(f"\rProcessing frame {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)", end='')
                    
                    # Process frame here
                    processed = self.process_frame(frame)
                    predictions = self.model.predict(processed)
                    results = decode_predictions(predictions, top=3)[0]
                    
                    timestamp = frame_count / fps
                    for class_id, class_name, confidence in results:
                        writer.writerow([timestamp, frame_count, class_name, confidence])
                
                frame_count += 1
            
        cap.release()
        print(f"\nCompleted processing {video_path}")

    def process_frame(self, frame):
        resized = cv2.resize(frame, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(rgb, axis=0)
        return preprocess_input(img_array)

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
                print("\nStopping video classifier...")
                break
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                time.sleep(5)

def main():
    # You can customize these paths
    INPUT_DIR = "input"
    OUTPUT_DIR = "output"
    
    classifier = VideoClassifier(INPUT_DIR, OUTPUT_DIR)
    classifier.watch_directory()

if __name__ == "__main__":
    main()
