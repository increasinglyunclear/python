"""
Simple video Classifier using ResNet50
Outputs a local CSV file for a given video file
Runs locally only for now

Dependencies:
- Python 3.10
- opencv-python (cv2): for video processing
- tensorflow: for deep learning model
- numpy: for numerical operations
- keras (included with tensorflow): for ResNet50 model

Install with conda:
    conda create -n videoclassifier python=3.10
    conda activate videoclassifier
    conda install numpy
    conda install tensorflow
    conda install opencv

Alternative pip installation:
    pip install opencv-python
    pip install tensorflow
    pip install numpy

Note: GPU support requires additional setup with CUDA and cuDNN if using tensorflow-gpu
"""


import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import csv
from datetime import datetime
from pathlib import Path

class VideoClassifier:
    def __init__(self):
        print("Loading ResNet model...")
        self.model = ResNet50(weights='imagenet')
        print("Model loaded successfully")
        
    def process_frame(self, frame):
        resized = cv2.resize(frame, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(rgb, axis=0)
        return preprocess_input(img_array)
        
    def classify_frame(self, frame):
        processed = self.process_frame(frame)
        predictions = self.model.predict(processed)
        return decode_predictions(predictions, top=3)[0]
    
    def process_video(self, video_path):
        # Create output filename based on input video name
        video_name = Path(video_path).stem
        results_file = f"{video_name}_results.csv"
        
        print(f"Processing video: {video_path}")
        print(f"Results will be saved to: {results_file}")
        
        cap = cv2.VideoCapture(video_path)
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
                    
                # Process every nth frame
                if frame_count % sample_rate == 0:
                    timestamp = frame_count / fps
                    print(f"\rProcessing frame {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)", end='')
                    
                    results = self.classify_frame(frame)
                    
                    for class_id, class_name, confidence in results:
                        writer.writerow([
                            timestamp,
                            frame_count,
                            class_name,
                            confidence
                        ])
                
                frame_count += 1
            
        cap.release()
        print(f"\nCompleted processing {video_path}")
        print(f"Results saved to {results_file}")

def main():
    # Replace this with your video file path
    VIDEO_PATH = "path/your_vid.mp4"
    
    classifier = VideoClassifier()
    classifier.process_video(VIDEO_PATH)

if __name__ == "__main__":
    main()
