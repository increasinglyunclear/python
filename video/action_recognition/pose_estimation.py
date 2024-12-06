"""
Pose Estimation Module

This module provides functionality for extracting human pose data from videos using YOLOv8.

Dependencies:
    - opencv-python
    - ultralytics
    - numpy
    - torch

Usage:
    As a script:
        python -m scripts.pose_estimation
    
    As a module:
        from pose_estimation import PoseEstimator
        estimator = PoseEstimator()
        estimator.process_video("input.mp4", "output.json")
"""

import cv2
from ultralytics import YOLO
import json
from pathlib import Path
import numpy as np
import sys

class PoseEstimator:
    def __init__(self):
        """Initialize the YOLO pose estimation model"""
        print("Initializing YOLO model...")
        self.model = YOLO('yolov8n-pose.pt')
        print("Model initialized")

    def process_frame(self, frame):
        """
        Process a single frame to extract pose data.
        
        Args:
            frame: numpy array, Input frame
            
        Returns:
            list: List of detected poses with keypoints
        """
        try:
            results = self.model(frame, verbose=False)[0]
            
            if results.keypoints is not None and len(results.keypoints) > 0:
                keypoints = results.keypoints.data.cpu().numpy()
                
                frame_data = []
                for person_keypoints in keypoints:
                    kpts = person_keypoints.reshape(-1, 3).tolist()
                    frame_data.append({
                        "keypoints": kpts
                    })
                return frame_data
        except Exception as e:
            print(f"Error processing frame: {e}")
        return []

    def process_video(self, video_path, output_path):
        """
        Process entire video and save pose data to JSON.
        
        Args:
            video_path: str, Path to input video
            output_path: str, Path to save JSON output
            
        Returns:
            list: All detected poses across frames
        """
        print(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        frame_count = 0
        all_poses = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                poses = self.process_frame(frame)
                all_poses.append({
                    "frame": frame_count,
                    "poses": poses
                })
                
                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"Processed {frame_count} frames")

        except Exception as e:
            print(f"Error during video processing: {e}")
        finally:
            cap.release()

        print(f"Saving results to: {output_path}")
        try:
            with open(output_path, 'w') as f:
                json.dump({
                    "total_frames": frame_count,
                    "frames": all_poses
                }, f, indent=2)
        except Exception as e:
            print(f"Error saving results: {e}")

        return all_poses

def process_input_directory():
    """Process all videos in the input directory"""
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    pose_estimator = PoseEstimator()

    video_extensions = ('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV')
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_dir.glob(f'*{ext}'))

    print(f"\nFound {len(video_files)} videos to process")
    
    results = {"successful": [], "failed": []}
    
    for video_path in video_files:
        try:
            output_path = output_dir / f"{video_path.stem}_poses.json"
            print(f"\nProcessing: {video_path.name}")
            
            poses = pose_estimator.process_video(str(video_path), str(output_path))
            results["successful"].append(str(video_path))
            
        except Exception as e:
            print(f"Failed to process {video_path.name}: {e}")
            results["failed"].append(str(video_path))

    print("\nProcessing complete!")
    print(f"Successfully processed: {len(results['successful'])} videos")
    if results['successful']:
        print("Successful videos:")
        for video in results['successful']:
            print(f" - {video}")
    
    print(f"\nFailed to process: {len(results['failed'])} videos")
    if results['failed']:
        print("Failed videos:")
        for video in results['failed']:
            print(f" - {video}")

if __name__ == "__main__":
    process_input_directory()
