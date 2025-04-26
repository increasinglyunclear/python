"""
Video analysis module for action recognition
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from .pose_estimation import PoseEstimator
from .config import Config
from .train_model import PoseClassifier  # Import the action recognition model
import torch

logger = logging.getLogger(__name__)

def analyze_video(video_path, model_path):
    """
    Analyze video for pose estimation and action recognition
    
    Args:
        video_path: Path to video file
        model_path: Path to the model checkpoint
        
    Returns:
        dict: Analysis results containing poses and actions
    """
    try:
        # Initialize pose estimator (uses YOLO model from Config)
        pose_estimator = PoseEstimator()
        
        # Initialize action recognition model
        action_model = PoseClassifier(input_size=51, hidden_size=128, num_classes=101)
        checkpoint = torch.load(model_path, map_location='cpu')
        action_model.load_state_dict(checkpoint['model_state_dict'])
        action_model.eval()
        
        # Get class names from checkpoint
        class_names = checkpoint.get('class_names', [str(i) for i in range(101)])
        
        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        frames = []
        frame_count = 0
        
        # Read frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every Nth frame
            if frame_count % Config.FRAME_SAMPLE_RATE == 0:
                frames.append(frame)
            frame_count += 1
            
        cap.release()
        
        # Process frames for pose estimation
        poses = pose_estimator.process_video_frames(frames)
        
        # Process poses with action recognition model
        if poses is not None:
            with torch.no_grad():
                pose_data = poses.reshape(1, poses.shape[0], -1)  # Reshape to (1, N, 51)
                pose_tensor = torch.FloatTensor(pose_data)
                action_output = action_model(pose_tensor)
                predicted_actions = torch.softmax(action_output, dim=1)
                # Get top actions and their probabilities
                top_probs, top_indices = predicted_actions[0].topk(3)
                # Map indices to class names
                actions = [(class_names[idx.item()], float(prob.item())) 
                          for idx, prob in zip(top_indices, top_probs)]
        else:
            actions = []
        
        results = {
            'poses': poses if poses is not None else [],
            'actions': actions,
            'frame_count': frame_count,
            'processed_frames': len(frames)
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        return None

if __name__ == "__main__":
    # Test video analysis
    test_video = "path/to/test/video.mp4"  # Replace with actual path
    test_model = "path/to/model.pth"       # Replace with actual path
    if Path(test_video).exists():
        results = analyze_video(test_video, test_model)
        if results:
            print("Successfully analyzed video!")
            print(f"Processed {results['processed_frames']} frames")
            print(f"Found {len(results['poses'])} valid poses")
