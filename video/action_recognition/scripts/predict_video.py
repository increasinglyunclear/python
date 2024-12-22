import logging
from pathlib import Path
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from .train_model import PoseClassifier
from collections import Counter
import argparse  # Added this import

# Rest of the script remains the same...

class ActionPredictor:
    def __init__(self, model_path: Path):
        # Load YOLO model
        self.pose_model = YOLO('yolov8x-pose.pt')
        
        # Load our trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize classifier
        input_size = 17 * 3  # 17 keypoints with x,y,conf
        hidden_size = 128
        num_classes = len(checkpoint['class_names'])
        
        self.model = PoseClassifier(input_size, hidden_size, num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.class_names = checkpoint['class_names']
        self.sequence_length = 32
        self.pose_sequence = []
        self.all_predictions = []
    
    def process_frame(self, frame):
        results = self.pose_model(frame)
        return results[0]
    
    def update_pose_sequence(self, pose_keypoints):
        self.pose_sequence.append(pose_keypoints)
        if len(self.pose_sequence) > self.sequence_length:
            self.pose_sequence.pop(0)
    
    def get_prediction(self):
        if len(self.pose_sequence) < self.sequence_length:
            return None
        
        pose_data = np.array(self.pose_sequence)
        pose_tensor = torch.FloatTensor(pose_data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(pose_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            values, indices = torch.topk(probabilities, 3)
        
        predictions = [(self.class_names[idx], prob.item()) 
                      for prob, idx in zip(values[0], indices[0])]
        self.all_predictions.extend([pred[0] for pred in predictions[:1]])  # Store top prediction
        return predictions
    
    def get_final_predictions(self):
        if not self.all_predictions:
            return []
        
        # Count occurrences of each action
        action_counts = Counter(self.all_predictions)
        total_predictions = len(self.all_predictions)
        
        # Get top 3 actions with their percentages
        top_actions = action_counts.most_common(3)
        return [(action, count/total_predictions) for action, count in top_actions]

def analyze_video(video_path: str, model_path: Path):
    """Analyze video and return top 3 predicted actions."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    predictor = ActionPredictor(model_path)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = predictor.process_frame(frame)
            
            if results.keypoints is not None and len(results.keypoints) > 0:
                kpts = results.keypoints[0].data.cpu().numpy()
                flattened_kpts = kpts.flatten()
                predictor.update_pose_sequence(flattened_kpts)
                predictor.get_prediction()
                
    finally:
        cap.release()
    
    return predictor.get_final_predictions()

def main():
    parser = argparse.ArgumentParser(description='Action Recognition from Video')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--model', type=str,
                        help='Path to model file (will use latest if not specified)')
    
    args = parser.parse_args()
    
    # Find model path
    if args.model:
        model_path = Path(args.model)
    else:
        project_root = Path(__file__).parent.parent
        model_dir = project_root / "models"
        model_files = list(model_dir.glob("model_*.pth"))
        if not model_files:
            raise FileNotFoundError("No model file found in models directory!")
        model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    
    # Analyze video
    predictions = analyze_video(args.video, model_path)
    
    # Print results
    print("\nTop 3 Predicted Actions:")
    print("-" * 30)
    for action, confidence in predictions:
        print(f"{action}: {confidence:.1%}")

if __name__ == "__main__":
    main()