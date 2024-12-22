"""
Pose estimation using YOLOv8 pose model
"""

from ultralytics import YOLO
import numpy as np
import logging
from pathlib import Path
from .config import Config

logger = logging.getLogger(__name__)

class PoseEstimator:
    def __init__(self):
        """Initialize YOLO pose estimation model"""
        try:
            logger.info("Loading YOLO pose model...")
            self.model = YOLO(Config.YOLO_POSE_MODEL)
            logger.info("YOLO pose model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO pose model: {str(e)}")
            raise

    def extract_poses(self, frame):
        """
        Extract pose keypoints from a single frame
        
        Args:
            frame: numpy array of shape (H, W, 3)
            
        Returns:
            numpy array of shape (17, 3) containing keypoint coordinates and confidence
            or None if no pose detected
        """
        try:
            results = self.model(frame, verbose=False)
            
            # Get keypoints from first detected person
            if results and len(results) > 0 and results[0].keypoints is not None:
                keypoints = results[0].keypoints.data
                if len(keypoints) > 0:
                    # Get first person's keypoints
                    person_keypoints = keypoints[0].cpu().numpy()
                    
                    # Check confidence threshold
                    if np.mean(person_keypoints[:, 2]) > Config.POSE_CONF_THRESHOLD:
                        return person_keypoints
            
            return None
            
        except Exception as e:
            logger.error(f"Error in pose extraction: {str(e)}")
            return None
    
    def process_video_frames(self, frames):
        """
        Process multiple frames and extract pose sequences
        
        Args:
            frames: list of numpy arrays, each of shape (H, W, 3)
            
        Returns:
            numpy array of shape (N, 17, 3) where N is number of valid poses
        """
        poses = []
        
        for frame in frames:
            pose = self.extract_poses(frame)
            if pose is not None:
                poses.append(pose)
        
        if not poses:
            return None
            
        return np.stack(poses)

if __name__ == "__main__":
    # Test the pose estimator
    import cv2
    
    # Initialize pose estimator
    estimator = PoseEstimator()
    
    # Test on a single image
    test_image = "path/to/test/image.jpg"  # Replace with actual path
    if Path(test_image).exists():
        frame = cv2.imread(test_image)
        pose = estimator.extract_poses(frame)
        if pose is not None:
            print("Successfully extracted pose!")
            print(f"Shape: {pose.shape}")
            print(f"Average confidence: {np.mean(pose[:, 2]):.2f}")