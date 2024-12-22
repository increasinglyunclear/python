import cv2
import numpy as np
from ultralytics import YOLO
import logging
from pathlib import Path

class ObjectDetector:
    def __init__(self):
        logging.info("Loading YOLO model...")
        self.model = YOLO('yolov8n.pt')
        logging.info("Object detection model loaded successfully")
    
    def detect_objects(self, video_path: str, confidence_threshold: float = 0.3) -> list:
        """
        Detect objects in a video file and return unique detected objects.
        
        Args:
            video_path: Path to video file
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of unique object classes detected
        """
        detected_objects = set()
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logging.error(f"Could not open video file: {video_path}")
            return list(detected_objects)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        sample_rate = 30  # Process every 30th frame
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_rate == 0:
                    # Run detection on frame
                    results = self.model(frame, verbose=False)
                    
                    # Process each detection
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            confidence = float(box.conf)
                            if confidence >= confidence_threshold:
                                class_id = int(box.cls)
                                class_name = self.model.names[class_id]
                                detected_objects.add(class_name)
                
                frame_count += 1
                
        finally:
            cap.release()
        
        return list(detected_objects)

# Initialize global detector instance
_detector = None

def detect_objects(video_path: str) -> list:
    """
    Wrapper function to detect objects in a video using a singleton detector.
    
    Args:
        video_path: Path to video file
        
    Returns:
        List of unique object classes detected
    """
    global _detector
    if _detector is None:
        _detector = ObjectDetector()
    
    return _detector.detect_objects(video_path)

if __name__ == "__main__":
    # Test the detector
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        objects = detect_objects(video_path)
        print("\nDetected objects:")
        print(", ".join(objects))