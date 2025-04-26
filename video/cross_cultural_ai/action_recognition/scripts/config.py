"""
Configuration settings for action recognition
"""

class Config:
    # Model settings
    INPUT_SIZE = 34 * 3  # 17 keypoints * (x, y, confidence)
    HIDDEN_SIZE = 64
    NUM_CLASSES = 5  # Update this based on your model
    
    # Video processing
    FRAME_SKIP = 2  # Process every nth frame
    FRAME_SAMPLE_RATE = 30  # Process every Nth frame
    
    # Confidence thresholds
    POSE_CONF_THRESHOLD = 0.25  # Keeping your original value
    ACTION_CONF_THRESHOLD = 0.5
    
    # YOLO settings
    YOLO_POSE_MODEL = 'yolov8x-pose.pt'  # Keeping your original model
    
    # Processing settings
    BATCH_SIZE = 32
    
    @classmethod
    def get_model_config(cls):
        return {
            'input_size': cls.INPUT_SIZE,
            'hidden_size': cls.HIDDEN_SIZE,
            'num_classes': cls.NUM_CLASSES
        }