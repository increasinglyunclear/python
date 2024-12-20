from pathlib import Path
import json
import sys
import os
import logging
import torch
import numpy as np

# Add the scripts directory to the Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from prepare_data import create_dataloaders

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def inspect_pose_file(file_path, logger):
    """Inspect the contents of a pose file"""
    logger.info(f"\nInspecting pose file: {file_path.name}")
    try:
        with open(file_path) as f:
            data = json.load(f)
            
        logger.info(f"File structure:")
        logger.info(f"├── Keys: {list(data.keys())}")
        logger.info(f"├── Total frames: {data['total_frames']}")
        logger.info(f"└── Frames with poses: {sum(1 for frame in data['frames'] if frame['poses'])}")
        
        # Inspect first frame with poses
        for frame in data['frames']:
            if frame['poses']:
                logger.info("\nSample frame structure:")
                logger.info(f"├── Frame number: {frame['frame']}")
                logger.info(f"├── Number of people detected: {len(frame['poses'])}")
                logger.info(f"└── Keypoints per person: {len(frame['poses'][0]['keypoints'])}")
                break
                
        return data
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None

def test_data_preparation():
    logger = setup_logging()
    logger.info("Starting data preparation test...")
    
    # Initialize paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "UCF101" / "poses"
    
    # Check directory structure
    logger.info("\nChecking directory structure:")
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Data directory: {data_dir}")
    
    if not data_dir.exists():
        logger.error(f"Error: Data directory not found: {data_dir}")
        return
        
    # List and categorize files
    logger.info("\nScanning for pose files...")
    pose_files = list(data_dir.glob("**/*_poses.json"))
    if not pose_files:
        logger.error("No pose files found!")
        return
        
    # Analyze directory structure
    action_classes = set(f.parent.name for f in pose_files)
    logger.info(f"\nDataset statistics:")
    logger.info(f"├── Total pose files: {len(pose_files)}")
    logger.info(f"├── Action classes: {len(action_classes)}")
    logger.info(f"└── Sample classes: {list(action_classes)[:5]}")
    
    # Inspect sample files
    logger.info("\nInspecting sample files...")
    for file in pose_files[:3]:
        sample_data = inspect_pose_file(file, logger)
        if sample_data is None:
            continue
    
    # Test dataloader creation
    logger.info("\nTesting dataloader creation...")
    try:
        train_loader, val_loader, label_encoder = create_dataloaders(
            data_dir,
            batch_size=32,
            sequence_length=30
        )
        
        logger.info("\nDataloader statistics:")
        logger.info(f"├── Training batches: {len(train_loader)}")
        logger.info(f"├── Validation batches: {len(val_loader)}")
        logger.info(f"└── Action classes: {len(label_encoder.classes_)}")
        
        logger.info("\nAction class mapping:")
        for i, class_name in enumerate(label_encoder.classes_):
            logger.info(f"├── {i}: {class_name}")
            
        # Test batch loading
        logger.info("\nTesting batch loading...")
        for poses, labels in train_loader:
            logger.info("\nBatch information:")
            logger.info(f"├── Batch shape: {poses.shape}")
            logger.info(f"├── Labels shape: {labels.shape}")
            logger.info(f"├── Pose value range: [{torch.min(poses):.2f}, {torch.max(poses):.2f}]")
            logger.info(f"├── Memory usage: {poses.element_size() * poses.nelement() / 1024 / 1024:.2f} MB")
            logger.info(f"└── Sample labels: {[label_encoder.inverse_transform([l]) for l in labels[:5]]}")
            
            # Detailed pose analysis
            logger.info("\nPose sequence analysis:")
            sample_pose = poses[0]  # First sequence in batch
            logger.info(f"├── Sequence length: {sample_pose.shape[0]}")
            logger.info(f"├── Keypoints per frame: {sample_pose.shape[1]}")
            logger.info(f"├── Features per keypoint: {sample_pose.shape[2]}")
            logger.info(f"└── Non-zero poses: {torch.count_nonzero(sample_pose).item()}")
            break
            
    except Exception as e:
        logger.error(f"Error during dataloader testing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    logger.info("\nData preparation test completed successfully!")

if __name__ == "__main__":
    test_data_preparation()