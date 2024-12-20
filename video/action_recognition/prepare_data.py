"""
Data preparation module for action recognition.
Handles loading and preprocessing of pose data from JSON files.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PoseDataset(Dataset):
    def __init__(self, data_dir, sequence_length=30):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str or Path): Directory containing pose JSON files
            sequence_length (int): Number of frames to use in each sequence
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        
        # Load all pose files and their labels
        self.pose_files = []
        self.labels = []
        
        logger.info(f"Initializing PoseDataset from {self.data_dir}")
        
        # Walk through the poses directory
        for pose_file in self.data_dir.glob("**/*_poses.json"):
            # Extract class name from the parent directory
            action_class = pose_file.parent.name
            self.pose_files.append(pose_file)
            self.labels.append(action_class)
        
        # Encode labels to numbers
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        
        logger.info(f"Found {len(self.pose_files)} sequences")
        logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
        logger.info(f"Classes: {self.label_encoder.classes_}")

    def __len__(self):
        return len(self.pose_files)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item to get
            
        Returns:
            tuple: (poses, label) where poses is a tensor of shape 
                  (sequence_length, 17, 3) and label is a tensor of shape (1,)
        """
        try:
            # Load pose data from JSON
            with open(self.pose_files[idx]) as f:
                pose_data = json.load(f)

            # Extract poses from frames
            poses = []
            for frame in pose_data['frames']:
                if frame['poses'] and len(frame['poses']) > 0:
                    pose = frame['poses'][0]  # Take first person's poses
                    if 'keypoints' in pose and len(pose['keypoints']) == 17:
                        # Ensure keypoints are in the correct format
                        keypoints = np.array(pose['keypoints']).reshape(-1, 3)
                        poses.append(keypoints)
                    else:
                        # If keypoints are missing or malformed, use zero array
                        poses.append(np.zeros((17, 3)))
                else:
                    # If no poses detected, use zero array
                    poses.append(np.zeros((17, 3)))

            # Convert list of arrays to a single array
            poses = np.stack(poses, axis=0)

            # Pad or truncate to sequence_length
            if len(poses) > self.sequence_length:
                poses = poses[:self.sequence_length]
            elif len(poses) < self.sequence_length:
                # Pad with zeros
                pad_length = self.sequence_length - len(poses)
                padding = np.zeros((pad_length, 17, 3))
                poses = np.concatenate([poses, padding], axis=0)

            # Normalize coordinates (optional)
            # You might want to add normalization here depending on your needs

            # Convert to tensor
            poses = torch.FloatTensor(poses)
            label = torch.LongTensor([self.encoded_labels[idx]])

            return poses, label

        except Exception as e:
            logger.error(f"Error processing file {self.pose_files[idx]}: {e}")
            # Return a zero tensor if there's an error
            return torch.zeros((self.sequence_length, 17, 3)), torch.LongTensor([self.encoded_labels[idx]])

def create_dataloaders(data_dir, batch_size=32, sequence_length=30):
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir (str or Path): Directory containing pose JSON files
        batch_size (int): Batch size for the dataloaders
        sequence_length (int): Number of frames to use in each sequence
        
    Returns:
        tuple: (train_loader, val_loader, label_encoder)
    """
    try:
        logger.info(f"Creating dataloaders from {data_dir}")
        
        # Create dataset
        dataset = PoseDataset(data_dir, sequence_length)
        
        # Split into train and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        logger.info(f"Training set size: {len(train_dataset)}")
        logger.info(f"Validation set size: {len(val_dataset)}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,  # Reduced for debugging
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2,  # Reduced for debugging
            pin_memory=True,
            drop_last=True
        )
        
        return train_loader, val_loader, dataset.label_encoder

    except Exception as e:
        logger.error(f"Error creating dataloaders: {e}")
        raise

if __name__ == "__main__":
    # Test the data loading
    data_dir = Path("data/UCF101/poses")
    train_loader, val_loader, label_encoder = create_dataloaders(data_dir)
    
    # Print some statistics
    logger.info("\nTesting data loading:")
    logger.info(f"Number of training batches: {len(train_loader)}")
    logger.info(f"Number of validation batches: {len(val_loader)}")
    
    # Test a batch
    for poses, labels in train_loader:
        logger.info(f"\nBatch information:")
        logger.info(f"Pose batch shape: {poses.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        logger.info(f"Sample labels: {[label_encoder.inverse_transform([l]) for l in labels[:5]]}")
        break