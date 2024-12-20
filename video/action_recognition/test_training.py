import logging
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class PoseDataset(Dataset):
    def __init__(self, data_dir: Path, sequence_length: int = 32):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        
        # Get all pose files
        self.pose_files = list(data_dir.glob("*_poses.json"))
        
        # Extract class names from filenames (v_[ClassName]_g##_c##_poses.json)
        self.class_names = sorted(list(set(
            f.stem.split('_')[1] for f in self.pose_files
        )))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        logging.info(f"Found {len(self.class_names)} classes: {self.class_names}")

    def __len__(self):
        return len(self.pose_files)

    def __getitem__(self, idx):
        pose_file = self.pose_files[idx]
        
        # Get class name and index
        class_name = pose_file.stem.split('_')[1]
        class_idx = self.class_to_idx[class_name]
        
        # Load and process pose data
        pose_tensor = self._process_pose_data(pose_file)
        
        return pose_tensor, class_idx

    def _process_pose_data(self, pose_file: Path) -> torch.Tensor:
        """Process pose data from a JSON file into a tensor.
        
        Args:
            pose_file: Path to the JSON file containing pose data
            
        Returns:
            Tensor of shape (sequence_length, num_keypoints * 3) containing the pose data
        """
        with open(pose_file) as f:
            data = json.load(f)
        
        total_frames = data['total_frames']
        num_keypoints = 17  # COCO format has 17 keypoints
        
        # Initialize array with zeros
        poses = np.zeros((total_frames, num_keypoints, 3))
        
        # Process each frame
        for frame_data in data['frames']:
            frame_idx = frame_data['frame']
            
            # Skip if no poses detected in this frame or empty keypoints
            if not frame_data['poses'] or not frame_data['poses'][0]['keypoints']:
                continue
                
            # Take the first detected pose in the frame
            pose = frame_data['poses'][0]
            keypoints = np.array(pose['keypoints'])
            
            # Verify keypoints shape before assignment
            if keypoints.shape == (num_keypoints, 3):
                poses[frame_idx] = keypoints
        
        # Normalize sequence length
        if self.sequence_length > 0:
            # Use average pooling to normalize sequence length
            target_len = self.sequence_length
            current_len = total_frames
            
            if current_len > target_len:
                # Downsample using average pooling
                indices = np.linspace(0, current_len-1, target_len, dtype=int)
                poses = poses[indices]
            else:
                # Upsample by repeating last pose
                padding = np.repeat(poses[-1:], target_len - current_len, axis=0)
                poses = np.vstack([poses, padding])
        
        # Flatten keypoints dimension
        poses = poses.reshape(poses.shape[0], -1)
        
        return torch.FloatTensor(poses)

class PoseClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(PoseClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # LSTM expects input of shape (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Use the last time step's output for classification
        out = self.fc(lstm_out[:, -1, :])
        return out

def test_training(data_dir: Path):
    """Test training loop with pose data.
    
    Args:
        data_dir: Path to directory containing pose JSON files
    """
    logging.info(f"Starting test training with data from: {data_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create dataset
    dataset = PoseDataset(data_dir)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logging.info(f"Training set size: {len(train_dataset)}")
    logging.info(f"Validation set size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    input_size = 17 * 3  # 17 keypoints * 3 values (x, y, confidence)
    hidden_size = 128
    num_classes = len(dataset.class_names)
    
    model = PoseClassifier(input_size, hidden_size, num_classes)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 1
    
    for epoch in range(num_epochs):
        model.train()
        
        # Training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for poses, labels in pbar:
            poses = poses.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(poses)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for poses, labels in val_loader:
                poses = poses.to(device)
                labels = labels.to(device)
                
                outputs = model(poses)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        logging.info(f"Validation Accuracy: {val_accuracy:.2f}%")

if __name__ == "__main__":
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "UCF101" / "poses"
    
    test_training(data_dir)