import logging
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
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
        """Process pose data from a JSON file into a tensor."""
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
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc1(lstm_out[:, -1, :])  # Take only the last time step
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def train_model(data_dir: Path):
    """Train the pose classifier model."""
    logging.info(f"Starting full training with data from: {data_dir}")
    
    # Set device
    device = torch.device('cpu')
    logging.info(f"Using device: {device}")
    
    # Create dataset
    dataset = PoseDataset(data_dir)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logging.info(f"Training set size: {len(train_dataset)}")
    logging.info(f"Validation set size: {len(val_dataset)}")
    
    # Create data loaders with smaller batch size for CPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16,  # Reduced batch size
        shuffle=True,
        num_workers=2   # Reduced workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16,
        shuffle=False,
        num_workers=2
    )
    
    # Create model
    input_size = 17 * 3
    hidden_size = 128
    num_classes = len(dataset.class_names)
    
    model = PoseClassifier(input_size, hidden_size, num_classes)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=2, verbose=True
    )
    
    # Training loop
    num_epochs = 10
    best_accuracy = 0
    best_model_path = Path(f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        
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
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        
        with torch.no_grad():
            for poses, labels in val_loader:
                poses = poses.to(device)
                labels = labels.to(device)
                
                outputs = model(poses)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - start_time
        
        # Log metrics
        logging.info(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Time: {epoch_time:.2f}s - "
            f"Train Loss: {avg_train_loss:.4f} - "
            f"Val Loss: {avg_val_loss:.4f} - "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'class_names': dataset.class_names,
            }, best_model_path)
            logging.info(f"Saved new best model with accuracy: {best_accuracy:.2f}%")
        
        # Adjust learning rate
        scheduler.step(val_accuracy)
    
    logging.info(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
    logging.info(f"Best model saved to: {best_model_path}")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "UCF101" / "poses"
    
    train_model(data_dir)