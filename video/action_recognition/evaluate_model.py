import logging
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from datetime import datetime
from .train_model import PoseDataset, PoseClassifier  # Updated import

def setup_directories(project_root: Path) -> tuple:
    """Create and return paths to project directories."""
    model_dir = project_root / "models"
    log_dir = project_root / "training_logs"
    eval_dir = project_root / "evaluation_results"
    
    for d in [model_dir, log_dir, eval_dir]:
        d.mkdir(exist_ok=True)
    
    return model_dir, log_dir, eval_dir

def load_model(model_path: Path, device: torch.device) -> tuple:
    """Load the trained model and class names."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with same architecture
    input_size = 17 * 3
    hidden_size = 128
    num_classes = len(checkpoint['class_names'])
    
    model = PoseClassifier(input_size, hidden_size, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint['class_names']

def evaluate_model(model_path: Path, data_dir: Path):
    """Evaluate the trained model and generate performance metrics."""
    project_root = Path(__file__).parent.parent
    model_dir, log_dir, eval_dir = setup_directories(project_root)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = eval_dir / f'evaluation_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    device = torch.device('cpu')
    logging.info(f"Using device: {device}")
    
    # Load model
    model, class_names = load_model(model_path, device)
    logging.info(f"Loaded model from {model_path}")
    
    # Create dataset and dataloader
    dataset = PoseDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Collect predictions and true labels
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for poses, labels in tqdm(dataloader, desc="Evaluating"):
            poses = poses.to(device)
            outputs = model(poses)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate and print classification report
    report = classification_report(all_labels, all_preds, 
                                target_names=class_names, 
                                digits=3)
    logging.info("\nClassification Report:\n" + report)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names, 
                annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    # Save plot
    cm_path = eval_dir / f'confusion_matrix_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(cm_path)
    logging.info(f"Saved confusion matrix plot to {cm_path}")
    
    # Find top-5 best and worst performing classes
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    class_samples = cm.sum(axis=1)
    
    # Sort classes by accuracy
    sorted_idx = np.argsort(class_accuracies)
    
    # Print worst performing classes
    logging.info("\nWorst performing classes:")
    for idx in sorted_idx[:5]:
        logging.info(f"{class_names[idx]}: {class_accuracies[idx]:.3f} "
                    f"(samples: {class_samples[idx]})")
    
    # Print best performing classes
    logging.info("\nBest performing classes:")
    for idx in sorted_idx[-5:]:
        logging.info(f"{class_names[idx]}: {class_accuracies[idx]:.3f} "
                    f"(samples: {class_samples[idx]})")

def predict_video(model_path: Path, video_pose_path: Path):
    """Predict the action class for a single video."""
    device = torch.device('cpu')
    
    # Load model
    model, class_names = load_model(model_path, device)
    
    # Process single video
    dataset = PoseDataset(video_pose_path.parent)
    pose_tensor = dataset._process_pose_data(video_pose_path)
    pose_tensor = pose_tensor.unsqueeze(0).to(device)  # Add batch dimension
    
    # Get prediction
    with torch.no_grad():
        output = model(pose_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        values, indices = torch.topk(probabilities, 5)
        
    # Print top 5 predictions
    logging.info(f"\nPredictions for {video_pose_path.name}:")
    for prob, idx in zip(values[0], indices[0]):
        logging.info(f"{class_names[idx]}: {prob:.3f}")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "UCF101" / "poses"
    model_dir, _, _ = setup_directories(project_root)
    
    # Find the most recent model file
    model_files = list(model_dir.glob("model_*.pth"))
    if not model_files:
        raise FileNotFoundError("No model file found in models directory!")
    
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    
    # Evaluate the model
    evaluate_model(latest_model, data_dir)
    
    # Optional: predict a single video
    # video_pose_path = data_dir / "v_BaseballPitch_g01_c01_poses.json"
    # predict_video(latest_model, video_pose_path)