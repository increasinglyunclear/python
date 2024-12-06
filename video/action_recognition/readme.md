# Action Recognition using Pose Estimation

This project implements an action recognition system using pose estimation. It processes videos to extract human pose data, which can then be used to train a model for action recognition.

## Project Structure

├── data/
│ └── UCF101/ # UCF101 dataset
│ ├── videos/ # Raw video files
│ └── poses/ # Processed pose JSON files
├── input/ # Directory for test videos
├── output/ # Directory for test output
└── scripts/
├── init.py
├── pose_estimation.py # Core pose estimation functionality
└── process_ucf101.py # UCF101 dataset processing


## Dependencies
- Python 3.8+
- OpenCV (`opencv-python`)
- PyTorch
- Ultralytics YOLO (`ultralytics`)
- NumPy

## Installation

bash
# Create and activate conda environment (recommended):
conda create -n action_recognition python=3.8
conda activate action_recognition

# Install dependencies:
pip install ultralytics opencv-python numpy torch torchvision


## Usage

1. **Process Test Videos**
   # Place test videos in the input/ directory
   python -m scripts.pose_estimation

2. **Process UCF101 Dataset**
   # Ensure UCF101 videos are in data/UCF101/videos/
   python -m scripts.process_ucf101


## Output Format
The pose estimation generates JSON files containing:
- Frame-by-frame pose keypoints
- Confidence scores for each keypoint
- Total frame count and processing metadata

## License
# GPL - any commercial uses require notification
