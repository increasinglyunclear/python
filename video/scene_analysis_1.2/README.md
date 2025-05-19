# Scene Analysis - Advanced Video Understanding System

This project implements an advanced video analysis system that combines multiple AI models to provide comprehensive video understanding, including scene analysis, object detection, pose estimation, temporal analysis, and audio processing.

## Project Structure

```
scene_analysis/
├── video_analysis/
│   ├── video_analyzer.py      # Main video analysis implementation
│   ├── scene_understanding.py # Scene analysis module
│   └── test_analyzer.py       # Test script for video analysis
├── models/                    # Directory for model files (not included in repo)
├── training_data/            # Directory for training data and results
├── requirements.txt          # Python dependencies
├── download_models.py        # Python script to download models
├── download_models.sh        # Shell script to download models
└── README.md                # This file
```

## Features

- **Multi-stream Video Analysis**
  - Scene understanding using DINO/ViT models
  - Object detection with enhanced YOLO
  - Human pose estimation using MediaPipe
  - Temporal analysis with 3D CNN
  - Audio analysis with spectrogram processing

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for video processing)
- FFmpeg (for audio processing)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd scene_analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models:
```bash
# Using Python script (recommended)
python download_models.py

# Or using shell script
chmod +x download_models.sh
./download_models.sh
```

The models will be downloaded to the `models` directory. This step is required before running the video analysis.

## Usage

```python
from video_analysis.video_analyzer import EnhancedVideoAnalyzer

# Initialize the analyzer
analyzer = EnhancedVideoAnalyzer()

# Analyze a video file
results = analyzer.analyze_video("path/to/your/video.mov")

# Access different analysis components
scene_analysis = results['scene_understanding']
object_analysis = results['object_analysis']
pose_analysis = results['pose_analysis']
temporal_analysis = results['temporal_analysis']
audio_analysis = results['audio_analysis']
```

## Testing

Run the test script to verify the video analysis functionality:

```bash
python video_analysis/test_analyzer.py
```

## Output Format

The video analysis results are returned as a dictionary with the following structure:

```python
{
    'scene_understanding': {
        'features': [...],
        'summary': {
            'primary_scene': str,
            'scene_transitions': int,
            'unique_scenes': list
        }
    },
    'object_analysis': {
        'detections': [...],
        'summary': {
            'total_objects_detected': int,
            'unique_objects': int,
            'most_common_objects': dict
        }
    },
    'pose_analysis': {
        'poses': [...],
        'summary': {
            'total_frames': int,
            'frames_with_poses': int,
            'pose_detection_rate': float
        }
    },
    'temporal_analysis': {
        'features': [...],
        'summary': {
            'temporal_segments': int,
            'average_feature_magnitude': float,
            'feature_std_dev': float
        }
    },
    'audio_analysis': {
        'features': [...],
        'summary': {
            'audio_chunks': int,
            'total_audio_energy': float,
            'average_chunk_energy': float
        }
    }
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Acknowledgments

- YOLOv8 for object detection
- MediaPipe for pose estimation
- PyTorch for deep learning models
- FFmpeg for audio processing 