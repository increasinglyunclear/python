# Scene Description Pipeline v1.1

A robust pipeline for scene understanding and classification using deep learning. This pipeline uses an ensemble of ResNet50 models pre-trained on the Places365 dataset to classify scenes into 365 different categories with improved accuracy.

## Features

- Ensemble-based scene classification using multiple pre-trained models
- Scene classification into 365 distinct categories
- High-accuracy scene recognition through model averaging
- Detailed probability outputs for top predictions
- Visualization of classification results
- JSON output for programmatic use
- Support for multiple image formats (JPG, JPEG, PNG)

## Directory Structure

```
scene_analysis/
├── README.md
├── requirements.txt
├── scene_understanding.py      # Core scene understanding module
├── test_scene_understanding.py # Test script for the pipeline
├── test_images/               # Directory for test images
│   └── ...                    # Your test images here
└── results/                   # Output directory
    ├── analysis_*.json        # JSON results files
    └── *_analysis.png         # Visualization files
```

## Requirements

- Python 3.8+
- PyTorch 1.7+
- torchvision
- OpenCV
- NumPy
- Matplotlib

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your images in the `test_images` directory
2. Run the test script:
```bash
python test_scene_understanding.py
```

The script will:
- Process all images in the test_images directory
- Generate visualizations in the results directory
- Save detailed analysis in JSON format

## Model Architecture

The pipeline uses an ensemble of two ResNet50 models:
1. Places365 model - Trained on the full Places365 dataset
2. Places365-Standard model - A more comprehensive version with additional training data

The final prediction is obtained by averaging the probabilities from both models, which helps to:
- Reduce prediction variance
- Improve accuracy on ambiguous scenes
- Handle edge cases better

## Output Format

### JSON Output
```json
{
    "image": "filename.jpg",
    "timestamp": "2024-04-26T19:21:55.123456",
    "results": {
        "classification": {
            "top_categories": [
                {
                    "category": "category_name",
                    "probability": 0.9999
                },
                ...
            ],
            "raw_output": [...]
        },
        "segmentation": null  # Currently disabled
    }
}
```

### Visualization
For each processed image, a visualization is generated showing:
- Original image
- Top scene categories with probabilities
- Bar chart of confidence scores

## Performance

- Processing time: ~2-3 seconds per image (CPU)
- Accuracy: Improved through ensemble approach
- Memory usage: ~1GB for both model weights

## Future Improvements

- Add scene segmentation capabilities
- Implement batch processing for multiple images
- Add support for video input
- Include more detailed scene descriptions
- Add confidence threshold filtering
- Experiment with different ensemble weighting strategies

## License

MIT License

## Acknowledgments

- Places365 dataset and models from MIT CSAIL
- PyTorch and torchvision teams
- OpenCV community 