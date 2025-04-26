# Scene Description Pipeline v1.0

A robust pipeline for scene understanding and classification using deep learning. This pipeline uses a ResNet50 model pre-trained on the Places365 dataset to classify scenes into 365 different categories.

## Features

- Scene classification into 365 distinct categories
- High-accuracy scene recognition
- Detailed probability outputs for top predictions
- Visualization of classification results
- JSON output for programmatic use
- Support for multiple image formats (JPG, JPEG, PNG)

## Directory Structure

```
scene_analysis/
├── README.md
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
pip install torch torchvision opencv-python numpy matplotlib
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

## Model Details

The pipeline uses a ResNet50 model pre-trained on the Places365 dataset, which includes 365 scene categories covering:
- Indoor scenes (homes, offices, public spaces)
- Outdoor scenes (natural landscapes, urban environments)
- Architectural structures
- Transportation spaces
- Recreational areas

## Performance

- Processing time: ~1-2 seconds per image (CPU)
- Accuracy: State-of-the-art scene classification
- Memory usage: ~500MB for model weights

## Future Improvements

- Add scene segmentation capabilities
- Implement batch processing for multiple images
- Add support for video input
- Include more detailed scene descriptions
- Add confidence threshold filtering

## License

MIT License

## Acknowledgments

- Places365 dataset and model from MIT CSAIL
- PyTorch and torchvision teams
- OpenCV community 