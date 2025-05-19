#!/bin/bash

# Create models directory if it doesn't exist
mkdir -p models

# Download YOLOv8n model (smaller version)
echo "Downloading YOLOv8n model..."
wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Download YOLOv8x model (larger version)
echo "Downloading YOLOv8x model..."
wget -P models https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt

echo "Download complete!"
echo "Models are stored in the 'models' directory" 