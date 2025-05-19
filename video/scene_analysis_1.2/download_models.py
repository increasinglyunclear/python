#!/usr/bin/env python3

import os
import requests
from tqdm import tqdm
from pathlib import Path

def download_file(url: str, destination: Path):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Create destination directory if it doesn't exist
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def main():
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Model URLs
    models = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolov8x.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"
    }
    
    # Download each model
    for model_name, url in models.items():
        print(f"\nDownloading {model_name}...")
        destination = models_dir / model_name
        download_file(url, destination)
    
    print("\nDownload complete!")
    print("Models are stored in the 'models' directory")

if __name__ == "__main__":
    main() 