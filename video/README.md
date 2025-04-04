# Cross-Cultural AI
Funded by InnovateUK, 2024-25

This is a machine learning model for video analysis developed by me a part of the Cross-Cultural AI project. It automatically translates & analyses user-generated video to help generate culturally-specific insights and facilitate cross-cultural communication. It can recognise objects, people and actions, transcribe and translate more than 100 spoken languages to English. This emerged from my work on <a href="https://increasinglyunclear.substack.com/p/ai-for-nine-earths">Nine Earths</a> and <a href="https://ai.postdigitalcultures.org/">Ethnographic AI</a>. I completed a visual ethnography using hours of video footage from 12 countries to create films like <a href="https://www.youtube.com/watch?v=WFMKrOkJ7fA">this one</a>, and I thought AI could help – it can perform basic pattern recognition, but could it go further to make ethnographic interpretations about specific cultural practices? 

## Disclaimers

This is a v.1 prototype for proof of concept only. As a next step, participants will be engaged in a conversation with the system to retrain the model, or train a new one from scratch using unsupervised learning. The initial uses the YOLO and model, which was trained on publicly available image and video datasets known to be biased toward Western people using data harvested from the internet with consent or remuneration. Since this project aims to keep ethics at the centre, the first version was developed for proof of concept only. 

The other disclaimer is around security and potential use by 'bad actors'. As a 'dual-use technology', an AI system that identifies objects, people and actions across cultures might clearly be of interest to governments or organisations seeking to use such technology for political uses or human rights abuses. In this case, our prototype is simple and there are much more sophisticated ones in the public domain. More generally, we have been advised that open sourcing such software is a form of risk mitigation in itself.

# Using this video Analysis pipeline

Disclaimers given, this software is fully functional, and you can download and test it. When video files are placed into the 'input' folder, they are automatically analysed, and a CSV file is output containing the objects, people and actions recognised, as well as a transcript and translation of any spoken language.

A comprehensive video analysis pipeline that performs:
- Action recognition using YOLO pose estimation (101 action classes)
- Object detection using YOLOv8
- Audio transcription and translation with automatic detection of over 100 languages

## Directory Structure
```
pipeline/
├── action_recognition/
│   ├── models/
│   │   └── model_20241220_145338.pth  # Pre-trained action recognition model
│   └── scripts/
│       ├── __init__.py
│       ├── config.py          # Configuration settings
│       ├── check_model.py     # Utility to inspect model architecture
│       ├── pose_estimation.py # YOLO pose extraction
│       ├── predict_video.py   # Action prediction
│       └── train_model.py     # Model architecture definition
├── input/                     # Place videos here for processing
├── results/                   # Processed videos and results
├── scripts/
│   ├── __init__.py
│   ├── watch_directory.py     # Main pipeline script
│   ├── object_detection.py
│   └── audio_processing.py
└── processing.log
```

## Prerequisites

1. **Install Miniconda** (if not already installed):
   - Download from: https://docs.conda.io/en/latest/miniconda.html
   - For MacOS:
     ```bash
     # Make installer executable
     chmod +x ~/Downloads/Miniconda3-latest-MacOS-*.sh
     # Run installer
     ~/Downloads/Miniconda3-latest-MacOS-*.sh
     # Restart terminal after installation
     ```

2. **Install Homebrew** (for MacOS users):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

## Installation

1. **Create and activate conda environment:**
   ```bash
   conda create -n videopipeline python=3.10
   conda activate videopipeline
   ```

2. **Install PyTorch and related packages:**
   ```bash
   conda install pytorch torchvision torchaudio -c pytorch
   ```

3. **Install YOLO and video processing:**
   ```bash
   conda install -c conda-forge ultralytics opencv moviepy
   ```

4. **Install OpenCV with required dependencies:**
   ```bash
   conda install -c conda-forge opencv lapack openblas
   ```

5. **Install audio processing:**
   ```bash
   conda install -c conda-forge speechrecognition pydub
   pip install "deep-translator[languages]"
   ```

6. **Install other dependencies:**
   ```bash
   pip install watchdog langdetect
   ```

7. **Install ffmpeg:**
   
   MacOS:
   ```bash
   brew install ffmpeg
   ```
   
   Windows:
   ```bash
   choco install ffmpeg
   ```
   
   Linux:
   ```bash
   sudo apt-get install ffmpeg
   ```

## Setup

1. **Clone repository:**
   ```bash
   git clone <repository-url>
   cd pipeline
   ```

2. **Create directories:**
   ```bash
   mkdir -p input results action_recognition/models
   ```

3. **Verify model file:**
   - Ensure `model_20241220_145338.pth` is in `action_recognition/models/`
   - This model supports 101 action classes including:
     - WritingOnBoard
     - PlayingPiano
     - Basketball
     - And many more

## Usage

1. **Start the pipeline:**
   ```bash
   python -m scripts.watch_directory
   ```

2. **Process videos:**
   - Place video files in `input/` directory
   - Supported formats: .mp4, .avi, .mov
   - Files are processed automatically
   - Results appear in `results/` directory

## Output Format

The pipeline provides:
1. **Action Recognition:**
   - Top 3 predicted actions with confidence scores
   - Example: "WritingOnBoard: 62.4%"

2. **Object Detection:**
   - List of detected objects (person, chair, laptop, etc.)
   - Confidence scores for each detection

3. **Audio Processing:**
   - Transcription of speech (if present)
   - Translation to English (for non-English speech)

## Troubleshooting

1. **Model Loading Issues:**
   - Run `python -m action_recognition.scripts.check_model` to verify model architecture
   - Ensure model file exists in correct location

2. **CUDA/GPU Errors:**
   - Pipeline automatically uses CPU if CUDA unavailable
   - For GPU support, install CUDA toolkit

3. **Permission Issues:**
   - For ffmpeg on MacOS: `sudo chown -R $(whoami) /usr/local/var/homebrew`
   - For directory access: Ensure write permissions in input/results folders

4. **Import Errors:**
   - Verify conda environment is activated
   - Check all dependencies are installed
   - Ensure correct directory structure

## Dependencies List

Core dependencies:
- Python 3.10+
- PyTorch & torchvision
- ultralytics (YOLOv8)
- opencv-python with lapack/openblas
- moviepy
- SpeechRecognition
- pydub
- deep-translator
- watchdog
- langdetect
- ffmpeg

## Version Information
Current version: 0.1.0
