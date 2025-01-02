# Cross-Cultural AI
Funded by InnovateUK, 2024-25

Cross-cultural AI automatically translates & analyses user-generated video to help generate culturally-specific insights. As filmmakers, we increasingly draw on crowdsourcing to generate content,ethnography to relate these hints and glimpses of life to the issues & indexes that matter, and data visualisation to draw it all together -- while keeping people's stories at the centre. 

AI can recognise objects, people, actions; culturally-specific datasets plus crowdsourcing mean that creative practitioners could work more easily with participants globally. While our work focuses on climate change, the AI-driven platform we're developing could be applied to ranges of content, contexts, projects & products.

Our approach combines quality of insights derived from user-generated content, with quantity coming with scale: Our Nine Earths project (http://nine-earths.net) drew from hundreds of hours of video from 12 countries. AI can now help translate languages, identify relevant content & recognise patterns to say something meaningful, especially when combined with participants' input and verification.

# Video Analysis pipeline

A comprehensive video analysis pipeline that performs:
- Action recognition using YOLO pose estimation (101 action classes)
- Object detection using YOLOv8
- Audio transcription and translation with automatic detection of over 100 languages
- This work in progress. As a next step, participants re-train the model through feedback.

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
