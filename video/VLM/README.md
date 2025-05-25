# Video Analysis with Kimi-VL-A3B-Thinking

This project demonstrates how to perform frame-level video analysis using the [Kimi-VL-A3B-Thinking](https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking) vision-language model. The script extracts the first frame from a video, generates a detailed, philosophically reflective description, and saves the result to a text file.

## Features
- Uses a state-of-the-art open-source vision-language model (Kimi-VL-A3B-Thinking)
- Processes video files and analyzes frames
- Outputs a natural language description for the first frame

## Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Create and activate a Python virtual environment (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the Kimi-VL-A3B-Thinking model files
- Visit the [Kimi-VL-A3B-Thinking model page](https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking)
- Download all files **except** the large `.safetensors` files if you do not plan to run locally, or download all files for full local inference
- Place them in `models/kimi_vl_a3b_thinking/`

**Note:** The large model files (`.safetensors`) are not included in this repository. You must download them manually due to their size.

### 5. Add a sample video
- Place a small video file named `video.mp4` in the `test_data/` directory (or update the script to use your own file)

## Usage

To analyze the first frame of the video and save the result:

```bash
python video_analysis.py
```

- The output will be saved to `single_frame_analysis.txt`.

## File Structure
- `video_analysis.py` — Main script for video analysis
- `models/kimi_vl_a3b_thinking/` — Model code and configuration (download from Hugging Face)
- `test_data/video.mp4` — Example video file (not included)
- `requirements.txt` — Python dependencies
- `.gitignore` — Excludes large, generated, and system files

## Notes
- This project is for research and demonstration purposes.
- For more information on the Kimi-VL model, see the [official documentation](https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking).

## License
MIT License (see model and code for details) 