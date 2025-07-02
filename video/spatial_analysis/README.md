# Video to Philosophy Pipeline

A streamlined AI-powered pipeline that analyzes videos and generates spatial-philosophical interpretations using a vision-language model and fine-tuned language model. I created and ran this locally, and it pushed my new Macbook Air (M4 chip) CPU to its limit. 

## Overview

This pipeline combines two AI models to create philosophical interpretations of video content:

1. **Kimi-VL-A3B-Thinking**: A vision-language model that analyzes video frames and generates detailed descriptions
2. **Microsoft Phi-2 with LoRA Adapters**: Fine-tuned language models that generate philosophical interpretations based on the visual descriptions

The pipeline is specifically designed for analyzing spatial practices and generating critical spatial theory interpretations.

## Quick Start

### Prerequisites

- Python 3.8+
- Required Python packages (see `requirements.txt`)
- Kimi-VL-A3B-Thinking model downloaded to `models/kimi_vl_a3b_thinking/`
- Microsoft Phi-2 model downloaded to `models/phi/`
- Fine-tuned LoRA adapters in `models/phi_lora_adapter_*/`

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd video_analysis
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models:
   - **Kimi-VL-A3B-Thinking**: Download from [Hugging Face](https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking) to `models/kimi_vl_a3b_thinking/`
   - **Microsoft Phi-2**: Download from [Hugging Face](https://huggingface.co/microsoft/phi-2) to `models/phi/`
   - **LoRA Adapters**: Ensure fine-tuned adapters are in `models/phi_lora_adapter_*/`

### Usage

#### Simple Pipeline (Recommended)

Run the complete pipeline with a single command:

```bash
python video_to_philosophy.py <video_file>
```

Example:
```bash
python video_to_philosophy.py test04b.mov
```

This will:
1. Analyze video frames using Kimi-VL-A3B-Thinking
2. Generate philosophical interpretations using multiple LoRA adapters
3. Create a final one-paragraph spatial-philosophical interpretation
4. Save all results to the `results/` directory

#### Individual Components

You can also run individual components:

**Video Analysis Only:**
```bash
python video_analysis.py
```

**Philosophical Analysis Only:**
```bash
python philosophical_pipeline.py <image_or_video> --adapter ah
```

## File Structure

### Core Pipeline Files

- **`video_to_philosophy.py`**: Main pipeline script that orchestrates the entire workflow
- **`video_analysis.py`**: Video frame analysis using Kimi-VL-A3B-Thinking
- **`philosophical_pipeline.py`**: Philosophical interpretation using fine-tuned Phi models
- **`spatial_philosophical_interpretation.py`**: Final interpretation generation

### Model Directories

- **`models/kimi_vl_a3b_thinking/`**: Kimi-VL-A3B-Thinking vision-language model
- **`models/phi/`**: Microsoft Phi-2 base model
- **`models/phi_lora_adapter_*/`**: Fine-tuned LoRA adapters (aa, ab_1hr, ac, ad, ae, af, ag, ah)

### Output Directories

- **`results/`**: Pipeline outputs (frame analyses, philosophical results, interpretations)
- **`analysis_results/`**: Detailed analysis files from individual components

## Pipeline Workflow

### Step 1: Video Frame Analysis
- **Input**: Video file (MP4, MOV, AVI, etc.)
- **Process**: Extract frames and analyze each using Kimi-VL-A3B-Thinking
- **Output**: Detailed descriptions of visual content, objects, actions, and relationships

### Step 2: Philosophical Interpretation
- **Input**: Combined frame descriptions
- **Process**: Generate philosophical interpretations using multiple LoRA adapters
- **Output**: Diverse philosophical perspectives on the visual content

### Step 3: Final Interpretation
- **Input**: Frame analyses and philosophical results
- **Process**: Synthesize into a coherent one-paragraph interpretation
- **Output**: Final spatial-philosophical interpretation

## Model Details

### Kimi-VL-A3B-Thinking
- **Purpose**: Vision-language analysis of video frames
- **Capabilities**: Detailed visual description, object recognition, spatial relationships
- **Output**: Rich textual descriptions suitable for philosophical analysis

### Microsoft Phi-2 with LoRA Adapters
- **Base Model**: Microsoft Phi-2 (2.7B parameters)
- **Fine-tuning**: LoRA adapters trained on critical spatial practice texts
- **Adapters**: 8 different adapters (aa through ah) representing different philosophical perspectives
- **Output**: Philosophical interpretations focused on spatial theory and practice

## Configuration

### Video Analysis Settings
- **Default Duration**: 4 seconds of video analyzed
- **Frame Rate**: Analyzes frames at video's native FPS
- **Image Resolution**: Resized to 512x512 for model processing

### Philosophical Analysis Settings
- **Default Adapters**: All 8 adapters (aa, ab_1hr, ac, ad, ae, af, ag, ah)
- **Temperature**: 0.8 for creative interpretation
- **Max Tokens**: 500 for philosophical responses

## Output Files

### Pipeline Results
- **`{video_name}_frame_analyses_{timestamp}.json`**: Raw frame analysis data
- **`{video_name}_philosophical_{timestamp}.json`**: Philosophical interpretation results
- **`{video_name}_interpretation_{timestamp}.txt`**: Final one-paragraph interpretation

### Example Output
```
VIDEO TO PHILOSOPHY PIPELINE
================================================================================
Input video: test04b.mov
Timestamp: 2025-06-27 10:30:15

Step 1: Analyzing video frames with vision-language model...
✓ Analyzed 120 frames

Step 2: Running spatial-philosophical analysis...
✓ Generated philosophical interpretations

Step 3: Creating final interpretation...
✓ Final interpretation complete

================================================================================
FINAL SPATIAL-PHILOSOPHICAL INTERPRETATION
================================================================================
The video depicts a person lying on the ground in a natural, grassy environment, 
covered with a dark cloth or blanket, with a dog present nearby, over a duration 
of 22 seconds. The fine-tuned Phi model's analysis reveals a consistent focus on 
themes of vulnerability and human-animal relationships rather than temporal 
duration...
================================================================================
```

## Troubleshooting

### Common Issues

**Model Loading Errors:**
- Ensure models are downloaded to correct directories
- Check file permissions and disk space
- Verify model file integrity

**Memory Issues:**
- Pipeline uses CPU-only processing to avoid GPU memory issues
- Reduce `max_seconds` parameter in `video_analysis.py` if needed

**Video Format Issues:**
- Supported formats: MP4, MOV, AVI, MKV
- Ensure video file is not corrupted
- Check video codec compatibility

### Performance Tips

- **Shorter Videos**: Analyze 2-4 seconds for faster processing
- **Lower Resolution**: Videos are automatically resized to 512x512
- **CPU Processing**: Pipeline is optimized for CPU-only operation

## Advanced Usage

### Custom Adapter Selection
```python
from philosophical_pipeline import run_spatial_philosophical_analysis

# Use only specific adapters
results = run_spatial_philosophical_analysis(
    frame_analyses, 
    adapter_names=['ah', 'ag', 'af']
)
```

### Custom Video Analysis
```python
from video_analysis import analyze_video_frames

# Analyze more seconds
frame_analyses = analyze_video_frames(
    video_path, 
    max_seconds=8
)
```

### Custom Interpretation
```python
from spatial_philosophical_interpretation import create_interpretation

# Generate custom interpretation
interpretation = create_interpretation(
    frame_analyses, 
    philosophical_results
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{video_to_philosophy_pipeline,
  title={Video to Philosophy Pipeline: AI-powered spatial-philosophical analysis},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/video_analysis}
}
```

## Support

For questions and support:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the example outputs in the `results/` directory 
