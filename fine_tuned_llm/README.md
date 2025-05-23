# LLM Fine-tuning

This project fine-tunes Llama 2 for text analysis using domain-specific training data.

## Project Structure

```
philosophical_llm/
├── configs/
│   └── training_config.py    # Training configuration
├── data/                     # Processed data directory
├── models/                   # Fine-tuned models
├── scripts/
│   └── preprocess.py        # PDF text extraction and preprocessing
└── requirements.txt         # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Process the training data:
```bash
python scripts/preprocess.py
```

## Fine-tuning Approach

We use a domain-specific fine-tuning approach with the following key features:

1. **Base Model**: Llama 2 7B (optimized for M4 MacBook)
2. **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
3. **Training Data**: Processed texts
4. **Optimization**: Gradient accumulation and mixed precision training

## Configuration

The training configuration can be modified in `configs/training_config.py`. Key parameters include:

- Model size and architecture
- Training hyperparameters
- LoRA configuration
- Dataset settings

## Usage

1. Preprocess the data:
```bash
python scripts/preprocess.py
```

2. Fine-tune the model (script to be added):
```bash
python scripts/train.py
```

## Hardware Requirements

- Apple M4 chip
- 32GB RAM
- Sufficient disk space for model weights and training data

## Notes

- The model is optimized for text analysis
- Uses LoRA for efficient fine-tuning
- Supports long-form text generation and analysis 
