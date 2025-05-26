# Ollama LLM Batch Philosophical Analysis

This directory contains a script to generate deep philosophical reflections on image descriptions using the Llama-3 8B model via [Ollama](https://ollama.com/).

## Requirements

- **Ollama** installed and running on your machine ([download here](https://ollama.com/download))
- Llama-3 8B model pulled via Ollama (`ollama pull llama3`)
- Python 3.8+
- Python packages: `requests`
- VLM (Vision-Language Model) output files in `../analysis_results/` (e.g., `test01_vlm.txt`)

## Setup

1. **Install Ollama**
   - Download and install from [https://ollama.com/download](https://ollama.com/download)

2. **Pull the Llama-3 8B model**
   ```sh
   ollama pull llama3
   ```

3. **Start the Ollama server**
   ```sh
   ollama serve
   ```
   (Keep this running in a terminal window)

4. **Install Python dependencies**
   ```sh
   pip install requests
   ```

5. **Generate VLM outputs**
   - Use your image analysis pipeline to create `*_vlm.txt` files in `../analysis_results/`.

## Usage

From the main project directory, run:

```sh
python ollama_llm_analysis/ollama_philosophy_batch.py
```

- The script will read all `*_vlm.txt` files in `analysis_results/`.
- For each file, it will send a prompt to the Llama-3 8B model via Ollama, asking for a deep philosophical reflection.
- Each output will be saved as `<image>_philosophy_ollama.txt` in this directory.

## Output
- Example: `test01_philosophy_ollama.txt`

## Troubleshooting
- **No VLM output files found:** Make sure you have run the VLM step and have `*_vlm.txt` files in `analysis_results/`.
- **Ollama connection errors:** Ensure `ollama serve` is running and accessible at `http://localhost:11434`.
- **Model not found:** Make sure you have pulled the model with `ollama pull llama3`.

## Customization
- You can change the model by editing the `ollama_model` variable in the script.
- You can adjust the prompt for different philosophical styles or depth.

---

For more information on Ollama, see [https://ollama.com/](https://ollama.com/) 