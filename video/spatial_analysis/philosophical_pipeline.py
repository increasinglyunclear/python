import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from peft import PeftModel
import json
import os
from datetime import datetime
import gc
import argparse

def run_spatial_philosophical_analysis(frame_analyses, adapter_names=None):
    """
    Run spatial-philosophical analysis on video frame analyses using multiple LoRA adapters.
    
    This function takes frame analyses from video analysis and processes them through
    multiple fine-tuned Phi model adapters to generate diverse philosophical interpretations.
    
    Args:
        frame_analyses (list): List of frame analysis dictionaries from video analysis
        adapter_names (list): List of adapter names to use (default: ['aa', 'ab_1hr', 'ac', 'ad', 'ae', 'af', 'ag', 'ah'])
    
    Returns:
        dict: Dictionary containing philosophical analysis results from all adapters
    """
    if adapter_names is None:
        adapter_names = ['aa', 'ab_1hr', 'ac', 'ad', 'ae', 'af', 'ag', 'ah']
    
    print("=" * 80)
    print("SPATIAL-PHILOSOPHICAL ANALYSIS")
    print("=" * 80)
    
    # Combine all frame analyses into a single description
    print("Combining frame analyses...")
    combined_description = combine_frame_analyses(frame_analyses)
    
    # Generate philosophical interpretations using each adapter
    results = {
        "timestamp": datetime.now().isoformat(),
        "frame_count": len(frame_analyses),
        "adapter_analyses": {},
        "merged_analysis": ""
    }
    
    for adapter_name in adapter_names:
        print(f"\nAnalyzing with adapter: {adapter_name}")
        try:
            interpretation = get_philosophical_interpretation(combined_description, adapter_name)
            results["adapter_analyses"][adapter_name] = interpretation
            print(f"✓ Completed analysis with {adapter_name}")
        except Exception as e:
            print(f"✗ Error with adapter {adapter_name}: {e}")
            results["adapter_analyses"][adapter_name] = f"Error: {str(e)}"
    
    # Create merged analysis
    print("\nCreating merged analysis...")
    results["merged_analysis"] = create_merged_analysis(results["adapter_analyses"])
    
    print("✓ Spatial-philosophical analysis complete!")
    return results

def combine_frame_analyses(frame_analyses):
    """
    Combine multiple frame analyses into a single comprehensive description.
    
    Args:
        frame_analyses (list): List of frame analysis dictionaries
    
    Returns:
        str: Combined description for philosophical analysis
    """
    # Extract analysis texts
    analyses = [frame['analysis'] for frame in frame_analyses]
    
    # Combine into a single description
    combined = " ".join(analyses)
    
    # Truncate if too long (to fit within model context)
    if len(combined) > 2000:
        combined = combined[:2000] + "..."
    
    return combined

def create_merged_analysis(adapter_analyses):
    """
    Create a merged analysis from multiple adapter interpretations.
    
    Args:
        adapter_analyses (dict): Dictionary of analyses from different adapters
    
    Returns:
        str: Merged philosophical analysis
    """
    # Filter out error messages
    valid_analyses = [analysis for analysis in adapter_analyses.values() 
                     if not analysis.startswith("Error:")]
    
    if not valid_analyses:
        return "No valid analyses available."
    
    # Simple concatenation with separators
    merged = "\n\n---\n\n".join(valid_analyses)
    
    return merged

def analyze_image_with_kimi(image_path, model_path="models/kimi_vl_a3b_thinking"):
    """
    Analyze an image using Kimi-VL-A3B-Thinking model.
    
    This function provides detailed visual analysis of images or video frames
    using the Kimi-VL-A3B-Thinking vision-language model.
    
    Args:
        image_path (str): Path to image or video file
        model_path (str): Path to Kimi-VL-A3B-Thinking model
    
    Returns:
        str: Detailed visual analysis of the image
    """
    # Set offline mode for transformers
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    # Set device to CPU
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load the image
    if image_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # If it's a video, extract the first frame
        cap = cv2.VideoCapture(image_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return None
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read the first frame")
            return None
        
        cap.release()
        
        # Convert frame to RGB and resize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (512, 512))
        pil_image = Image.fromarray(frame_rgb)
    else:
        # Load image directly
        pil_image = Image.open(image_path).convert('RGB')
        pil_image = pil_image.resize((512, 512))
    
    # Load the Kimi-VL model and processor
    print("Loading Kimi-VL-A3B-Thinking model...")
    try:
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True
        )
        model = model.to(device)
        model.eval()
        print("Kimi-VL model loaded successfully!")
    except Exception as e:
        print(f"Error loading Kimi-VL model: {str(e)}")
        return None
    
    # Prepare the prompt for detailed analysis
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": "Analyze this image in detail. Describe what you see, including objects, people, actions, colors, composition, and any symbolic or metaphorical elements. Consider the visual narrative and what it might represent. Provide a comprehensive description that could be used for philosophical analysis."}
            ],
        },
    ]
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = processor(images=[pil_image], text=text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Generate analysis
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=300, temperature=0.7)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    analysis = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return analysis

def get_philosophical_interpretation(description, adapter_name="ah"):
    """
    Generate philosophical interpretation using the fine-tuned Phi model.
    
    This function uses a specific LoRA adapter to generate philosophical
    interpretations of visual descriptions.
    
    Args:
        description (str): Visual description to analyze
        adapter_name (str): Name of the LoRA adapter to use
    
    Returns:
        str: Philosophical interpretation
    """
    # Force CPU usage
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = torch.device("cpu")
    
    # Load base model and tokenizer
    print(f"Loading Phi model with adapter {adapter_name}...")
    base_model_path = "models/phi"
    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the adapter
    adapter_path = f"models/phi_lora_adapter_{adapter_name}"
    model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    
    # Create a philosophical prompt based on the description
    philosophical_prompt = f"""Based on this visual description, provide a deep philosophical interpretation:

Description: {description}

Please analyze this from a philosophical perspective, considering:
- The nature of perception and reality
- Symbolic and metaphorical meanings
- Questions of existence, consciousness, or human condition
- Ethical or moral implications
- Aesthetic and artistic significance
- The relationship between the visual and the conceptual

Provide a thoughtful philosophical reflection:"""
    
    # Tokenize and generate
    inputs = tokenizer(philosophical_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=500,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove the prompt)
    if philosophical_prompt in response:
        response = response.split(philosophical_prompt)[-1].strip()
    
    return response

def run_philosophical_pipeline(image_path, adapter_name="ah", output_file=None):
    """
    Run the complete philosophical analysis pipeline for a single image/video.
    
    This function provides a complete workflow for analyzing a single image or
    video frame and generating philosophical interpretations.
    
    Args:
        image_path (str): Path to image or video file
        adapter_name (str): Name of the LoRA adapter to use
        output_file (str): Path to save results (optional)
    
    Returns:
        dict: Analysis results
    """
    print("=" * 80)
    print("PHILOSOPHICAL ANALYSIS PIPELINE")
    print("=" * 80)
    
    # Step 1: Analyze image/video with Kimi-VL
    print("\nStep 1: Analyzing image/video with Kimi-VL-A3B-Thinking...")
    visual_description = analyze_image_with_kimi(image_path)
    
    if visual_description is None:
        print("Failed to analyze image/video. Exiting.")
        return
    
    print(f"\nVisual Description:\n{visual_description}")
    
    # Step 2: Generate philosophical interpretation
    print("\n" + "=" * 80)
    print("Step 2: Generating philosophical interpretation...")
    philosophical_interpretation = get_philosophical_interpretation(visual_description, adapter_name)
    
    print(f"\nPhilosophical Interpretation:\n{philosophical_interpretation}")
    
    # Step 3: Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "input_file": image_path,
        "adapter_used": adapter_name,
        "visual_description": visual_description,
        "philosophical_interpretation": philosophical_interpretation
    }
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"philosophical_analysis_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "=" * 80)
    print(f"Analysis complete! Results saved to: {output_file}")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    # Example usage for single image/video analysis
    parser = argparse.ArgumentParser(description="Run philosophical analysis on image or video")
    parser.add_argument("input_file", help="Path to image or video file")
    parser.add_argument("--adapter", default="ah", help="LoRA adapter to use")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    run_philosophical_pipeline(args.input_file, args.adapter, args.output) 