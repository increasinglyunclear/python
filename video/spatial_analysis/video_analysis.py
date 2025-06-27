import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import json
import os
from datetime import datetime
import gc

def analyze_video_frames(video_path, model_path="models/kimi_vl_a3b_thinking", max_seconds=4):
    """
    Analyze video frames using Kimi-VL-A3B-Thinking model and return frame analyses.
    
    This function is the core video analysis component that:
    1. Loads a video file and extracts frames
    2. Uses the Kimi-VL-A3B-Thinking vision-language model to analyze each frame
    3. Returns structured analysis results for further processing
    
    Args:
        video_path (str): Path to the input video file
        model_path (str): Path to the Kimi-VL-A3B-Thinking model
        max_seconds (int): Maximum number of seconds to analyze (default: 4)
    
    Returns:
        list: List of frame analysis dictionaries with timestamp and analysis text
    """
    # Set offline mode for transformers to prevent internet downloads
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Set device to CPU for compatibility
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    
    # Calculate frames to analyze
    frames_to_analyze = int(max_seconds * fps)
    
    # Load the Kimi-VL-A3B-Thinking model
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
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("\nPlease ensure the model is downloaded to 'models/kimi_vl_a3b_thinking'")
        return None
    
    # Initialize results list
    frame_analyses = []
    
    # Process frames
    frame_count = 0
    while frame_count < frames_to_analyze:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert frame to RGB and resize to reduce memory usage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (512, 512))  # Resize to a reasonable size
        pil_image = Image.fromarray(frame_rgb)
        
        # Prepare chat template input for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": "Analyze this image in detail. What do you see? Consider the objects, actions, and their relationships. Provide a philosophical reflection on what you observe."}
                ],
            },
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = processor(images=[pil_image], text=text, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Generate analysis
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=200, temperature=0.6)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        analysis = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Store results for this frame
        frame_result = {
            "frame_number": frame_count,
            "timestamp": frame_count / fps,
            "analysis": analysis
        }
        frame_analyses.append(frame_result)
        
        # Print progress
        if frame_count % 10 == 0:
            print(f"Processed frame {frame_count}/{frames_to_analyze}")
            print(f"Sample analysis: {analysis[:200]}...")
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        frame_count += 1
    
    # Release the video capture
    cap.release()
    
    print(f"✓ Analyzed {len(frame_analyses)} frames")
    return frame_analyses

def analyze_video(video_path, output_dir="analysis_results", model_path="models/kimi_vl_a3b_thinking"):
    """
    Analyze video frames and save results to files.
    
    This function provides a complete video analysis workflow that:
    1. Analyzes video frames using Kimi-VL-A3B-Thinking
    2. Saves results to JSON and text files
    3. Creates combined analysis summaries
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save analysis results
        model_path (str): Path to the Kimi-VL-A3B-Thinking model
    """
    # Set offline mode for transformers
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set device to CPU only
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    
    # Initialize the Kimi-VL model and processor
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
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("\nPlease download the model first using:")
        print("1. Visit https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking")
        print("2. Download the model files")
        print("3. Place them in the 'models/kimi_vl_a3b_thinking' directory")
        return
    
    # Calculate frames to analyze (4 seconds worth)
    frames_to_analyze = int(4 * fps)
    
    # Initialize results dictionary
    results = {
        "video_info": {
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "frames_analyzed": frames_to_analyze
        },
        "frame_analysis": []
    }
    
    # Process frames
    frame_count = 0
    while frame_count < frames_to_analyze:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert frame to RGB and resize to reduce memory usage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (512, 512))  # Resize to a reasonable size
        pil_image = Image.fromarray(frame_rgb)
        
        # Prepare chat template input for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": "Analyze this image in detail. What do you see? Consider the objects, actions, and their relationships. Provide a philosophical reflection on what you observe."}
                ],
            },
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = processor(images=[pil_image], text=text, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=200, temperature=0.6)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        analysis = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Store results for this frame
        frame_result = {
            "frame_number": frame_count,
            "timestamp": frame_count / fps,
            "analysis": analysis
        }
        results["frame_analysis"].append(frame_result)
        
        # Print progress
        if frame_count % 10 == 0:
            print(f"Processed frame {frame_count}/{frames_to_analyze}")
            print(f"Sample analysis: {analysis[:200]}...")
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        frame_count += 1
    
    # Release the video capture
    cap.release()
    
    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"video_analysis_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nAnalysis complete. Results saved to: {output_file}")
    
    # Extract all analyses into a text file
    analyses = [frame["analysis"] for frame in results["frame_analysis"]]
    with open("frame_analyses.txt", "w") as f:
        f.write("\n\n".join(analyses))
    print("\nAll frame analyses saved to frame_analyses.txt")
    
    # Combine analyses into a single coherent narrative
    combined_analysis = " ".join(analyses)
    words = combined_analysis.split()
    if len(words) > 200:
        combined_analysis = " ".join(words[:200]) + "..."
    with open("combined_analysis.txt", "w") as f:
        f.write(combined_analysis)
    print("\nCombined analysis saved to combined_analysis.txt")

# Main function to analyze the first frame of a video using Kimi-VL-A3B-Thinking
# and save the result to a text file.
def analyze_first_frame(video_path, output_dir="analysis_results", model_path="/Users/kevin/Desktop/DP/philosophy model 002/video_analysis/models/kimi_vl_a3b_thinking"):
    """
    Analyze only the first frame of a video.
    
    This function is useful for quick testing or when only the first frame
    is needed for analysis.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save analysis results
        model_path (str): Path to the Kimi-VL-A3B-Thinking model
    """
    # Force offline mode for Hugging Face Transformers (no internet/model download)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Set device to CPU for compatibility
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame")
        return

    # Convert frame to RGB and resize for the model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (512, 512))
    pil_image = Image.fromarray(frame_rgb)

    # Load the processor and model from the local directory
    print("Loading Kimi-VL-A3B-Thinking model...")
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

    # Prepare the prompt and input for the model using the chat template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": "Analyze this image in detail. What do you see? Consider the objects, actions, and their relationships. Provide a philosophical reflection on what you observe."}
            ],
        },
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = processor(images=[pil_image], text=text, return_tensors="pt", padding=True, truncation=True).to(device)

    # Generate the analysis
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=200, temperature=0.6)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    analysis = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the analysis to a text file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"first_frame_analysis_{timestamp}.txt")
    with open(output_file, 'w') as f:
        f.write("First Frame Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write(analysis)
    print(f"Analysis saved to: {output_file}")

    # Release the video capture
    cap.release()

if __name__ == "__main__":
    # Example usage
    video_file = "test04b.mov"
    if os.path.exists(video_file):
        print("Analyzing video frames...")
        frame_analyses = analyze_video_frames(video_file)
        if frame_analyses:
            print(f"Successfully analyzed {len(frame_analyses)} frames")
    else:
        print(f"Video file {video_file} not found") 