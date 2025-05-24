import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import json
import os
from datetime import datetime

def analyze_video(video_path, output_dir="analysis_results"):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
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
    processor = AutoProcessor.from_pretrained("kimi-vl/kimi-vl-a3b-thinking")
    model = AutoModelForVision2Seq.from_pretrained("kimi-vl/kimi-vl-a3b-thinking")
    model.eval()
    
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
            
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Prepare prompt for the model
        prompt = "Analyze this image in detail. What do you see? Consider the objects, actions, and their relationships. Provide a philosophical reflection on what you observe."
        
        # Process the image and generate analysis
        inputs = processor(images=pil_image, text=prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                num_beams=5,
                temperature=0.7,
                do_sample=True
            )
        
        # Decode the output
        analysis = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
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
    
    # Limit the final narrative to approximately 200 words
    words = combined_analysis.split()
    if len(words) > 200:
        combined_analysis = " ".join(words[:200]) + "..."
    
    with open("combined_analysis.txt", "w") as f:
        f.write(combined_analysis)
    print("\nCombined analysis saved to combined_analysis.txt")

if __name__ == "__main__":
    video_path = "test_data/video.mp4"  # Updated path to video file
    analyze_video(video_path) 