"""
CCAI video analysis using Google Video Intelligence
https://cloud.google.com/video-intelligence
Kevin Walker
21 Nov 2024

Requires Google Cloud account and API key

"""

from google.cloud import videointelligence
from pathlib import Path
import time

def analyze_video(video_path):
    """Analyze a video file for objects and labels."""
    
    print(f"Analyzing video: {video_path}")
    client = videointelligence.VideoIntelligenceServiceClient()
    
    # Read the video file
    with open(video_path, "rb") as file:
        input_content = file.read()
    
    # Configure the request
    features = [
        videointelligence.Feature.OBJECT_TRACKING,
        videointelligence.Feature.LABEL_DETECTION,
    ]
    
    request = {
        'input_content': input_content,
        'features': features,
    }
    
    print("Starting video analysis (this may take a few minutes)...")
    operation = client.annotate_video(request)
    
    print("Waiting for analysis to complete...")
    result = operation.result(timeout=180)
    
    # Write results to file
    output_file = Path(video_path).stem + "_analysis.txt"
    
    with open(output_file, 'w') as f:
        # Process object tracking results
        print("\nObject tracking results:")
        f.write("OBJECT TRACKING RESULTS\n")
        f.write("=====================\n\n")
        
        for annotation in result.annotation_results[0].object_annotations:
            confidence = annotation.confidence
            name = annotation.entity.description
            print(f"Found object: {name} (confidence: {confidence:.2f})")
            
            f.write(f"\nTracked object: {name}\n")
            f.write(f"Confidence: {confidence:.2f}\n")
            
            for frame in annotation.frames:
                time_offset = frame.time_offset.total_seconds()
                box = frame.normalized_bounding_box
                f.write(f"Time: {time_offset:.1f}s, Location: "
                       f"({box.left:.2f}, {box.top:.2f}, "
                       f"{box.right:.2f}, {box.bottom:.2f})\n")
    
    print(f"\nAnalysis complete! Results saved to {output_file}")

def main():
    # Test with a video file
    video_path = "testclip01.mov"  # Update this path
    analyze_video(video_path)

if __name__ == "__main__":
    main()
