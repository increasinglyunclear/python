import cv2
import json
from pathlib import Path
import logging
from spatial_analyzer import SpatialAnalyzer
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_single_frame():
    """Test the spatial analyzer on a single frame from the test video"""
    # Paths
    video_path = "test_data/video.mp4"
    knowledge_base_path = "test_data/processed_texts.json"
    output_path = "test_output/frame_analysis.json"
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize analyzer
        logger.info("Initializing Spatial Analyzer...")
        analyzer = SpatialAnalyzer(knowledge_base_path)
        
        # Open video
        logger.info(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Could not read frame from video")
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        frame_pil = Image.fromarray(frame_rgb)
        
        # Prepare frame data
        frame_data = {
            'frame': frame_pil,
            'timestamp': 0  # First frame timestamp
        }
        
        # Analyze frame
        logger.info("Analyzing frame...")
        analysis = analyzer.analyze_frame(frame_data)
        
        # Save results
        logger.info(f"Saving analysis to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        # Print results in a readable format
        print("\nAnalysis Results:")
        print("\nDetected Objects:")
        for obj in analysis.get('objects', []):
            print(f"- {obj['name']} (confidence: {obj['confidence']:.2f})")
        
        print("\nSpatial Relationships:")
        for rel in analysis.get('spatial_relationships', []):
            print(f"- {rel['object1']} is {rel['relationship']} {rel['object2']}")
        
        print("\nSpatial Practices:")
        for practice in analysis.get('spatial_practice', []):
            print(f"- {practice}")
        
        print("\nCritical Engagements:")
        for engagement in analysis.get('critical_engagement', []):
            print(f"- {engagement}")
        
        print("\nPoetic Observations:")
        for observation in analysis.get('poetic_observations', []):
            print(f"- {observation}")
        
        print("\nUrban Insights:")
        for insight in analysis.get('urban_insights', []):
            print(f"- {insight}")
            
        print("\nPhilosophical Insights:")
        print(analysis.get('philosophical_insights', ''))
        
        # Release video capture
        cap.release()
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    test_single_frame() 