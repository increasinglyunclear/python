import json
from pathlib import Path
from spatial_analyzer import SpatialAnalyzer
import logging
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_video_data(video_data_path: str) -> list:
    """Load the video analysis data"""
    with open(video_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded data keys: {list(data.keys())}")
    
    # Convert the data into the expected format
    frames = []
    
    # Handle different possible data structures
    if isinstance(data, list):
        # If data is already a list of frames
        frames_data = data
    elif isinstance(data, dict):
        # If data is a dictionary with frames
        frames_data = data.get('frames', [])
        if not frames_data:
            # Try to find frame data in other keys
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    frames_data = value
                    logger.info(f"Found frame data in key: {key}")
                    break
    
    logger.info(f"Found {len(frames_data)} frames")
    
    # Process each frame
    for i, frame in enumerate(frames_data):
        if i < 2:  # Log first two frames for debugging
            logger.info(f"Frame {i} keys: {list(frame.keys())}")
        
        frame_data = {
            'timestamp': frame.get('timestamp'),
            'objects': frame.get('objects', []),
            'scene': frame.get('scene', {}),
            'poses': frame.get('poses', []),
            'actions': frame.get('actions', [])
        }
        
        if i < 2:  # Log first two frames' processed data
            logger.info(f"Processed frame {i} data: {frame_data}")
        
        frames.append(frame_data)
    
    logger.info(f"Processed {len(frames)} frames")
    return frames

def save_analysis(analysis: dict, output_path: str):
    """Save the analysis results"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

def main():
    # Get the scene analysis path from command line or use default
    scene_analysis_path = sys.argv[1] if len(sys.argv) > 1 else "../../scene_analysis/video_analysis_results.json"
    
    # Convert to absolute path if needed
    if not os.path.isabs(scene_analysis_path):
        scene_analysis_path = os.path.abspath(scene_analysis_path)
    
    logger.info(f"Using scene analysis path: {scene_analysis_path}")
    
    # Other paths
    knowledge_base_path = "philosophy model 002/training_data/processed/processed_texts.json"
    output_path = "philosophy model 002/video_analysis/results/spatial_analysis.json"
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize analyzer
        logger.info("Initializing Spatial Analyzer...")
        analyzer = SpatialAnalyzer(knowledge_base_path)
        
        # Load video data
        logger.info(f"Loading video data from {scene_analysis_path}...")
        video_data = load_video_data(scene_analysis_path)
        
        if not video_data:
            logger.error("No video data found!")
            return
        
        # Analyze video
        logger.info("Analyzing video...")
        analysis = analyzer.analyze_video(video_data)
        
        # Save results
        logger.info("Saving analysis results...")
        save_analysis(analysis, output_path)
        
        logger.info(f"Analysis complete! Results saved to {output_path}")
        
        # Print sample insights
        print("\nSample Insights from Analysis:")
        
        print("\nSpatial Practices:")
        for frame in analysis['frame_analyses'][:2]:
            for insight in frame['spatial_practice'][:1]:
                print(f"- {insight}")
        
        print("\nCritical Engagements:")
        for frame in analysis['frame_analyses'][:2]:
            for insight in frame['critical_engagement'][:1]:
                print(f"- {insight}")
        
        print("\nPoetic Observations:")
        for frame in analysis['frame_analyses'][:2]:
            for insight in frame['poetic_observations'][:1]:
                print(f"- {insight}")
        
        print("\nTemporal Insights:")
        for insight in analysis['temporal_insights'][:2]:
            print(f"- {insight}")
        
        print("\nSpatial Narrative:")
        for insight in analysis['spatial_narrative']:
            print(f"- {insight}")
        
        print("\nCritical Reflections:")
        for insight in analysis['critical_reflections']:
            print(f"- {insight}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 