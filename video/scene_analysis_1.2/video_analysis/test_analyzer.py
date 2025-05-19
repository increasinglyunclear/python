import logging
from pathlib import Path
from video_analysis.video_analyzer import EnhancedVideoAnalyzer
import json
import os
import numpy as np
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def to_serializable(obj):
    """Recursively convert numpy arrays and torch tensors to lists for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    else:
        return obj

def save_results(results, output_path):
    """Save analysis results to a JSON file."""
    try:
        serializable_results = to_serializable(results)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def main():
    try:
        # Initialize the analyzer
        logger.info("Initializing video analyzer...")
        analyzer = EnhancedVideoAnalyzer()
        
        # Get video path
        video_path = Path("training_data/video.mov")
        abs_path = video_path.resolve()
        logger.info(f"Checking video file at: {abs_path}")
        logger.info(f"Path.exists(): {video_path.exists()}")
        logger.info(f"os.path.exists: {os.path.exists(str(video_path))}")
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found at {video_path} (absolute: {abs_path})")
        
        # Create output directory
        output_dir = Path("philosophy model 002/training_data/analysis_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analyze video
        logger.info(f"Starting analysis of {video_path}...")
        results = analyzer.analyze_video(str(video_path))
        
        # Save results
        output_path = output_dir / "video_analysis_results.json"
        save_results(results, output_path)
        
        # Print summary
        logger.info("\nAnalysis Summary:")
        logger.info("-" * 50)
        
        # Scene understanding summary
        scene_summary = results['scene_understanding']['summary']
        logger.info("\nScene Understanding:")
        logger.info(f"Primary Scene: {scene_summary['primary_scene']}")
        logger.info(f"Number of Scene Transitions: {scene_summary['scene_transitions']}")
        logger.info(f"Unique Scenes: {', '.join(scene_summary['unique_scenes'])}")
        
        # Object detection summary
        obj_summary = results['object_analysis']['summary']
        logger.info("\nObject Detection:")
        logger.info(f"Total Objects Detected: {obj_summary['total_objects_detected']}")
        logger.info(f"Unique Objects: {obj_summary['unique_objects']}")
        logger.info("Most Common Objects:")
        for obj, count in obj_summary['most_common_objects'].items():
            logger.info(f"  - {obj}: {count}")
        
        # Pose analysis summary
        pose_summary = results['pose_analysis']['summary']
        logger.info("\nPose Analysis:")
        logger.info(f"Total Frames: {pose_summary['total_frames']}")
        logger.info(f"Frames with Poses: {pose_summary['frames_with_poses']}")
        logger.info(f"Pose Detection Rate: {pose_summary['pose_detection_rate']:.2%}")
        
        # Temporal analysis summary
        temp_summary = results['temporal_analysis']['summary']
        logger.info("\nTemporal Analysis:")
        logger.info(f"Temporal Segments: {temp_summary['temporal_segments']}")
        logger.info(f"Average Feature Magnitude: {temp_summary['average_feature_magnitude']:.4f}")
        logger.info(f"Feature Standard Deviation: {temp_summary['feature_std_dev']:.4f}")
        
        # Audio analysis summary
        audio_summary = results['audio_analysis']['summary']
        logger.info("\nAudio Analysis:")
        logger.info(f"Audio Chunks: {audio_summary['audio_chunks']}")
        logger.info(f"Total Audio Energy: {audio_summary['total_audio_energy']:.2f}")
        logger.info(f"Average Chunk Energy: {audio_summary['average_chunk_energy']:.2f}")
        
        logger.info("\nAnalysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 