"""
UCF101 Dataset Processing Script with Resume Capability

This script processes the UCF101 video dataset, extracting pose data from each video
using the PoseEstimator class. It can resume from previous interruptions by checking
existing processed files.

Dependencies:
    - All dependencies from pose_estimation.py
    - Properly structured UCF101 dataset in data/UCF101/videos/

Usage:
    python -m scripts.process_ucf101

Output:
    Generates JSON files containing pose data for each video in data/UCF101/poses/
"""

from pathlib import Path
import logging
import sys
import json

# Add the scripts directory to the Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from pose_estimation import PoseEstimator

def get_processed_videos(output_dir):
    """Get list of already processed videos"""
    processed = set()
    for json_file in output_dir.glob('**/*_poses.json'):
        # Remove '_poses.json' to get original video name
        video_name = json_file.stem[:-6]  # Remove '_poses' suffix
        processed.add(video_name)
    return processed

def process_ucf101_videos(resume=True):
    """
    Process all videos in the UCF101 dataset and extract pose data.
    Saves the results as JSON files in the poses directory.
    
    Args:
        resume (bool): If True, skip already processed videos
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize paths
    base_dir = Path(__file__).parent.parent
    ucf101_dir = base_dir / "data" / "UCF101" / "videos"
    output_dir = base_dir / "data" / "UCF101" / "poses"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify dataset directory exists
    if not ucf101_dir.exists():
        logger.error(f"UCF101 directory not found: {ucf101_dir}")
        logger.error("Please ensure the dataset is downloaded and placed in the correct location")
        return

    # Initialize pose estimator
    pose_estimator = PoseEstimator()
    logger.info("Pose Estimator initialized")

    # Get all video files
    video_extensions = ('.avi', '.mp4', '.mov')
    video_files = []
    for ext in video_extensions:
        video_files.extend(ucf101_dir.glob(f'**/*{ext}'))

    # Get already processed videos if resuming
    processed_videos = set()
    if resume:
        processed_videos = get_processed_videos(output_dir)
        logger.info(f"Found {len(processed_videos)} already processed videos")

    # Filter out already processed videos if resuming
    if resume:
        video_files = [v for v in video_files if v.stem not in processed_videos]
        logger.info(f"Remaining videos to process: {len(video_files)}")

    # Process each video
    results = {"successful": [], "failed": []}
    
    for i, video_path in enumerate(video_files, 1):
        try:
            # Create output path preserving the class directory structure
            rel_path = video_path.relative_to(ucf101_dir)
            output_path = output_dir / f"{rel_path.stem}_poses.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"\nProcessing [{i}/{len(video_files)}]: {video_path.name}")
            logger.info(f"Progress: {((i/len(video_files))*100):.2f}%")

            poses = pose_estimator.process_video(str(video_path), str(output_path))
            results["successful"].append(str(video_path))
            logger.info(f"Successfully processed: {video_path.name}")

        except Exception as e:
            logger.error(f"Failed to process {video_path.name}: {e}")
            results["failed"].append(str(video_path))

    # Print summary
    logger.info("\nProcessing complete!")
    logger.info(f"Successfully processed: {len(results['successful'])} videos")
    if results['successful']:
        logger.info("First few successful videos:")
        for video in results['successful'][:5]:  # Show only first 5
            logger.info(f" - {video}")
        if len(results['successful']) > 5:
            logger.info(f" ... and {len(results['successful'])-5} more")

    logger.info(f"\nFailed to process: {len(results['failed'])} videos")
    if results['failed']:
        logger.info("Failed videos:")
        for video in results['failed']:
            logger.info(f" - {video}")

    # Save processing results to a log file
    log_file = output_dir / "processing_results.json"
    try:
        # Load existing results if any
        existing_results = {"successful": [], "failed": []}
        if log_file.exists():
            with open(log_file, 'r') as f:
                existing_results = json.load(f)
        
        # Merge results
        combined_results = {
            "successful": existing_results["successful"] + results["successful"],
            "failed": existing_results["failed"] + results["failed"]
        }
        
        with open(log_file, 'w') as f:
            json.dump(combined_results, f, indent=2)
        logger.info(f"\nDetailed results saved to: {log_file}")
    except Exception as e:
        logger.error(f"Failed to save results log: {e}")

if __name__ == "__main__":
    process_ucf101_videos(resume=True)