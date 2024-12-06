"""
UCF101 Dataset Processing Script

This script processes the UCF101 video dataset, extracting pose data from each video
using the PoseEstimator class.

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

# Add the scripts directory to the Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from pose_estimation import PoseEstimator

def process_ucf101_videos():
    """
    Process all videos in the UCF101 dataset and extract pose data.
    Saves the results as JSON files in the poses directory.
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

    logger.info(f"Found {len(video_files)} videos to process")

    # Process each video
    results = {"successful": [], "failed": []}
    
    for i, video_path in enumerate(video_files, 1):
        try:
            # Create output path preserving the class directory structure
            rel_path = video_path.relative_to(ucf101_dir)
            output_path = output_dir / f"{rel_path.stem}_poses.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"\nProcessing [{i}/{len(video_files)}]: {video_path.name}")

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
        with open(log_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nDetailed results saved to: {log_file}")
    except Exception as e:
        logger.error(f"Failed to save results log: {e}")

if __name__ == "__main__":
    process_ucf101_videos()
