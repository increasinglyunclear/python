from pathlib import Path
import sys
import os

# Add the scripts directory to the Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from pose_estimation import PoseEstimator
import json

def test_pose_estimator():
    # Initialize pose estimator
    print("Creating pose estimator...")
    pose_estimator = PoseEstimator()
    print("Pose Estimator initialized")

    # Get paths
    base_dir = Path(__file__).parent.parent
    video_path = base_dir / "input" / "testclip01.mp4"
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_poses.json"

    print(f"Testing with video: {video_path}")
    print(f"Output will be saved to: {output_path}")

    try:
        print("Starting video processing...")
        poses = pose_estimator.process_video(str(video_path), str(output_path))
        print(f"\nProcessing complete!")
        print(f"Processed video: {video_path}")
        print(f"Poses saved to: {output_path}")
        
        # Print some statistics
        if poses and isinstance(poses, list):
            total_frames = len(poses)
            frames_with_poses = sum(1 for frame in poses if frame["poses"])
            print(f"\nStatistics:")
            print(f"Total frames processed: {total_frames}")
            print(f"Frames with detected poses: {frames_with_poses}")
            if total_frames > 0:
                print(f"Detection rate: {frames_with_poses/total_frames*100:.1f}%")
            
            # Print example pose data from first frame with detections
            for frame in poses:
                if frame["poses"]:
                    print(f"\nExample pose data (frame {frame['frame']}):")
                    print(json.dumps(frame["poses"][0], indent=2))
                    break
        else:
            print("No pose data returned")
                
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pose_estimator()