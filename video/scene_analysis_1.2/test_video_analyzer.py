from video_analyzer import EnhancedVideoAnalyzer
import os
import cv2

def test_video_analysis():
    # Initialize the analyzer
    analyzer = EnhancedVideoAnalyzer()
    
    # Test video path - you'll need to provide a video file
    video_path = "test_video.mov"  # Changed to .mov
    
    if not os.path.exists(video_path):
        print(f"Please provide a video file at {video_path}")
        return
    
    # Verify video can be opened
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"\nVideo Properties:")
    print(f"FPS: {fps}")
    print(f"Total Frames: {frame_count}")
    print(f"Duration: {duration:.2f} seconds")
    
    cap.release()
    
    try:
        # Run analysis
        print("\nStarting video analysis...")
        results = analyzer.analyze_video(video_path)
        
        # Print results
        print("\nAnalysis Results:")
        print("----------------")
        
        # Scene understanding
        print("\nScene Understanding:")
        print(results['scene_understanding']['summary'])
        
        # Object detection
        print("\nObject Detection:")
        print(results['object_analysis']['summary'])
        
        # Pose analysis
        print("\nPose Analysis:")
        print(results['pose_analysis']['summary'])
        
        # Temporal analysis
        print("\nTemporal Analysis:")
        print(results['temporal_analysis']['summary'])
        
        # Audio analysis
        print("\nAudio Analysis:")
        print(results['audio_analysis']['summary'])
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        print("\nDetailed error:")
        print(traceback.format_exc())

if __name__ == "__main__":
    test_video_analysis() 