"""
Google Cloud Video Intelligence Analyzer
Kevin Walker
23 Nov 2024
--------------------------------------

Setup Steps:
1. Install Anaconda if not already installed
   Visit: https://www.anaconda.com/products/distribution

2. Initialize conda in terminal:
   source /opt/anaconda3/bin/activate

3. Create and activate conda environment:
   conda create -n videoanalyzer python=3.10
   conda activate videoanalyzer

4. Install required packages:
   conda install -c conda-forge google-cloud-videointelligence

5. Set up Google Cloud credentials:
   a. Go to Google Cloud Console (https://console.cloud.google.com)
   b. Create/Select a project
   c. Enable Video Intelligence API
   d. Create Service Account and download JSON key
   e. Move key to secure location:
      mkdir -p ~/.google
      mv ~/Downloads/your-credentials.json ~/.google/credentials.json

6. Set environment variable:
   Add to ~/.zshrc:
   export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.google/credentials.json"
   
7. Create directories:
   mkdir -p input output

8. Run script:
   python video_analyzer.py

Usage:
- Place video files in 'input' directory
- Script will automatically process new files
- Results are saved as CSVs in 'output' directory
- Processed videos are moved to 'output' directory
- Press Ctrl+C to stop the script

Supported video formats: .mp4, .avi, .mov, .MOV
"""

from google.cloud import videointelligence
from pathlib import Path
import time
import csv
import logging
from datetime import datetime

class VideoAnalyzer:
    def __init__(self, input_dir="input", output_dir="output"):
        # Setup directories
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create directories if they don't exist
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialize the client
        self.client = videointelligence.VideoIntelligenceServiceClient()
        print(f"Initialized Google Cloud Video Intelligence client")
        
    def analyze_video(self, video_path, file_number, total_files):
        """Analyze a video file for objects and labels."""
        
        print(f"\nAnalyzing video {file_number}/{total_files}: {video_path}")
        
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
        operation = self.client.annotate_video(request)
        
        print("Waiting for analysis to complete...")
        result = operation.result(timeout=180)
        
        # Write results to CSV file
        output_file = self.output_dir / f"{Path(video_path).stem}_analysis.csv"
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow([
                'timestamp',
                'object_name',
                'confidence',
                'left',
                'top',
                'right',
                'bottom'
            ])
            
            # Process object tracking results
            print("\nObject tracking results:")
            
            for annotation in result.annotation_results[0].object_annotations:
                confidence = annotation.confidence
                name = annotation.entity.description
                print(f"Found object: {name} (confidence: {confidence:.2f})")
                
                for frame in annotation.frames:
                    time_offset = frame.time_offset.total_seconds()
                    box = frame.normalized_bounding_box
                    
                    writer.writerow([
                        f"{time_offset:.2f}",
                        name,
                        f"{confidence:.2f}",
                        f"{box.left:.3f}",
                        f"{box.top:.3f}",
                        f"{box.right:.3f}",
                        f"{box.bottom:.3f}"
                    ])
        
        print(f"\nAnalysis complete! Results saved to {output_file}")
        return output_file

    def watch_directory(self):
        """Watch input directory for new video files"""
        print(f"Watching directory: {self.input_dir.absolute()}")
        print(f"Results will be saved to: {self.output_dir.absolute()}")
        print("Waiting for video files... (Press Ctrl+C to stop)")
        
        # Define supported video formats
        supported_formats = ['*.mp4', '*.avi', '*.mov', '*.MOV']
        
        while True:
            try:
                # Look for video files
                video_files = []
                for format in supported_formats:
                    video_files.extend(self.input_dir.glob(format))
                
                if video_files:
                    total_files = len(video_files)
                    print(f"\nFound {total_files} video files:")
                    for idx, video_path in enumerate(video_files, 1):
                        print(f"- {video_path}")
                        try:
                            logging.info(f"Processing file {idx}/{total_files}: {video_path}")
                            self.analyze_video(video_path, idx, total_files)
                            
                            # Move processed file to output directory
                            processed_path = self.output_dir / f"processed_{video_path.name}"
                            video_path.rename(processed_path)
                            logging.info(f"Moved processed video to: {processed_path}")
                            
                        except Exception as e:
                            logging.error(f"Error processing {video_path}: {str(e)}")
                
                # Wait before checking again
                print("\nChecking for new files...")
                time.sleep(5)  # Check every 5 seconds
                
            except KeyboardInterrupt:
                print("\nStopping video analyzer...")
                break
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                time.sleep(5)

def main():
    analyzer = VideoAnalyzer()
    analyzer.watch_directory()

if __name__ == "__main__":
    main()
