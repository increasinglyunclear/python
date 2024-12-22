"""
Main script that watches input directory for videos and processes them.
"""

import time
import logging
from pathlib import Path
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import csv
from datetime import datetime

# Import our processing modules
from .object_detection import detect_objects
from .audio_processing import transcribe_audio, translate_text
from action_recognition.scripts.predict_video import analyze_video

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('processing.log')
    ]
)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        # Set up paths relative to script location
        self.project_root = Path(__file__).parent.parent
        self.watch_dir = self.project_root / 'input'
        self.output_dir = self.project_root / 'results'
        
        # Create directories if they don't exist
        self.watch_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize CSV file with headers
        self.csv_path = self.output_dir / 'results.csv'
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp',
                    'Filename',
                    'Actions',
                    'Objects_Detected',
                    'Transcription',
                    'Translation'
                ])
        
        logger.info(f"Watching directory: {self.watch_dir}")
        logger.info(f"Results will be saved to: {self.output_dir}")
    
    def process_video(self, video_path: Path):
        """Process a single video file and log results."""
        logger.info(f"\nProcessing: {video_path.name}")
        logger.info("-" * 50)
        
        try:
            results = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'filename': video_path.name,
                'actions': None,
                'objects': None,
                'transcription': None,
                'translation': None
            }
            
            # 1. Action Recognition
            try:
                logger.info("Running action recognition...")
                model_dir = self.project_root / "action_recognition/models"
                model_files = list(model_dir.glob("model_*.pth"))
                if model_files:
                    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
                    actions = analyze_video(str(video_path), latest_model)
                    results['actions'] = actions
                    logger.info("Action recognition complete")
                else:
                    logger.warning("No action recognition model found")
            except Exception as e:
                logger.error(f"Action recognition failed: {str(e)}")
            
            # 2. Object Detection
            try:
                logger.info("Running object detection...")
                objects = detect_objects(str(video_path))
                results['objects'] = objects
                logger.info("Object detection complete")
            except Exception as e:
                logger.error(f"Object detection failed: {str(e)}")
            
            # 3. Audio Processing
            try:
                logger.info("Running audio transcription...")
                transcription = transcribe_audio(str(video_path))
                if transcription:
                    results['transcription'] = transcription
                    logger.info("Transcription complete")
                    
                    logger.info("Running translation...")
                    translation = translate_text(transcription)
                    results['translation'] = translation
                    logger.info("Translation complete")
            except Exception as e:
                logger.error(f"Audio processing failed: {str(e)}")
            
            # Write results to CSV
            self._write_to_csv(results)
            
            # Log results to terminal
            self._log_results(results)
            
            # Move processed file to output directory
            processed_path = self.output_dir / f"processed_{video_path.name}"
            video_path.rename(processed_path)
            logger.info(f"Moved processed video to: {processed_path}")
            
        except Exception as e:
            logger.error(f"Error processing {video_path.name}: {str(e)}")
    
    def _write_to_csv(self, results):
        """Write results to CSV file."""
        row = [
            results['timestamp'],
            results['filename']
        ]
        
        # Add action recognition results
        if results['actions']:
            actions_str = '; '.join([f"{action}: {conf:.1%}" for action, conf in results['actions']])
            row.append(actions_str)
        else:
            row.append('')
        
        # Add object detection results
        if results['objects']:
            row.append(', '.join(results['objects']))
        else:
            row.append('')
        
        # Add audio processing results
        row.extend([
            results.get('transcription', ''),
            results.get('translation', '')
        ])
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def _log_results(self, results):
        """Log results to terminal."""
        logger.info("\nResults Summary:")
        logger.info("-" * 50)
        
        if results.get('actions'):
            logger.info("\nTop Actions:")
            for action, conf in results['actions']:
                logger.info(f"  {action}: {conf:.1%}")
        
        if results.get('objects'):
            logger.info("\nObjects Detected:")
            logger.info(f"  {', '.join(results['objects'])}")
        
        if results.get('transcription'):
            logger.info("\nTranscription:")
            logger.info(f"  {results['transcription'][:200]}...")
            if results.get('translation'):
                logger.info("\nTranslation:")
                logger.info(f"  {results['translation'][:200]}...")
        
        logger.info("-" * 50)

class VideoHandler(FileSystemEventHandler):
    def __init__(self, processor: VideoProcessor):
        self.processor = processor
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        path = Path(event.src_path)
        if path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            # Wait a bit to ensure file is completely written
            time.sleep(1)
            self.processor.process_video(path)

def main():
    processor = VideoProcessor()
    event_handler = VideoHandler(processor)
    observer = Observer()
    observer.schedule(event_handler, str(processor.watch_dir), recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Stopping directory watch")
    
    observer.join()

if __name__ == "__main__":
    main()