import torch
import cv2
import numpy as np
from pathlib import Path
import logging
import mediapipe as mp
from ultralytics import YOLO
import torchaudio
from torch import nn
import torch.nn.functional as F
import torchvision
from video_analysis.scene_understanding import SceneUnderstanding
import tempfile
import subprocess
import os
import soundfile as sf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self):
        """Initialize YOLO model for object detection."""
        try:
            # Use the pipeline's YOLOv8 model
            model_path = Path(__file__).parent.parent / 'pipeline' / 'yolov8n.pt'
            if not model_path.exists():
                logger.info("Downloading YOLOv8 model...")
                self.model = YOLO('yolov8n.pt')
            else:
                self.model = YOLO(str(model_path))
            logger.info("Object detection model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing object detection model: {str(e)}")
            raise

    def process_objects(self, video_path):
        """Detect and track objects in video."""
        try:
            cap = cv2.VideoCapture(video_path)
            object_features = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Run YOLO detection
                results = self.model(frame)
                
                # Extract object information
                frame_objects = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        obj_info = {
                            'class': result.names[int(box.cls)],
                            'confidence': float(box.conf),
                            'bbox': box.xyxy[0].tolist()
                        }
                        frame_objects.append(obj_info)
                
                object_features.append(frame_objects)
                
            cap.release()
            return object_features
            
        except Exception as e:
            logger.error(f"Error processing objects: {str(e)}")
            raise

class PoseEstimator:
    def __init__(self):
        """Initialize MediaPipe Pose."""
        try:
            self.mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("Pose estimation model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing pose estimation model: {str(e)}")
            raise

    def process_poses(self, video_path):
        """Estimate human poses in video."""
        try:
            cap = cv2.VideoCapture(video_path)
            pose_features = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                results = self.mp_pose.process(frame_rgb)
                
                if results.pose_landmarks:
                    # Extract pose landmarks
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z,
                            'visibility': landmark.visibility
                        })
                    pose_features.append(landmarks)
                else:
                    pose_features.append(None)
                    
            cap.release()
            return pose_features
            
        except Exception as e:
            logger.error(f"Error processing poses: {str(e)}")
            raise

class TemporalAnalyzer:
    def __init__(self):
        """Initialize 3D CNN for temporal analysis."""
        try:
            self.model = torchvision.models.video.r3d_18(pretrained=True)
            self.model.eval()
            logger.info("Temporal analysis model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing temporal analysis model: {str(e)}")
            raise

    def process_temporal(self, video_path):
        """Analyze temporal patterns in video."""
        try:
            cap = cv2.VideoCapture(video_path)
            temporal_features = []
            frame_buffer = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Preprocess frame
                frame = cv2.resize(frame, (112, 112))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame).float() / 255.0
                frame = frame.permute(2, 0, 1)
                
                frame_buffer.append(frame)
                
                # Process every 16 frames
                if len(frame_buffer) == 16:
                    # Stack frames
                    frames = torch.stack(frame_buffer)  # [time, channels, height, width]
                    frames = frames.permute(1, 0, 2, 3)  # [channels, time, height, width]
                    frames = frames.unsqueeze(0)  # [batch, channels, time, height, width]
                    
                    # Get temporal features
                    with torch.no_grad():
                        features = self.model(frames)
                    
                    temporal_features.append(features.numpy())
                    frame_buffer = []
                    
            cap.release()
            return np.array(temporal_features)
            
        except Exception as e:
            logger.error(f"Error processing temporal features: {str(e)}")
            raise

class AudioAnalyzer:
    def __init__(self):
        """Initialize audio processing components."""
        try:
            self.sample_rate = 16000
            self.n_fft = 400
            self.hop_length = 160
            logger.info("Audio analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing audio analyzer: {str(e)}")
            raise

    def process_audio(self, video_path):
        """Extract and analyze audio features."""
        try:
            # Extract audio from video using ffmpeg
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                tmp_wav_path = tmp_wav.name
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le', '-ar', str(self.sample_rate), '-ac', '1', tmp_wav_path
            ]
            subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            # Load audio with soundfile
            audio_np, sr = sf.read(tmp_wav_path)
            if audio_np.ndim == 1:
                audio_np = audio_np[None, :]  # [1, samples]
            else:
                audio_np = audio_np.T  # [channels, samples]
            audio_stream = torch.from_numpy(audio_np).float()
            audio_features = []
            
            # Process audio in chunks
            chunk_size = self.sample_rate * 5  # 5-second chunks
            for i in range(0, audio_stream.shape[1], chunk_size):
                chunk = audio_stream[:, i:i + chunk_size]
                
                # Compute spectrogram
                spec = torchaudio.transforms.Spectrogram(
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )(chunk)
                
                # Compute mel spectrogram
                mel_spec = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )(chunk)
                
                # Extract features
                features = {
                    'spectrogram': spec.numpy(),
                    'mel_spectrogram': mel_spec.numpy()
                }
                audio_features.append(features)
            
            # Clean up temp file
            os.remove(tmp_wav_path)
            return audio_features
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            raise

class EnhancedVideoAnalyzer:
    def __init__(self):
        """Initialize all analysis components."""
        try:
            self.scene_analyzer = SceneUnderstanding()
            self.object_detector = ObjectDetector()
            self.pose_estimator = PoseEstimator()
            self.temporal_analyzer = TemporalAnalyzer()
            self.audio_analyzer = AudioAnalyzer()
            logger.info("Video analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing video analyzer: {str(e)}")
            raise

    def analyze_video(self, video_path):
        """Analyze video using multiple streams and fuse the results."""
        try:
            logger.info(f"Starting analysis of video: {video_path}")
            
            # Process each stream
            scene_features = self.scene_analyzer.process_scenes(video_path)
            object_features = self.object_detector.process_objects(video_path)
            pose_features = self.pose_estimator.process_poses(video_path)
            temporal_features = self.temporal_analyzer.process_temporal(video_path)
            audio_features = self.audio_analyzer.process_audio(video_path)
            
            # Fuse features
            analysis_results = self.fuse_features(
                scene_features,
                object_features,
                pose_features,
                temporal_features,
                audio_features
            )
            
            logger.info("Video analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing video: {str(e)}")
            raise

    def fuse_features(self, scene_features, object_features, pose_features, temporal_features, audio_features):
        """Fuse features from different streams into a comprehensive analysis."""
        try:
            # Create a comprehensive analysis dictionary
            analysis = {
                'scene_understanding': {
                    'features': scene_features,
                    'summary': self.summarize_scenes(scene_features)
                },
                'object_analysis': {
                    'detections': object_features,
                    'summary': self.summarize_objects(object_features)
                },
                'pose_analysis': {
                    'poses': pose_features,
                    'summary': self.summarize_poses(pose_features)
                },
                'temporal_analysis': {
                    'features': temporal_features,
                    'summary': self.summarize_temporal(temporal_features)
                },
                'audio_analysis': {
                    'features': audio_features,
                    'summary': self.summarize_audio(audio_features)
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error fusing features: {str(e)}")
            raise

    def summarize_scenes(self, scene_features):
        """Summarize scene understanding results."""
        try:
            # Get most common scene categories
            scene_categories = [frame['scene_category'] for frame in scene_features]
            unique_categories = set(scene_categories)
            
            summary = {
                'primary_scene': scene_categories[0],
                'scene_transitions': len(unique_categories),
                'unique_scenes': list(unique_categories)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing scenes: {str(e)}")
            raise

    def summarize_objects(self, object_features):
        """Summarize object detection results."""
        try:
            # Count object occurrences
            object_counts = {}
            for frame_objects in object_features:
                for obj in frame_objects:
                    obj_class = obj['class']
                    object_counts[obj_class] = object_counts.get(obj_class, 0) + 1
            
            # Get most common objects
            common_objects = sorted(
                object_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            summary = {
                'total_objects_detected': sum(object_counts.values()),
                'unique_objects': len(object_counts),
                'most_common_objects': dict(common_objects)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing objects: {str(e)}")
            raise

    def summarize_poses(self, pose_features):
        """Summarize pose estimation results."""
        try:
            # Count frames with detected poses
            pose_detections = sum(1 for pose in pose_features if pose is not None)
            total_frames = len(pose_features)
            
            summary = {
                'total_frames': total_frames,
                'frames_with_poses': pose_detections,
                'pose_detection_rate': pose_detections / total_frames if total_frames > 0 else 0
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing poses: {str(e)}")
            raise

    def summarize_temporal(self, temporal_features):
        """Summarize temporal analysis results."""
        try:
            # Calculate temporal statistics
            summary = {
                'temporal_segments': len(temporal_features),
                'average_feature_magnitude': float(np.mean(np.abs(temporal_features))),
                'feature_std_dev': float(np.std(temporal_features))
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing temporal features: {str(e)}")
            raise

    def summarize_audio(self, audio_features):
        """Summarize audio analysis results."""
        try:
            # Calculate audio statistics
            total_energy = 0
            for chunk in audio_features:
                total_energy += np.sum(chunk['spectrogram'])
            
            summary = {
                'audio_chunks': len(audio_features),
                'total_audio_energy': float(total_energy),
                'average_chunk_energy': float(total_energy / len(audio_features)) if audio_features else 0
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing audio: {str(e)}")
            raise 