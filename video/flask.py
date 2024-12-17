"""
CCAI Video Analysis Web Application
----------------------------
A comprehensive web app for AI video analysis including:
- Object detection
- Pose estimation
- Action recognition
- Audio transcription and translation

Dependencies:
- Flask
- OpenCV
- YOLOv5
- MediaPipe
- SpeechRecognition
- deep-translator
[other dependencies in individual scripts]
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from pathlib import Path

# Import our previous components
from object_detection import ObjectDetector
from pose_estimation import PoseEstimator
from transcribe import transcribe_audio, translate_text

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process video
        results = process_video(filepath)
        
        return jsonify(results)
    
    return jsonify({'error': 'Invalid file type'}), 400

def process_video(video_path):
    """Process video through all analysis components"""
    results = {
        'objects': [],
        'poses': [],
        'actions': [],
        'transcription': None,
        'translation': None
    }
    
    try:
        # Object Detection
        object_detector = ObjectDetector()
        results['objects'] = object_detector.detect(video_path)
        
        # Pose Estimation
        pose_estimator = PoseEstimator()
        results['poses'] = pose_estimator.process_video(video_path)
        
        # Audio Processing
        audio_path = extract_audio(video_path)
        if audio_path:
            transcription = transcribe_audio(audio_path)
            if transcription:
                results['transcription'] = transcription
                results['translation'] = translate_text(transcription)
                
    except Exception as e:
        results['error'] = str(e)
    
    return results

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
