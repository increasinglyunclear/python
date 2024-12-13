"""
Video to Text Transcription Script
--------------------------------

This script extracts audio from video files and transcribes it to text using
Google's Speech Recognition API. Supports multiple video formats including
.mp4, .mov, .avi, .mkv, and .webm.

Dependencies:
------------
- Python 3.x
- moviepy (pip install moviepy)
- SpeechRecognition (pip install SpeechRecognition)
- pydub (pip install pydub)

Optional but recommended:
- ffmpeg (brew install ffmpeg on Mac, or choco install ffmpeg on Windows)

Installation:
------------
1. Create and activate a virtual environment (optional but recommended):
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   venv\Scripts\activate     # On Windows

2. Install required packages:
   pip install moviepy SpeechRecognition pydub

3. Install ffmpeg (recommended):
   brew install ffmpeg      # On Mac
   choco install ffmpeg     # On Windows

Usage:
------
1. Replace the video_path variable with your video file path
2. Run the script: python transcribe.py
3. The transcription will be saved to 'transcription.txt'

Note:
-----
- Requires internet connection for Google Speech Recognition API
- Accuracy depends on audio quality
- Free tier of Google Speech Recognition has usage limits
- Supports multiple video formats (.mp4, .mov, .avi, .mkv, .webm)
"""

from moviepy.editor import VideoFileClip
import speech_recognition as sr
import os

def is_supported_format(file_path):
    """Check if the file format is supported"""
    supported_formats = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    file_extension = os.path.splitext(file_path)[1].lower()
    return file_extension in supported_formats

def extract_audio_from_video(video_path, audio_path):
    """Extract audio from video file"""
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return False
        
    if not is_supported_format(video_path):
        print(f"Warning: File format {os.path.splitext(video_path)[1]} might not be supported.")
        print("Attempting to process anyway...")

    try:
        print(f"Extracting audio from {video_path}...")
        video = VideoFileClip(video_path)
        
        if video.audio is None:
            print("Error: No audio stream found in the video file.")
            return False
            
        video.audio.write_audiofile(audio_path)
        video.close()
        print("Audio extraction completed!")
        return True
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return False

def transcribe_audio(audio_path):
    """Transcribe audio to text"""
    recognizer = sr.Recognizer()
    try:
        print("Starting transcription...")
        with sr.AudioFile(audio_path) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source)
            # Record the audio
            audio = recognizer.record(source)
            
        print("Processing audio...")
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def main():
    # File paths
    video_path = "path/to/your/video.mov"  # Works with multiple video formats
    audio_path = "temp_audio.wav"
    output_text_path = "transcription.txt"

    try:
        # Extract audio
        if not extract_audio_from_video(video_path, audio_path):
            return

        # Transcribe audio
        transcribed_text = transcribe_audio(audio_path)
        
        if transcribed_text:
            # Save transcription to file
            with open(output_text_path, 'w') as file:
                file.write(transcribed_text)
            print(f"Transcription saved to {output_text_path}")
            print("\nTranscription preview:")
            print(transcribed_text[:500] + "..." if len(transcribed_text) > 500 else transcribed_text)
        else:
            print("Transcription failed")

    finally:
        # Cleanup: remove temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print("Temporary audio file removed")

if __name__ == "__main__":
    main()
