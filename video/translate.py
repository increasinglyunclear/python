"""
Video to Text Transcription and Translation Script
-----------------------------------------------

This script extracts audio from video files, transcribes it to text using
Google's Speech Recognition API, (secondarily Microsoft's) and translates 
between multiple languages. Supports many languages, specify in the code.

Dependencies:
------------
- Python 3.x
- moviepy (conda install -c conda-forge moviepy)
- SpeechRecognition (conda install -c conda-forge speechrecognition)
- pydub (conda install -c conda-forge pydub)
- deep-translator (pip install deep-translator)

Optional but recommended:
- ffmpeg (brew install ffmpeg on Mac, or choco install ffmpeg on Windows)

Installation:
------------
1. Create and activate a conda environment (recommended):
   conda create -n translation python=3.x
   conda activate translation

2. Install required packages:
   conda install -c conda-forge moviepy speechrecognition pydub
   conda install pip
   pip install deep-translator

Usage:
------
1. Replace the video_path variable with your video file path
2. Set source_language and target_language as needed
3. Run the script: python transcribe.py

Note:
-----
- Requires internet connection for Google and MS APIs
- Supports multiple video formats (.mp4, .mov, .avi, .mkv, .webm)
- Supports translation between 100+ languages
"""

from moviepy.editor import VideoFileClip
import speech_recognition as sr
from deep_translator import (GoogleTranslator, 
                           MicrosoftTranslator)
import os
import time

def extract_audio_from_video(video_path, audio_path):
    """Extract audio from video file"""
    try:
        print(f"Extracting audio from {video_path}...")
        video = VideoFileClip(video_path)
        # Optimize audio settings for speech recognition
        video.audio.write_audiofile(audio_path, fps=16000, nbytes=2, codec='pcm_s16le')
        video.close()
        print("Audio extraction completed!")
        return True
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return False

def transcribe_audio(audio_path, source_language='ko-KR'):
    """Transcribe audio to text"""
    recognizer = sr.Recognizer()
    try:
        print("Starting transcription...")
        with sr.AudioFile(audio_path) as source:
            print("Reading audio file...")
            # Adjust noise reduction
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("Recording audio...")
            audio = recognizer.record(source)
            
        print(f"Processing audio ({source_language})...")
        text = recognizer.recognize_google(
            audio,
            language=source_language,
            show_all=False
        )
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        print("Try checking your internet connection or audio file format")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def translate_text(text, source='ko', target='en'):
    """Attempt translation using multiple services"""
    if not text:
        return None

    # Try Google Translator
    try:
        print("Attempting Google translation...")
        translator = GoogleTranslator(source=source, target=target)
        result = translator.translate(text)
        if result:
            return result
    except Exception as e:
        print(f"Google translation failed: {str(e)}")

    # Try Microsoft Translator
    try:
        print("Attempting Microsoft translation...")
        translator = MicrosoftTranslator(source=source, target=target)
        result = translator.translate(text)
        if result:
            return result
    except Exception as e:
        print(f"Microsoft translation failed: {str(e)}")

    # If both fail, try chunk-based translation
    try:
        print("Attempting chunk-based translation...")
        chunk_size = 1000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        translated_chunks = []

        for i, chunk in enumerate(chunks, 1):
            print(f"Translating chunk {i} of {len(chunks)}...")
            try:
                translator = GoogleTranslator(source=source, target=target)
                translated_chunk = translator.translate(chunk)
                translated_chunks.append(translated_chunk)
                time.sleep(1)  # Delay between chunks
            except Exception as chunk_e:
                print(f"Chunk {i} translation failed: {str(chunk_e)}")
                continue

        if translated_chunks:
            return ' '.join(translated_chunks)
    except Exception as e:
        print(f"Chunk-based translation failed: {str(e)}")

    return None

def main():
    # File paths
    video_path = "/path/to/video.mov"  # Replace with your video file path
    audio_path = "temp_audio.wav"
    output_text_path = "transcription_korean.txt"
    
    # Language settings
    source_language = 'ko-KR'  # Korean for speech recognition
    target_language = 'en'     # English
    
    try:
        # Extract audio
        if not extract_audio_from_video(video_path, audio_path):
            return

        # Verify audio file
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            print("Error: Audio extraction failed or file is empty")
            return

        # Transcribe audio
        print(f"Attempting transcription in {source_language}...")
        transcribed_text = transcribe_audio(audio_path, source_language)
        
        if transcribed_text:
            # Save original transcription
            with open(output_text_path, 'w', encoding='utf-8') as file:
                file.write(transcribed_text)
            print(f"\nOriginal transcription saved to {output_text_path}")
            print("\nTranscription preview:")
            print(transcribed_text[:500] + "..." if len(transcribed_text) > 500 else transcribed_text)
            
            # Translate text
            print(f"\nTranslating from Korean to English...")
            translated_text = translate_text(
                transcribed_text,
                source='ko',
                target='en'
            )
            
            if translated_text:
                # Save translation
                translation_path = f"translation_{target_language}.txt"
                with open(translation_path, 'w', encoding='utf-8') as file:
                    file.write(translated_text)
                print(f"\nTranslation saved to {translation_path}")
                print("\nTranslation preview:")
                print(translated_text[:500] + "..." if len(translated_text) > 500 else translated_text)
            else:
                print("\nAll translation attempts failed")
                print("Consider trying at a different time or using a different translation service")

        else:
            print("Transcription failed")

    finally:
        # Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print("\nTemporary audio file removed")

if __name__ == "__main__":
    main()
