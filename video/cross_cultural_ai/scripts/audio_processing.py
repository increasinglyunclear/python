"""
Audio processing module for transcription and translation.
Supports automatic language detection.
"""

from moviepy.editor import VideoFileClip
import speech_recognition as sr
from deep_translator import (GoogleTranslator, 
                           MicrosoftTranslator)
from langdetect import detect
import os
import tempfile
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_audio_from_video(video_path):
    """Extract audio from video file to a temporary WAV file"""
    try:
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        temp_audio = Path(temp_dir) / f"temp_audio_{os.getpid()}.wav"
        
        video = VideoFileClip(str(video_path))
        video.audio.write_audiofile(
            str(temp_audio), 
            fps=16000, 
            nbytes=2, 
            codec='pcm_s16le',
            verbose=False,
            logger=None
        )
        video.close()
        
        return temp_audio
    except Exception as e:
        logging.error(f"Error extracting audio: {str(e)}")
        return None

def detect_language(text):
    """Detect the language of the transcribed text"""
    try:
        # Force detection of Indonesian if it looks like Indonesian
        lang_code = detect(text)
        if lang_code in ['ja', 'ko'] and any(c.isalpha() for c in text):  # If detected as Japanese/Korean but contains Latin chars
            return 'id'  # Assume Indonesian
        return lang_code
    except Exception as e:
        logging.error(f"Language detection failed: {str(e)}")
        return 'id'  # Default to Indonesian if detection fails

def transcribe_audio(video_path: str) -> str:
    """
    Transcribe audio from video file with automatic language detection.
    Returns the transcribed text or None if failed.
    """
    recognizer = sr.Recognizer()
    temp_audio = None
    
    try:
        # Extract audio to temporary file
        temp_audio = extract_audio_from_video(video_path)
        if not temp_audio:
            return None
            
        with sr.AudioFile(str(temp_audio)) as source:
            # Adjust noise reduction
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)
            
        # Try transcription with language detection
        try:
            # First try with Indonesian
            text = recognizer.recognize_google(audio, language='id-ID')
        except:
            # If that fails, try other common languages
            common_languages = ['id-ID', 'en-US', 'ko-KR', 'ja-JP', 'zh-CN']
            for lang in common_languages:
                try:
                    text = recognizer.recognize_google(audio, language=lang)
                    if text:
                        break
                except:
                    continue
            else:
                return None
        
        return text
        
    except Exception as e:
        logging.error(f"Transcription error: {str(e)}")
        return None
        
    finally:
        # Cleanup temporary file
        if temp_audio and temp_audio.exists():
            temp_audio.unlink()

def translate_text(text: str, target='en') -> str:
    """
    Translate text to target language with automatic source language detection.
    Returns the translated text or None if failed.
    """
    if not text:
        return None

    try:
        # Detect source language
        source_lang = detect_language(text)
        if not source_lang:
            logging.warning("Could not detect source language, assuming Indonesian")
            source_lang = 'id'
            
        # Skip translation if source is same as target
        if source_lang == target:
            return text

        # Try Google Translator
        try:
            translator = GoogleTranslator(source=source_lang, target=target)
            result = translator.translate(text)
            if result:
                return result
        except Exception as e:
            logging.error(f"Google translation failed: {str(e)}")

        # Try Microsoft Translator as backup
        try:
            translator = MicrosoftTranslator(source=source_lang, target=target)
            result = translator.translate(text)
            if result:
                return result
        except Exception as e:
            logging.error(f"Microsoft translation failed: {str(e)}")

        # If both fail, try chunk-based translation
        chunk_size = 1000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        translated_chunks = []

        for chunk in chunks:
            try:
                translator = GoogleTranslator(source=source_lang, target=target)
                translated_chunk = translator.translate(chunk)
                translated_chunks.append(translated_chunk)
            except Exception as e:
                logging.error(f"Chunk translation failed: {str(e)}")
                continue

        if translated_chunks:
            return ' '.join(translated_chunks)

    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
    
    return None

if __name__ == "__main__":
    # Test the module
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        print(f"Processing video: {video_path}")
        
        text = transcribe_audio(video_path)
        if text:
            print("\nTranscribed text:")
            print(text)
            
            translation = translate_text(text)
            if translation:
                print("\nTranslation:")
                print(translation)
