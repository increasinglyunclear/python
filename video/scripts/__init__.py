"""
Pipeline scripts for video processing
"""

from .watch_directory import main as watch_directory
from .object_detection import detect_objects
from .audio_processing import transcribe_audio, translate_text

__all__ = [
    'watch_directory',
    'detect_objects',
    'transcribe_audio',
    'translate_text'
]
