from setuptools import setup, find_packages

setup(
    name="pipeline",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'ultralytics',
        'opencv-python',
        'numpy',
        'torch',
        'torchvision',
        'moviepy',
        'speechrecognition',
        'pydub',
        'deep-translator',
        'watchdog',
        'langdetect'
    ]
)