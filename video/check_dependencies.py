import pkg_resources
import subprocess
import sys
from pathlib import Path

def check_python_version():
    print("\nChecking Python version...")
    python_version = sys.version_info
    required_version = (3, 10)
    
    if python_version >= required_version:
        print(f"✓ Python {python_version.major}.{python_version.minor} (meets minimum requirement of 3.10)")
    else:
        print(f"✗ Python {python_version.major}.{python_version.minor} (requires 3.10 or higher)")

def check_ffmpeg():
    print("\nChecking ffmpeg installation...")
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("✓ ffmpeg is installed")
        else:
            print("✗ ffmpeg is not installed properly")
    except FileNotFoundError:
        print("✗ ffmpeg is not installed")

def check_cuda():
    print("\nChecking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available (torch version: {torch.__version__})")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
        else:
            print("! CUDA is not available (will use CPU)")
    except ImportError:
        print("✗ PyTorch is not installed")

def check_pip_packages():
    print("\nChecking Python packages...")
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'ultralytics': 'Ultralytics',
        'opencv-python': 'OpenCV',
        'moviepy': 'MoviePy',
        'SpeechRecognition': 'SpeechRecognition',
        'pydub': 'Pydub',
        'deep-translator': 'Deep Translator',
        'watchdog': 'Watchdog'
    }
    
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    for package, display_name in required_packages.items():
        if package in installed_packages:
            print(f"✓ {display_name} ({installed_packages[package]})")
        else:
            print(f"✗ {display_name} is not installed")

def check_directory_structure():
    print("\nChecking directory structure...")
    required_dirs = [
        'input',
        'results',
        'action_recognition/models',
        'action_recognition/scripts',
        'scripts'
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (missing)")

def check_model_file():
    print("\nChecking for action recognition model...")
    model_dir = Path('action_recognition/models')
    if model_dir.exists():
        model_files = list(model_dir.glob("model_*.pth"))
        if model_files:
            print(f"✓ Found model: {model_files[0].name}")
        else:
            print("! No model file found (action recognition will be skipped)")
    else:
        print("✗ Models directory not found")

def main():
    print("=== Video Analysis Pipeline Dependency Check ===")
    
    check_python_version()
    check_ffmpeg()
    check_cuda()
    check_pip_packages()
    check_directory_structure()
    check_model_file()
    
    print("\nDependency check complete!")

if __name__ == "__main__":
    main()