"""
Utility script to inspect saved model architecture
"""

import torch
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_model(model_path):
    """
    Inspect the contents of a saved model checkpoint
    
    Args:
        model_path: Path to the .pth checkpoint file
    """
    try:
        # Load checkpoint
        logger.info(f"Loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print("\nCheckpoint contents:")
        print("-" * 20)
        for key in checkpoint.keys():
            print(f"- {key}")
        
        print("\nModel state dict keys:")
        print("-" * 20)
        for key in checkpoint['model_state_dict'].keys():
            shape = checkpoint['model_state_dict'][key].shape
            print(f"- {key}: {shape}")
            
        print("\nClass names:")
        print("-" * 20)
        print(checkpoint['class_names'])
        
    except Exception as e:
        logger.error(f"Error inspecting model: {str(e)}")
        raise

def main():
    # Use absolute path from project root
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / "action_recognition/models/model_20241220_145338.pth"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    inspect_model(model_path)

if __name__ == "__main__":
    main()