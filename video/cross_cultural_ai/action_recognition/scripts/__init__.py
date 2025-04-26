"""
Action Recognition Package

This package contains scripts for processing videos and extracting pose data
for action recognition tasks.
"""

from .config import Config
from .pose_estimation import PoseEstimator
# Commenting out the other imports until they are used in future version
# from .action_training import ProgressiveTrainer
# from .cross_validation import CrossDatasetValidator
# from .action_classifier import ActionClassifier

__all__ = [
    'Config',
    'PoseEstimator',
    # 'ProgressiveTrainer',
    # 'CrossDatasetValidator',
    # 'ActionClassifier'
]

__version__ = '0.1.0'
