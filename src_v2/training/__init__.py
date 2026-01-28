"""
Training module: Trainer y callbacks
"""

from .trainer import LandmarkTrainer
from .callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback

__all__ = [
    'LandmarkTrainer',
    'EarlyStopping',
    'ModelCheckpoint',
    'LRSchedulerCallback',
]
