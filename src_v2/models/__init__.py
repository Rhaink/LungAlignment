"""
Models module: Arquitecturas y funciones de perdida
"""

from .resnet_landmark import ResNet18Landmarks, create_model
from .losses import (
    WingLoss,
    WeightedWingLoss,
    CentralAlignmentLoss,
    SoftSymmetryLoss,
    CombinedLandmarkLoss,
)
from .classifier import (
    ImageClassifier,
    create_classifier,
    get_classifier_transforms,
    get_class_weights,
    load_classifier_checkpoint,
    GrayscaleToRGB,
)

__all__ = [
    # Landmark detection
    'ResNet18Landmarks',
    'create_model',
    # Losses
    'WingLoss',
    'WeightedWingLoss',
    'CentralAlignmentLoss',
    'SoftSymmetryLoss',
    'CombinedLandmarkLoss',
    # Classification
    'ImageClassifier',
    'create_classifier',
    'get_classifier_transforms',
    'get_class_weights',
    'load_classifier_checkpoint',
    'GrayscaleToRGB',
]
