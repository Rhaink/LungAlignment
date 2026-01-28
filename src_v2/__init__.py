"""
COVID-19 Detection via Anatomical Landmarks and Geometric Normalization.

Landmark Prediction v2 - Clean Implementation
Objetivo: Error < 8 pixeles con ResNet-18

Key Results (v2.1.0):
- Landmark Error: 3.71 px (ensemble 4 models + TTA)
- Classification Accuracy: 99.10% (warped_96, RECOMMENDED)
- Robustness: 2.4x better than warped_99
"""

__version__ = "2.1.0"
__author__ = "Proyecto de Tesis"
