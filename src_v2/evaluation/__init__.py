"""
Evaluation module: Metricas y visualizacion de resultados
"""

from .metrics import (
    compute_pixel_error,
    compute_error_per_landmark,
    compute_error_per_category,
    evaluate_model,
    generate_evaluation_report,
)

__all__ = [
    'compute_pixel_error',
    'compute_error_per_landmark',
    'compute_error_per_category',
    'evaluate_model',
    'generate_evaluation_report',
]
