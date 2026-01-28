"""
Funciones geométricas para procesamiento de landmarks.

Este módulo centraliza operaciones geométricas comunes para evitar
duplicación de código en losses.py y hierarchical.py.
"""

import numpy as np
import torch


def compute_perpendicular_vector_np(axis_vec: np.ndarray) -> np.ndarray:
    """
    Calcula vector perpendicular (rotación 90 grados antihorario) - versión NumPy.

    Útil para funciones de análisis que no requieren autograd.

    Args:
        axis_vec: Vector del eje (2,) o (N, 2)

    Returns:
        Vector perpendicular unitario
    """
    axis_len = np.linalg.norm(axis_vec) + 1e-8
    axis_unit = axis_vec / axis_len
    return np.array([-axis_unit[1], axis_unit[0]])


def compute_perpendicular_vector(axis_vec: torch.Tensor) -> torch.Tensor:
    """
    Calcula vector perpendicular (rotación 90 grados antihorario).

    Dado un vector de eje, calcula el vector perpendicular unitario.
    Útil para calcular distancias perpendiculares de landmarks al eje L1-L2.

    Args:
        axis_vec: Vector del eje (B, 2) donde B es batch size

    Returns:
        Vector perpendicular unitario (B, 2)
    """
    axis_len = torch.norm(axis_vec, dim=1, keepdim=True) + 1e-8
    axis_unit = axis_vec / axis_len
    return torch.stack([-axis_unit[:, 1], axis_unit[:, 0]], dim=1)
