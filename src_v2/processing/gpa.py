"""
Generalized Procrustes Analysis (GPA) for Canonical Shape Computation.

This module implements GPA to compute the canonical (mean) shape
from a set of landmark configurations.

GPA eliminates:
- Translation (center at origin)
- Scale (normalize to unit norm)
- Rotation (align with reference)

References:
    Gower, J.C. (1975). Generalized procrustes analysis.
    Psychometrika, 40(1), 33-51.

    Dryden, I.L. & Mardia, K.V. (1998). Statistical Shape Analysis.
    Wiley Series in Probability and Statistics. Wiley.
"""

import logging
import numpy as np
import warnings
from typing import Tuple, Dict, Optional
from scipy.spatial import Delaunay


logger = logging.getLogger(__name__)


def center_shape(shape: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Center a shape at the origin (remove translation).

    Args:
        shape: Array (n_landmarks, 2) with (x, y) coordinates

    Returns:
        centered: Shape centered at origin
        centroid: Original centroid (to revert if needed)
    """
    centroid = shape.mean(axis=0)
    centered = shape - centroid
    return centered, centroid


def scale_shape(shape: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Scale a shape to unit norm (Frobenius norm = 1).

    Args:
        shape: Array (n_landmarks, 2) centered at origin

    Returns:
        scaled: Shape with unit norm
        scale: Original scale factor (to revert if needed)
    """
    scale = np.linalg.norm(shape, 'fro')
    if scale < 1e-10:
        warnings.warn("Shape has near-zero scale")
        return shape, 1.0
    scaled = shape / scale
    return scaled, scale


def optimal_rotation_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Compute optimal rotation matrix using SVD (Procrustes).

    Finds R that minimizes ||source @ R - target||^2

    Args:
        source: Shape to rotate (n_landmarks, 2)
        target: Reference shape (n_landmarks, 2)

    Returns:
        R: 2x2 rotation matrix
    """
    # Cross-correlation: H = source^T @ target
    H = source.T @ target

    # SVD: H = U @ S @ Vt
    U, S, Vt = np.linalg.svd(H)

    # Optimal rotation: R = V @ U^T
    R = Vt.T @ U.T

    # Ensure proper rotation (det = +1, not reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def align_shape(shape: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Align a shape with a reference (remove rotation).

    Args:
        shape: Shape to align (n_landmarks, 2) - already centered and scaled
        reference: Reference shape (n_landmarks, 2) - already centered and scaled

    Returns:
        aligned: Rotated shape that minimizes distance to reference
    """
    R = optimal_rotation_matrix(shape, reference)
    aligned = shape @ R
    return aligned


def procrustes_distance(shape1: np.ndarray, shape2: np.ndarray) -> float:
    """
    Compute Procrustes distance between two shapes.

    Distance is computed AFTER aligning both shapes.

    Args:
        shape1, shape2: Shapes to compare (n_landmarks, 2)

    Returns:
        distance: Procrustes distance (Frobenius norm of residual)
    """
    # Center and scale both
    s1_centered, _ = center_shape(shape1)
    s1_scaled, _ = scale_shape(s1_centered)

    s2_centered, _ = center_shape(shape2)
    s2_scaled, _ = scale_shape(s2_centered)

    # Align shape1 with shape2
    s1_aligned = align_shape(s1_scaled, s2_scaled)

    # Distance = norm of difference
    distance = np.linalg.norm(s1_aligned - s2_scaled, 'fro')
    return distance


def gpa_iterative(
    shapes: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generalized Procrustes Analysis (iterative).

    Algorithm:
    1. Center and scale all shapes
    2. Initialize reference = mean of shapes
    3. Repeat until convergence:
       a) Rotate each shape to align with reference
       b) Compute new reference = mean of aligned shapes
       c) Normalize reference
       d) Check convergence (change < tolerance)

    Args:
        shapes: Array (n_shapes, n_landmarks, 2) with all shapes
        max_iterations: Maximum number of iterations
        tolerance: Tolerance for convergence
        verbose: If True, print progress

    Returns:
        canonical_shape: Canonical shape (Procrustes consensus)
        aligned_shapes: All aligned shapes (n_shapes, n_landmarks, 2)
        convergence_info: Convergence information dict
    """
    n_shapes, n_landmarks, n_dims = shapes.shape

    if verbose:
        logger.info("GPA: %d shapes, %d landmarks", n_shapes, n_landmarks)

    # Step 1: Center and scale all shapes
    normalized_shapes = np.zeros_like(shapes)
    original_scales = np.zeros(n_shapes)
    original_centroids = np.zeros((n_shapes, 2))

    for i in range(n_shapes):
        centered, centroid = center_shape(shapes[i])
        scaled, scale = scale_shape(centered)
        normalized_shapes[i] = scaled
        original_scales[i] = scale
        original_centroids[i] = centroid

    # Step 2: Initialize reference as mean
    reference = normalized_shapes.mean(axis=0)
    reference_scaled, _ = scale_shape(reference)

    # Store convergence history
    distances_history = []

    # Step 3: Iterate until convergence
    aligned_shapes = normalized_shapes.copy()
    change = float('inf')
    iteration = -1  # Initialize in case max_iterations=0

    for iteration in range(max_iterations):
        # 3a) Rotate each shape to align with reference
        for i in range(n_shapes):
            aligned_shapes[i] = align_shape(normalized_shapes[i], reference_scaled)

        # 3b) Compute new reference
        new_reference = aligned_shapes.mean(axis=0)

        # 3c) Normalize reference
        new_reference_scaled, _ = scale_shape(new_reference)

        # 3d) Compute change (distance between references)
        change = np.linalg.norm(new_reference_scaled - reference_scaled, 'fro')

        # Compute mean distance to consensus
        mean_distance = np.mean([
            np.linalg.norm(aligned_shapes[i] - new_reference_scaled, 'fro')
            for i in range(n_shapes)
        ])
        distances_history.append(mean_distance)

        if verbose and (iteration < 5 or iteration % 10 == 0):
            logger.debug("  Iter %d: change=%.2e, mean_dist=%.6f", iteration, change, mean_distance)

        # Check convergence
        if change < tolerance:
            if verbose:
                logger.info("  Converged at iteration %d (change=%.2e)", iteration, change)
            reference_scaled = new_reference_scaled
            break

        reference_scaled = new_reference_scaled

    else:
        if verbose:
            logger.warning("  Reached max iterations (%d)", max_iterations)

    # Final canonical shape
    canonical_shape = reference_scaled

    # Convergence info
    convergence_info = {
        'n_iterations': iteration + 1,
        'converged': change < tolerance,
        'final_change': float(change),
        'distances_history': distances_history,
        'original_scales': original_scales,
        'original_centroids': original_centroids,
        'n_shapes': n_shapes,
        'n_landmarks': n_landmarks
    }

    return canonical_shape, aligned_shapes, convergence_info


def scale_canonical_to_image(
    canonical_shape: np.ndarray,
    image_size: int = 224,
    padding: float = 0.1
) -> np.ndarray:
    """
    Convert normalized canonical shape to image pixel coordinates.

    Args:
        canonical_shape: Canonical shape (n_landmarks, 2) with norm ~1
        image_size: Target image size
        padding: Relative margin (0.1 = 10% padding)

    Returns:
        scaled_shape: Shape in image coordinates (pixels)
    """
    # The canonical shape is centered at (0,0) with norm ~1
    # We need to scale and translate to image center

    # Compute current range
    min_coords = canonical_shape.min(axis=0)
    max_coords = canonical_shape.max(axis=0)
    range_coords = max_coords - min_coords

    # Scale to fit in image with padding
    usable_size = image_size * (1 - 2 * padding)
    max_range = max(range_coords)
    if max_range < 1e-10:
        warnings.warn("Canonical shape has near-zero range, using default scale")
        max_range = 1.0
    scale_factor = usable_size / max_range

    # Scale and center
    scaled = canonical_shape * scale_factor

    # Translate to image center
    scaled_center = scaled.mean(axis=0)
    image_center = np.array([image_size / 2, image_size / 2])
    scaled_shape = scaled - scaled_center + image_center

    return scaled_shape


def compute_delaunay_triangulation(landmarks: np.ndarray) -> np.ndarray:
    """
    Compute Delaunay triangulation on landmarks.

    Args:
        landmarks: Array (n_landmarks, 2) with landmark coordinates

    Returns:
        triangles: Array (n_triangles, 3) with vertex indices for each triangle
    """
    tri = Delaunay(landmarks)
    return tri.simplices
