"""
Processing module for COVID-19 Landmark Detection.

Includes:
- GPA (Generalized Procrustes Analysis) for canonical shape computation
- Piecewise Affine Warping for geometric normalization
"""

from src_v2.processing.gpa import (
    center_shape,
    scale_shape,
    optimal_rotation_matrix,
    align_shape,
    procrustes_distance,
    gpa_iterative,
    scale_canonical_to_image,
    compute_delaunay_triangulation,
)

from src_v2.processing.warp import (
    piecewise_affine_warp,
    scale_landmarks_from_centroid,
    clip_landmarks_to_image,
    add_boundary_points,
)

__all__ = [
    # GPA functions
    "center_shape",
    "scale_shape",
    "optimal_rotation_matrix",
    "align_shape",
    "procrustes_distance",
    "gpa_iterative",
    "scale_canonical_to_image",
    "compute_delaunay_triangulation",
    # Warp functions
    "piecewise_affine_warp",
    "scale_landmarks_from_centroid",
    "clip_landmarks_to_image",
    "add_boundary_points",
]
