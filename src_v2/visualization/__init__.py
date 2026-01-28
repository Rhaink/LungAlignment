"""Visualization module for COVID-19 classification explainability.

This module provides:
- GradCAM: Class Activation Maps for model interpretability
- Error Analysis: Tools for analyzing classification errors
- PFS Analysis: Pulmonary Focus Score analysis tools
- Glass Box: Feature extraction and visualization for neural network interpretability
"""

from src_v2.visualization.gradcam import (
    GradCAM,
    get_target_layer,
    calculate_pfs,
    overlay_heatmap,
)
from src_v2.visualization.error_analysis import (
    ErrorAnalyzer,
    analyze_classification_errors,
)
from src_v2.visualization.pfs_analysis import (
    PFSAnalyzer,
    PFSResult,
    PFSSummary,
    run_pfs_analysis,
    create_pfs_visualizations,
    load_lung_mask,
    find_mask_for_image,
    generate_approximate_mask,
)
from src_v2.visualization.scientific_viz import (
    load_prediction_cache,
    scale_landmarks_to_viz,
    create_scientific_visualization,
    generate_viz_dataset,
)
from src_v2.visualization.comparison_viz import (
    generate_comparison_dataset,
    create_comparison_visualization,
    load_ground_truth_mapping,
    match_predictions_with_gt,
)

# Glass Box Visualization imports (new)
try:
    from src_v2.visualization.feature_extractor import FeatureExtractor
    from src_v2.visualization.feature_visualizer import FeatureVisualizer
    from src_v2.visualization.utils import (
        select_top_channels_by_variance,
        normalize_feature_map,
        create_figure_grid,
        draw_scientific_crosses_on_image,
    )
    _glass_box_available = True
except ImportError:
    _glass_box_available = False

__all__ = [
    "GradCAM",
    "get_target_layer",
    "calculate_pfs",
    "overlay_heatmap",
    "ErrorAnalyzer",
    "analyze_classification_errors",
    "PFSAnalyzer",
    "PFSResult",
    "PFSSummary",
    "run_pfs_analysis",
    "create_pfs_visualizations",
    "load_lung_mask",
    "find_mask_for_image",
    "generate_approximate_mask",
    "load_prediction_cache",
    "scale_landmarks_to_viz",
    "create_scientific_visualization",
    "generate_viz_dataset",
    "generate_comparison_dataset",
    "create_comparison_visualization",
    "load_ground_truth_mapping",
    "match_predictions_with_gt",
]

if _glass_box_available:
    __all__.extend([
        "FeatureExtractor",
        "FeatureVisualizer",
        "select_top_channels_by_variance",
        "normalize_feature_map",
        "create_figure_grid",
        "draw_scientific_crosses_on_image",
    ])
