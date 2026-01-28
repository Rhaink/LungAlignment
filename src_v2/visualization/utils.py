"""
Utility Functions for Glass Box Visualizations

This module provides helper functions for feature map processing,
channel selection, normalization, and layout management.
"""

from typing import List, Tuple, Optional, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2


def select_top_channels_by_variance(
    feature_map: Union[torch.Tensor, np.ndarray],
    n_channels: int = 25
) -> List[int]:
    """Select channels with highest spatial variance.

    High variance channels typically contain the most informative features
    for visualization purposes.

    Args:
        feature_map: Feature tensor (C, H, W) or (B, C, H, W)
        n_channels: Number of channels to select

    Returns:
        List of channel indices sorted by variance (highest first)

    Example:
        >>> features = torch.randn(512, 7, 7)
        >>> top_indices = select_top_channels_by_variance(features, 25)
        >>> print(len(top_indices))  # 25
    """
    if isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.detach().cpu().numpy()

    if feature_map.ndim == 4:
        feature_map = feature_map[0]  # Take first in batch

    assert feature_map.ndim == 3, f"Expected 3D tensor (C, H, W), got {feature_map.shape}"

    n_total_channels = feature_map.shape[0]
    n_channels = min(n_channels, n_total_channels)

    # Compute variance across spatial dimensions
    variances = np.var(feature_map, axis=(1, 2))  # Shape: (C,)

    # Get top indices
    top_indices = np.argsort(variances)[::-1][:n_channels]

    return top_indices.tolist()


def select_top_channels_by_gradient(
    feature_map: Union[torch.Tensor, np.ndarray],
    gradients: Union[torch.Tensor, np.ndarray],
    n_channels: int = 25
) -> List[int]:
    """Select channels with highest gradient magnitude.

    These are the channels most important for the model's prediction.

    Args:
        feature_map: Feature tensor (C, H, W)
        gradients: Gradient tensor (C, H, W) with same shape as feature_map
        n_channels: Number of channels to select

    Returns:
        List of channel indices sorted by gradient importance
    """
    if isinstance(gradients, torch.Tensor):
        gradients = gradients.detach().cpu().numpy()

    if gradients.ndim == 4:
        gradients = gradients[0]

    # Global average pooling of absolute gradients per channel
    importance = np.mean(np.abs(gradients), axis=(1, 2))

    top_indices = np.argsort(importance)[::-1][:n_channels]

    return top_indices.tolist()


def normalize_feature_map(
    feature_map: Union[torch.Tensor, np.ndarray],
    method: str = 'min-max',
    percentile: Tuple[float, float] = (2, 98)
) -> np.ndarray:
    """Normalize feature map for visualization.

    Args:
        feature_map: Feature map (H, W) or (C, H, W)
        method: Normalization method ('min-max', 'percentile', 'z-score')
        percentile: Percentile range for 'percentile' method

    Returns:
        Normalized feature map in range [0, 1]
    """
    if isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.detach().cpu().numpy()

    feature_map = feature_map.astype(np.float32)

    if method == 'min-max':
        vmin, vmax = feature_map.min(), feature_map.max()
        if vmax - vmin > 1e-6:
            normalized = (feature_map - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(feature_map)

    elif method == 'percentile':
        vmin = np.percentile(feature_map, percentile[0])
        vmax = np.percentile(feature_map, percentile[1])
        normalized = np.clip((feature_map - vmin) / (vmax - vmin + 1e-6), 0, 1)

    elif method == 'z-score':
        mean, std = feature_map.mean(), feature_map.std()
        if std > 1e-6:
            normalized = (feature_map - mean) / std
            # Map to [0, 1] using sigmoid-like function
            normalized = 1 / (1 + np.exp(-normalized))
        else:
            normalized = np.ones_like(feature_map) * 0.5

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized


def create_figure_grid(
    n_plots: int,
    n_cols: int = 4,
    figsize_per_plot: Tuple[float, float] = (3, 3),
    **subplot_kw
) -> Tuple[plt.Figure, np.ndarray]:
    """Create a figure with grid of subplots.

    Args:
        n_plots: Number of subplots needed
        n_cols: Number of columns
        figsize_per_plot: Size of each subplot (width, height)
        **subplot_kw: Additional arguments for plt.subplots

    Returns:
        Tuple of (figure, axes_array)
    """
    n_rows = (n_plots + n_cols - 1) // n_cols

    figsize = (n_cols * figsize_per_plot[0], n_rows * figsize_per_plot[1])

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, **subplot_kw)

    # Ensure axes is always 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    return fig, axes


def resize_feature_map(
    feature_map: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """Resize feature map to target size.

    Args:
        feature_map: Feature map (H, W) or (C, H, W)
        target_size: Target size (height, width)
        interpolation: OpenCV interpolation method

    Returns:
        Resized feature map
    """
    if feature_map.ndim == 3:
        # Resize each channel
        c, h, w = feature_map.shape
        resized = np.zeros((c, target_size[0], target_size[1]), dtype=feature_map.dtype)
        for i in range(c):
            resized[i] = cv2.resize(feature_map[i], (target_size[1], target_size[0]),
                                   interpolation=interpolation)
        return resized
    else:
        # Single channel
        return cv2.resize(feature_map, (target_size[1], target_size[0]),
                         interpolation=interpolation)


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """Overlay a heatmap on an image.

    Args:
        image: Grayscale or RGB image (H, W) or (H, W, C)
        heatmap: Heatmap in range [0, 1] (H, W)
        alpha: Transparency of heatmap
        colormap: OpenCV colormap (e.g., cv2.COLORMAP_JET, cv2.COLORMAP_VIRIDIS)

    Returns:
        RGB image with heatmap overlay
    """
    # Ensure image is RGB
    if image.ndim == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()

    # Normalize image to [0, 255]
    if image_rgb.max() <= 1.0:
        image_rgb = (image_rgb * 255).astype(np.uint8)

    # Resize heatmap if needed
    if heatmap.shape[:2] != image_rgb.shape[:2]:
        heatmap = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))

    # Normalize heatmap to [0, 255]
    heatmap_normalized = (heatmap * 255).astype(np.uint8)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, colormap)

    # Blend
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay


def create_annotation_box(
    ax: plt.Axes,
    text: str,
    position: Tuple[float, float] = (0.5, 0.95),
    fontsize: int = 9,
    bbox_props: Optional[dict] = None
):
    """Add an annotation box to an axes.

    Args:
        ax: Matplotlib axes
        text: Annotation text
        position: Position in axes coordinates (x, y)
        fontsize: Font size
        bbox_props: Custom bounding box properties
    """
    if bbox_props is None:
        bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor='gray', alpha=0.8)

    ax.text(position[0], position[1], text,
           transform=ax.transAxes,
           fontsize=fontsize,
           verticalalignment='top',
           horizontalalignment='center',
           bbox=bbox_props)


def get_color_palette(name: str = 'scientific') -> dict:
    """Get a predefined color palette for consistent visualizations.

    Args:
        name: Palette name ('scientific', 'vibrant', 'pastel')

    Returns:
        Dictionary mapping color names to RGB tuples or hex codes
    """
    palettes = {
        'scientific': {
            'blue': '#1f77b4',
            'orange': '#ff7f0e',
            'green': '#2ca02c',
            'red': '#d62728',
            'purple': '#9467bd',
            'brown': '#8c564b',
            'pink': '#e377c2',
            'gray': '#7f7f7f',
            'olive': '#bcbd22',
            'cyan': '#17becf',
        },
        'vibrant': {
            'red': '#E63946',
            'orange': '#F77F00',
            'yellow': '#FCBF49',
            'green': '#06A77D',
            'blue': '#118AB2',
            'purple': '#7209B7',
            'pink': '#F72585',
        },
        'pastel': {
            'blue': '#A8DADC',
            'red': '#E63946',
            'yellow': '#F1FAEE',
            'green': '#457B9D',
            'navy': '#1D3557',
        }
    }

    return palettes.get(name, palettes['scientific'])


def compute_activation_statistics(
    feature_map: Union[torch.Tensor, np.ndarray]
) -> dict:
    """Compute statistics for a feature map.

    Useful for understanding activation distributions and detecting
    potential issues (e.g., dead neurons, saturation).

    Args:
        feature_map: Feature tensor (C, H, W) or (B, C, H, W)

    Returns:
        Dictionary with statistics
    """
    if isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.detach().cpu().numpy()

    if feature_map.ndim == 4:
        feature_map = feature_map[0]

    stats = {
        'shape': feature_map.shape,
        'mean': float(np.mean(feature_map)),
        'std': float(np.std(feature_map)),
        'min': float(np.min(feature_map)),
        'max': float(np.max(feature_map)),
        'median': float(np.median(feature_map)),
        'n_zero_channels': int(np.sum(np.all(feature_map == 0, axis=(1, 2)))),
        'sparsity': float(np.mean(feature_map == 0)),  # Fraction of zero activations
    }

    # Per-channel statistics
    stats['channel_means'] = np.mean(feature_map, axis=(1, 2)).tolist()
    stats['channel_stds'] = np.std(feature_map, axis=(1, 2)).tolist()
    stats['channel_variances'] = np.var(feature_map, axis=(1, 2)).tolist()

    return stats


def save_figure_with_metadata(
    fig: plt.Figure,
    filepath: str,
    metadata: dict,
    dpi: int = 300
):
    """Save figure with embedded metadata.

    Args:
        fig: Matplotlib figure
        filepath: Output path
        metadata: Dictionary of metadata to embed
        dpi: Resolution
    """
    # Save figure
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', metadata=metadata)

    # Also save metadata as JSON sidecar
    import json
    metadata_path = filepath.replace('.png', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Figure saved to: {filepath}")
    print(f"Metadata saved to: {metadata_path}")


def draw_scientific_crosses_on_image(
    image: np.ndarray,
    landmarks: np.ndarray,
    cross_size: int = 5,
    thickness: int = 2,
    color: Union[str, Tuple[int, int, int]] = 'red',
    return_rgb: bool = True
) -> np.ndarray:
    """
    Draw scientific-style crosses on landmarks for publication-quality figures.

    Args:
        image: Input grayscale image (H, W) or RGB (H, W, 3)
        landmarks: Array of shape (15, 2) with (x, y) pixel coordinates
        cross_size: Half-length of cross arms in pixels (default: 5)
        thickness: Line thickness in pixels (default: 2)
        color: Color as string ('red', 'green', 'blue') or RGB tuple (B,G,R for OpenCV)
        return_rgb: If True, return RGB image; if False, return grayscale with colored marks

    Returns:
        Image with crosses drawn on landmarks

    Example:
        >>> img = cv2.imread('xray.png', cv2.IMREAD_GRAYSCALE)  # 299x299
        >>> landmarks_299 = landmarks_224 * (299 / 224)
        >>> img_viz = draw_scientific_crosses_on_image(img, landmarks_299)
        >>> cv2.imwrite('visualization.png', img_viz)
    """
    # Convert to RGB if grayscale
    if image.ndim == 2:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = image.copy()

    # Parse color
    color_map = {
        'red': (0, 0, 255),      # BGR format for OpenCV
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'cyan': (255, 255, 0),
        'magenta': (255, 0, 255),
        'white': (255, 255, 255),
        'black': (0, 0, 0)
    }

    if isinstance(color, str):
        bgr_color = color_map.get(color.lower(), (0, 0, 255))  # Default to red
    else:
        bgr_color = color

    # Draw crosses on each landmark
    for landmark_idx, (x, y) in enumerate(landmarks):
        x_int, y_int = int(round(x)), int(round(y))

        # Horizontal line of cross
        cv2.line(
            img_rgb,
            (x_int - cross_size, y_int),
            (x_int + cross_size, y_int),
            bgr_color,
            thickness,
            lineType=cv2.LINE_AA  # Anti-aliased for publication quality
        )

        # Vertical line of cross
        cv2.line(
            img_rgb,
            (x_int, y_int - cross_size),
            (x_int, y_int + cross_size),
            bgr_color,
            thickness,
            lineType=cv2.LINE_AA
        )

    if not return_rgb and image.ndim == 2:
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    return img_rgb
