"""
Feature Visualizer for Neural Network Glass Box Visualization

This module provides tools for rendering feature maps extracted from PyTorch models
into publication-quality visualizations suitable for non-technical audiences.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import torch


class FeatureVisualizer:
    """Render feature maps as publication-quality visualizations.

    This class provides methods for creating various types of feature map
    visualizations including grids, hierarchies, and comparative views.

    Example:
        >>> visualizer = FeatureVisualizer(dpi=300, style='scientific')
        >>> visualizer.plot_feature_grid(features, n_cols=8, title='Layer 1 Features')
        >>> visualizer.save('output.png')
    """

    def __init__(
        self,
        dpi: int = 300,
        style: str = 'scientific',
        figsize_scale: float = 1.0,
        font_size: int = 9
    ):
        """Initialize the feature visualizer.

        Args:
            dpi: Resolution for saved figures (300 for publication quality)
            style: Matplotlib style ('scientific', 'seaborn', 'default')
            figsize_scale: Scaling factor for figure sizes
            font_size: Base font size for labels and titles
        """
        self.dpi = dpi
        self.style = style
        self.figsize_scale = figsize_scale
        self.font_size = font_size

        # Configure matplotlib
        plt.rcParams['savefig.dpi'] = dpi
        plt.rcParams['savefig.format'] = 'png'
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['font.size'] = font_size
        plt.rcParams['axes.labelsize'] = font_size
        plt.rcParams['axes.titlesize'] = font_size + 1
        plt.rcParams['xtick.labelsize'] = font_size - 1
        plt.rcParams['ytick.labelsize'] = font_size - 1
        plt.rcParams['legend.fontsize'] = font_size - 1

        if style == 'scientific':
            plt.style.use('seaborn-v0_8-paper')

        self.fig = None
        self.axes = None

    def plot_feature_grid(
        self,
        feature_map: Union[torch.Tensor, np.ndarray],
        n_cols: int = 8,
        channel_indices: Optional[List[int]] = None,
        titles: Optional[List[str]] = None,
        cmap: str = 'viridis',
        show_colorbar: bool = False,
        normalize_per_channel: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Plot a grid of feature map channels.

        Args:
            feature_map: Feature tensor (C, H, W) or (B, C, H, W)
            n_cols: Number of columns in grid
            channel_indices: Which channels to plot (if None, plots all or first 64)
            titles: Optional titles for each subplot
            cmap: Colormap name
            show_colorbar: Whether to show colorbar for each subplot
            normalize_per_channel: Normalize each channel independently
            figsize: Figure size (width, height) or None for auto
            title: Overall figure title

        Returns:
            Tuple of (figure, axes_array)
        """
        # Handle different tensor formats
        if isinstance(feature_map, torch.Tensor):
            feature_map = feature_map.detach().cpu().numpy()

        if feature_map.ndim == 4:
            # Take first image in batch
            feature_map = feature_map[0]

        assert feature_map.ndim == 3, f"Expected 3D tensor (C, H, W), got {feature_map.shape}"

        n_channels, height, width = feature_map.shape

        # Select channels to plot
        if channel_indices is None:
            channel_indices = list(range(min(n_channels, 64)))  # Default: first 64 channels
        n_plots = len(channel_indices)

        n_rows = (n_plots + n_cols - 1) // n_cols

        # Calculate figure size
        if figsize is None:
            aspect_ratio = height / width
            cell_width = 2.0 * self.figsize_scale
            cell_height = cell_width * aspect_ratio
            figsize = (n_cols * cell_width, n_rows * cell_height)

        # Create figure and axes
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(n_rows, n_cols)

        # Plot each channel
        for idx, channel_idx in enumerate(channel_indices):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            channel_data = feature_map[channel_idx]

            # Normalize
            if normalize_per_channel:
                vmin, vmax = channel_data.min(), channel_data.max()
                if vmax - vmin > 1e-6:  # Avoid division by zero
                    channel_data = (channel_data - vmin) / (vmax - vmin)
            else:
                vmin, vmax = 0, 1

            # Plot
            im = ax.imshow(channel_data, cmap=cmap, vmin=0, vmax=1)
            ax.axis('off')

            # Title
            if titles and idx < len(titles):
                ax.set_title(titles[idx], fontsize=self.font_size - 2)
            else:
                ax.set_title(f'Ch {channel_idx}', fontsize=self.font_size - 2)

            # Colorbar
            if show_colorbar:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide extra subplots
        for idx in range(n_plots, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        # Overall title
        if title:
            fig.suptitle(title, fontsize=self.font_size + 2, y=1.02)

        plt.tight_layout()

        self.fig = fig
        self.axes = axes

        return fig, axes

    def plot_feature_hierarchy(
        self,
        layer_features: Dict[str, Union[torch.Tensor, np.ndarray]],
        n_channels_per_layer: int = 9,
        layer_order: Optional[List[str]] = None,
        annotations: Optional[Dict[str, str]] = None,
        layer_names: Optional[Dict[str, str]] = None,
        layer_details: Optional[Dict[str, str]] = None,
        cmap: str = 'viridis',
        figsize: Optional[Tuple[float, float]] = None,
        title: str = 'Feature Hierarchy'
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Plot feature hierarchy across multiple layers (e.g., ResNet layer1-4).

        Creates a grid where each row is a different layer, showing the evolution
        of feature representations from low-level to high-level.

        Args:
            layer_features: Dictionary mapping layer names to feature tensors
            n_channels_per_layer: Number of channels to show per layer
            layer_order: Order to display layers (if None, uses dict order)
            annotations: (Deprecated) Use layer_names instead
            layer_names: Dictionary mapping layer names to display names
            layer_details: Dictionary mapping layer names to technical details
            cmap: Colormap name
            figsize: Figure size or None for auto
            title: Overall figure title

        Returns:
            Tuple of (figure, axes_array)
        """
        if layer_order is None:
            layer_order = list(layer_features.keys())

        n_layers = len(layer_order)

        # Calculate figure size - increase width and height for better spacing
        if figsize is None:
            figsize = (n_channels_per_layer * 1.9 * self.figsize_scale,
                      n_layers * 3.0 * self.figsize_scale)

        # Create figure with generous spacing
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_layers, n_channels_per_layer + 1, figure=fig,
                     width_ratios=[1.0] + [1] * n_channels_per_layer,  # Even wider label column
                     hspace=0.6, wspace=0.18)  # Extra vertical and horizontal space

        all_axes = []

        for layer_idx, layer_name in enumerate(layer_order):
            feature_map = layer_features[layer_name]

            # Convert to numpy
            if isinstance(feature_map, torch.Tensor):
                feature_map = feature_map.detach().cpu().numpy()

            if feature_map.ndim == 4:
                feature_map = feature_map[0]  # First in batch

            n_channels, height, width = feature_map.shape

            # Select top channels by variance
            from src_v2.visualization.utils import select_top_channels_by_variance
            top_indices = select_top_channels_by_variance(feature_map, n_channels_per_layer)

            # Layer label axis
            ax_label = fig.add_subplot(gs[layer_idx, 0])

            # Use custom layer names if provided, otherwise default
            if layer_names and layer_name in layer_names:
                display_name = layer_names[layer_name]
                y_pos = 0.6
            else:
                display_name = f"Layer {layer_idx + 1}"
                y_pos = 0.65

            ax_label.text(0.5, y_pos, display_name, fontsize=self.font_size,
                         ha='center', va='center', rotation=0, weight='bold')

            # Technical details if provided
            if layer_details and layer_name in layer_details:
                ax_label.text(0.5, 0.35, layer_details[layer_name],
                             fontsize=self.font_size - 2,
                             ha='center', va='center', color='gray',
                             family='monospace')

            ax_label.axis('off')

            # Legacy annotation support (deprecated in favor of layer_names)
            if annotations and layer_name in annotations and not layer_names:
                ax_label.text(0.5, 0.15, annotations[layer_name],
                            fontsize=self.font_size - 1,
                            ha='center', va='center',
                            style='italic', color='#555555',
                            wrap=True)

            # Feature map axes
            layer_axes = []
            for ch_idx, channel_idx in enumerate(top_indices):
                ax = fig.add_subplot(gs[layer_idx, ch_idx + 1])

                channel_data = feature_map[channel_idx]

                # Normalize
                vmin, vmax = channel_data.min(), channel_data.max()
                if vmax - vmin > 1e-6:
                    channel_data = (channel_data - vmin) / (vmax - vmin)

                ax.imshow(channel_data, cmap=cmap, vmin=0, vmax=1)
                ax.axis('off')

                if layer_idx == 0:  # Only show channel numbers on top row
                    ax.set_title(f'Ch {channel_idx}', fontsize=self.font_size - 2)

                layer_axes.append(ax)

            all_axes.append(layer_axes)

        fig.suptitle(title, fontsize=self.font_size + 3, y=0.98)

        self.fig = fig
        self.axes = np.array(all_axes)

        return fig, np.array(all_axes)

    def plot_single_feature_overlay(
        self,
        image: Union[torch.Tensor, np.ndarray],
        feature_map: Union[torch.Tensor, np.ndarray],
        channel_idx: int = 0,
        alpha: float = 0.4,
        cmap: str = 'jet',
        figsize: Tuple[float, float] = (12, 4),
        titles: Optional[List[str]] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Plot original image, feature map, and overlay side by side.

        Args:
            image: Original image (H, W) or (C, H, W)
            feature_map: Feature map (C, H, W) or (B, C, H, W)
            channel_idx: Which channel to visualize
            alpha: Transparency for overlay
            cmap: Colormap for feature map
            figsize: Figure size
            titles: Custom titles for [image, feature, overlay]

        Returns:
            Tuple of (figure, axes_array)
        """
        # Convert to numpy
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(feature_map, torch.Tensor):
            feature_map = feature_map.detach().cpu().numpy()

        # Handle batch dimension
        if feature_map.ndim == 4:
            feature_map = feature_map[0]

        # Get channel
        feature_channel = feature_map[channel_idx]

        # Normalize feature map
        from src_v2.visualization.utils import normalize_feature_map
        feature_normalized = normalize_feature_map(feature_channel)

        # Handle image format
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = image.transpose(1, 2, 0)  # CHW -> HWC
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]  # Remove channel dim if grayscale

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Original image
        if image.ndim == 2:
            axes[0].imshow(image, cmap='gray')
        else:
            axes[0].imshow(image)
        axes[0].set_title(titles[0] if titles else 'Original Image', fontsize=self.font_size)
        axes[0].axis('off')

        # Feature map
        im1 = axes[1].imshow(feature_normalized, cmap=cmap)
        axes[1].set_title(titles[1] if titles and len(titles) > 1 else f'Feature Map (Ch {channel_idx})',
                         fontsize=self.font_size)
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        # Overlay
        if image.ndim == 2:
            axes[2].imshow(image, cmap='gray')
        else:
            axes[2].imshow(image)

        # Resize feature map to match image if needed
        if feature_normalized.shape != image.shape[:2]:
            import cv2
            feature_resized = cv2.resize(feature_normalized,
                                        (image.shape[1], image.shape[0]),
                                        interpolation=cv2.INTER_LINEAR)
        else:
            feature_resized = feature_normalized

        axes[2].imshow(feature_resized, cmap=cmap, alpha=alpha)
        axes[2].set_title(titles[2] if titles and len(titles) > 2 else 'Overlay',
                         fontsize=self.font_size)
        axes[2].axis('off')

        plt.tight_layout()

        self.fig = fig
        self.axes = axes

        return fig, axes

    def save(self, filepath: str, dpi: Optional[int] = None, bbox_inches: str = 'tight'):
        """Save the current figure to file.

        Args:
            filepath: Output file path
            dpi: Resolution (if None, uses instance default)
            bbox_inches: Bounding box mode
        """
        if self.fig is None:
            raise ValueError("No figure to save. Create a plot first.")

        self.fig.savefig(filepath, dpi=dpi or self.dpi, bbox_inches=bbox_inches)
        print(f"Figure saved to: {filepath}")

    def close(self):
        """Close the current figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = None


def quick_visualize_layer(
    feature_map: Union[torch.Tensor, np.ndarray],
    n_channels: int = 16,
    output_path: Optional[str] = None,
    title: str = 'Feature Maps'
) -> plt.Figure:
    """Quick visualization of a feature map layer.

    Convenience function for rapid prototyping.

    Args:
        feature_map: Feature tensor (C, H, W) or (B, C, H, W)
        n_channels: Number of channels to display
        output_path: If provided, saves to this path
        title: Figure title

    Returns:
        Matplotlib figure
    """
    visualizer = FeatureVisualizer(dpi=150)  # Lower DPI for quick viewing

    from src_v2.visualization.utils import select_top_channels_by_variance

    if isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.detach().cpu().numpy()

    if feature_map.ndim == 4:
        feature_map = feature_map[0]

    top_channels = select_top_channels_by_variance(feature_map, n_channels)

    fig, _ = visualizer.plot_feature_grid(
        feature_map,
        n_cols=int(np.sqrt(n_channels)),
        channel_indices=top_channels,
        title=title
    )

    if output_path:
        visualizer.save(output_path)

    return fig
