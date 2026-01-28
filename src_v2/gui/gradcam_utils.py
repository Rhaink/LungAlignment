"""
GradCAM utilities for model explainability.

Implements Gradient-weighted Class Activation Mapping (GradCAM) to visualize
which regions of the X-ray the classifier is focusing on.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GradCAM:
    """
    GradCAM implementation for ResNet-based classifiers.

    Generates heatmaps showing which regions of the input image contribute
    most to the model's prediction.
    """

    def __init__(self, model: nn.Module, target_layer: str = 'layer4'):
        """
        Initialize GradCAM.

        Args:
            model: Trained classifier model (must be in eval mode)
            target_layer: Name of the target layer (default: 'layer4' for ResNet18)
        """
        self.model = model
        self.model.eval()

        # Storage for activations and gradients
        self.activations = None
        self.gradients = None

        # Register hooks on the target layer
        self.target_layer = self._get_target_layer(target_layer)
        self._register_hooks()

    def _get_target_layer(self, layer_name: str) -> nn.Module:
        """Get the target layer module from the model."""
        # For ResNet18 classifier, the backbone is model.backbone
        if hasattr(self.model, 'backbone'):
            # Try to get layer from backbone
            if hasattr(self.model.backbone, layer_name):
                return getattr(self.model.backbone, layer_name)

        # Fallback: try to get layer directly from model
        if hasattr(self.model, layer_name):
            return getattr(self.model, layer_name)

        # If still not found, search through named modules
        for name, module in self.model.named_modules():
            if layer_name in name:
                return module

        raise ValueError(f"Layer '{layer_name}' not found in model")

    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, torch.Tensor]:
        """
        Generate GradCAM heatmap.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (None = predicted class)

        Returns:
            heatmap: Normalized heatmap (H, W) with values [0, 1]
            predicted_class: Predicted class index
            logits: Model output logits
        """
        # Ensure gradient computation is enabled
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        # Forward pass
        self.model.zero_grad()
        logits = self.model(input_tensor)

        # Get predicted class if not specified
        predicted_class = logits.argmax(dim=1).item()
        if target_class is None:
            target_class = predicted_class

        # Backward pass on target class score
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)

        # Get activations and gradients
        activations = self.activations  # (1, C, H, W)
        gradients = self.gradients      # (1, C, H, W)

        # Global average pooling on gradients to get weights
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1).squeeze()  # (H, W)

        # Apply ReLU to keep only positive contributions
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam = self._normalize_heatmap(cam)

        return cam, predicted_class, logits

    def _normalize_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """Normalize heatmap to [0, 1] range."""
        if heatmap.max() - heatmap.min() < 1e-8:
            # Avoid division by zero for constant heatmaps
            return np.zeros_like(heatmap)

        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)
        return heatmap


def generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: str = 'layer4',
    target_class: Optional[int] = None,
    resize_to: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, int, torch.Tensor]:
    """
    Convenience function to generate GradCAM heatmap.

    Args:
        model: Trained classifier model
        input_tensor: Input image tensor (1, C, H, W)
        target_layer: Target layer name (default: 'layer4')
        target_class: Target class index (None = predicted class)
        resize_to: Optional (H, W) to resize heatmap to match input size

    Returns:
        heatmap: Normalized heatmap (H, W) or (resize_H, resize_W)
        predicted_class: Predicted class index
        logits: Model output logits
    """
    gradcam = GradCAM(model, target_layer)
    heatmap, predicted_class, logits = gradcam.generate(input_tensor, target_class)

    # Resize heatmap if requested
    if resize_to is not None:
        heatmap = resize_heatmap(heatmap, resize_to)

    return heatmap, predicted_class, logits


def resize_heatmap(heatmap: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize heatmap to target size using bilinear interpolation.

    Args:
        heatmap: Input heatmap (H, W)
        size: Target size (H, W)

    Returns:
        resized: Resized heatmap
    """
    import cv2

    # OpenCV expects (W, H) for size
    target_size = (size[1], size[0])
    resized = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)

    return resized


def apply_colormap(
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Apply colormap to heatmap.

    Args:
        heatmap: Normalized heatmap (H, W) with values [0, 1]
        colormap: OpenCV colormap (default: COLORMAP_JET)

    Returns:
        colored: Colored heatmap (H, W, 3) in RGB format
    """
    import cv2

    # Convert to uint8
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # Apply colormap (returns BGR)
    colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)

    # Convert BGR to RGB
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    return colored_rgb


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay GradCAM heatmap on original image.

    Args:
        image: Original image (H, W) grayscale or (H, W, 3) RGB
        heatmap: Normalized heatmap (H, W) with values [0, 1]
        alpha: Heatmap transparency (0=invisible, 1=opaque)
        colormap: OpenCV colormap

    Returns:
        overlay: Blended image with heatmap (H, W, 3) RGB
    """
    import cv2

    # Ensure heatmap matches image size
    if image.shape[:2] != heatmap.shape[:2]:
        heatmap = resize_heatmap(heatmap, image.shape[:2])

    # Convert grayscale to RGB if needed
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()

    # Apply colormap to heatmap
    heatmap_colored = apply_colormap(heatmap, colormap)

    # Normalize image to [0, 255] if needed
    if image_rgb.max() <= 1.0:
        image_rgb = (image_rgb * 255).astype(np.uint8)
    else:
        image_rgb = image_rgb.astype(np.uint8)

    # Blend
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay
