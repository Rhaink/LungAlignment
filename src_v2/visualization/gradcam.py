"""Grad-CAM implementation for COVID-19 classifier explainability.

This module provides:
- GradCAM class for generating class activation maps
- Automatic target layer detection per architecture
- Pulmonary Focus Score (PFS) calculation
- Heatmap overlay visualization
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm


# Target layer mapping for each supported architecture
TARGET_LAYER_MAP: Dict[str, str] = {
    "resnet18": "backbone.layer4",
    "resnet50": "backbone.layer4",
    "densenet121": "backbone.features.denseblock4",
    "efficientnet_b0": "backbone.features.8",
    "vgg16": "backbone.features.30",
    "alexnet": "backbone.features.12",
    "mobilenet_v2": "backbone.features.18",
}


def get_target_layer(model: nn.Module, backbone_name: str, layer_name: Optional[str] = None) -> nn.Module:
    """Get the target layer for GradCAM based on architecture.

    Args:
        model: The classifier model (ImageClassifier)
        backbone_name: Name of the backbone architecture
        layer_name: Optional specific layer name (overrides auto-detection)

    Returns:
        The target layer module

    Raises:
        ValueError: If backbone is not supported or layer not found
    """
    if layer_name is not None and layer_name != "auto":
        # User specified a custom layer
        return _get_layer_by_name(model, layer_name)

    # Auto-detect based on architecture
    if backbone_name not in TARGET_LAYER_MAP:
        raise ValueError(
            f"Unsupported backbone: {backbone_name}. "
            f"Supported: {list(TARGET_LAYER_MAP.keys())}"
        )

    layer_path = TARGET_LAYER_MAP[backbone_name]
    return _get_layer_by_name(model, layer_path)


def _get_layer_by_name(model: nn.Module, layer_path: str) -> nn.Module:
    """Navigate through model to find layer by dot-separated path.

    Args:
        model: PyTorch model
        layer_path: Dot-separated path (e.g., 'backbone.layer4')

    Returns:
        The target layer module

    Raises:
        ValueError: If layer path is invalid
    """
    parts = layer_path.split(".")
    current = model

    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        elif part.isdigit() and hasattr(current, "__getitem__"):
            current = current[int(part)]
        else:
            raise ValueError(f"Layer not found: {layer_path} (failed at '{part}')")

    return current


class GradCAM:
    """Gradient-weighted Class Activation Mapping for model interpretability.

    Generates heatmaps showing which regions of an image contribute most
    to the model's prediction.

    Args:
        model: PyTorch model (ImageClassifier)
        target_layer: Target layer for activation extraction

    Example:
        >>> model = ImageClassifier(backbone="resnet18")
        >>> layer = get_target_layer(model, "resnet18")
        >>> gradcam = GradCAM(model, layer)
        >>> heatmap, pred_class, confidence = gradcam(input_tensor)
        >>> gradcam.remove_hooks()  # Important to prevent memory leaks
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layer."""
        # Forward hook to capture activations
        forward_hook = self.target_layer.register_forward_hook(self._save_activation)
        self._hooks.append(forward_hook)

        # Backward hook to capture gradients
        backward_hook = self.target_layer.register_full_backward_hook(self._save_gradient)
        self._hooks.append(backward_hook)

    def _save_activation(self, module: nn.Module, input: Tuple, output: torch.Tensor) -> None:
        """Hook callback to save layer activations."""
        self.activations = output.detach()

    def _save_gradient(self, module: nn.Module, grad_input: Tuple, grad_output: Tuple) -> None:
        """Hook callback to save gradients."""
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        """Remove registered hooks to prevent memory leaks.

        Important: Call this method when done with GradCAM to free GPU memory.
        """
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self.gradients = None
        self.activations = None

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """Generate GradCAM heatmap for input image.

        Args:
            input_tensor: Input image tensor (1, C, H, W). Batch size must be 1.
            target_class: Target class index. If None, uses predicted class.

        Returns:
            Tuple of:
                - heatmap: Normalized heatmap array (H, W) in [0, 1]
                - predicted_class: Class index of prediction
                - confidence: Softmax probability of predicted class

        Raises:
            ValueError: If batch size is not 1 or input dimensions are invalid.
            RuntimeError: If gradients/activations were not captured.
        """
        # Validate input dimensions
        if input_tensor.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B,C,H,W), got {input_tensor.dim()}D")
        if input_tensor.shape[0] != 1:
            raise ValueError(f"Batch size must be 1, got {input_tensor.shape[0]}")

        self.model.eval()

        # Ensure requires_grad for backward pass
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        # Get predicted class and confidence
        predicted_class = output.argmax(dim=1).item()
        confidence = probs[0, predicted_class].item()

        # Use predicted class if target not specified
        if target_class is None:
            target_class = predicted_class
        elif not (0 <= target_class < output.shape[1]):
            raise ValueError(
                f"target_class={target_class} out of range [0, {output.shape[1]-1}]"
            )

        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()

        # Validate gradients and activations were captured
        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Gradients or activations not captured. "
                "Ensure model is in eval mode and target layer is valid."
            )

        # Compute weights via global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # ReLU to keep only positive contributions
        cam = F.relu(cam)

        # Resize to input dimensions (224x224 typically)
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # Normalize to [0, 1] with epsilon for numerical stability
        cam = cam.squeeze()
        cam_max = cam.max()
        if cam_max > 1e-8:
            cam = cam / cam_max
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy(), predicted_class, confidence

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - remove hooks."""
        self.remove_hooks()
        return False


def calculate_pfs(
    heatmap: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Calculate Pulmonary Focus Score (PFS).

    PFS measures what fraction of the model's attention falls within
    the lung tissue region.

    PFS = sum(heatmap * mask) / sum(heatmap)

    Args:
        heatmap: GradCAM heatmap (H, W), values in [0, 1]
        mask: Binary lung mask (H, W), 1=lung tissue

    Returns:
        PFS score in [0, 1]:
            - 1.0 = Model focuses entirely on lungs
            - 0.5 = Equal focus on lung and non-lung regions
            - <0.5 = Model focuses more on non-lung areas
    """
    # Ensure heatmap is in valid range
    heatmap = np.clip(heatmap, 0, 1)

    # Handle zero heatmap
    total = heatmap.sum()
    if total == 0:
        return 0.0

    # Handle RGB masks (convert to grayscale)
    if len(mask.shape) == 3:
        mask = np.mean(mask, axis=2)

    # Normalize mask to [0, 1]
    if mask.max() > 1:
        mask = mask / 255.0

    # Resize mask to match heatmap if needed
    if mask.shape != heatmap.shape:
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((heatmap.shape[1], heatmap.shape[0]), Image.BILINEAR)
        mask = np.array(mask_pil) / 255.0

    # Calculate overlap
    overlap = (heatmap * mask).sum()

    return float(overlap / total)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlay heatmap on original image.

    Args:
        image: Original image (H, W) or (H, W, 3), values in [0, 255]
        heatmap: GradCAM heatmap (H, W), values in [0, 1]
        alpha: Transparency of heatmap overlay (0=image only, 1=heatmap only)
        colormap: Matplotlib colormap name ('jet', 'hot', 'viridis', etc.)

    Returns:
        Blended image (H, W, 3) as uint8
    """
    # Ensure image is RGB
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=2)
    elif image.shape[2] == 1:
        image = np.concatenate([image, image, image], axis=2)

    # Normalize image to [0, 255]
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # Resize heatmap to match image
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_pil = heatmap_pil.resize((image.shape[1], image.shape[0]), Image.BILINEAR)
        heatmap = np.array(heatmap_pil) / 255.0

    # Apply colormap
    cmap = getattr(cm, colormap, cm.jet)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Drop alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Blend images
    overlay = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)

    return overlay


def create_gradcam_visualization(
    image: np.ndarray,
    heatmap: np.ndarray,
    prediction: str,
    confidence: float,
    true_label: Optional[str] = None,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """Create a complete GradCAM visualization with annotations.

    Args:
        image: Original image (H, W) or (H, W, 3)
        heatmap: GradCAM heatmap (H, W) in [0, 1]
        prediction: Predicted class name
        confidence: Prediction confidence (0-1)
        true_label: Optional ground truth label
        alpha: Heatmap transparency
        colormap: Colormap name

    Returns:
        Visualization image (H, W, 3) as uint8
    """
    import cv2

    # Create overlay
    overlay = overlay_heatmap(image, heatmap, alpha=alpha, colormap=colormap)

    # Add text annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # Prediction text
    text = f"Pred: {prediction} ({confidence:.1%})"
    color = (0, 255, 0) if true_label is None or prediction == true_label else (0, 0, 255)
    cv2.putText(overlay, text, (10, 25), font, font_scale, color, thickness)

    # True label if provided
    if true_label is not None:
        text = f"True: {true_label}"
        cv2.putText(overlay, text, (10, 50), font, font_scale, (255, 255, 255), thickness)

    return overlay
