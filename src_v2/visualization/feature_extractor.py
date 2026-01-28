"""
Feature Extractor for Neural Network Glass Box Visualization

This module provides tools for extracting intermediate feature maps from
PyTorch models using forward hooks. Designed for both landmark detection
and classification models in the COVID-19 detection pipeline.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


class FeatureExtractor:
    """Extract intermediate feature maps from PyTorch models using hooks.

    This class registers forward hooks on specified layers to capture their
    outputs during inference. Useful for visualization and interpretability.

    Example:
        >>> model = ResNet18Landmarks()
        >>> extractor = FeatureExtractor(model, ['backbone_conv.layer1', 'backbone_conv.layer4'])
        >>> with torch.no_grad():
        ...     output = model(image)
        ...     features = extractor.get_features()
        >>> print(features['backbone_conv.layer1'].shape)  # (batch, channels, height, width)

    Attributes:
        model: The PyTorch model to extract features from
        target_layers: List of layer names to extract features from
        features: Dictionary mapping layer names to their output tensors
        hooks: List of registered hook handles
    """

    def __init__(self, model: nn.Module, target_layers: Optional[List[str]] = None):
        """Initialize the feature extractor.

        Args:
            model: PyTorch model to extract features from
            target_layers: List of layer names to hook. If None, hooks all named modules.
                          Use dot notation for nested modules (e.g., 'backbone_conv.layer1')
        """
        self.model = model
        self.target_layers = target_layers or []
        self.features: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on target layers."""
        if not self.target_layers:
            # If no target layers specified, hook all named modules
            for name, module in self.model.named_modules():
                if name and not list(module.children()):  # Leaf modules only
                    hook = module.register_forward_hook(self._make_hook(name))
                    self.hooks.append(hook)
        else:
            # Hook only specified layers
            for target_name in self.target_layers:
                module = self._get_module_by_name(target_name)
                if module is not None:
                    hook = module.register_forward_hook(self._make_hook(target_name))
                    self.hooks.append(hook)
                else:
                    print(f"Warning: Layer '{target_name}' not found in model")

    def _make_hook(self, name: str):
        """Create a hook function that stores the output of a layer.

        Args:
            name: Name of the layer

        Returns:
            Hook function that captures layer output
        """
        def hook(module, input, output):
            # Detach and move to CPU to avoid memory issues
            if isinstance(output, torch.Tensor):
                self.features[name] = output.detach().cpu()
            elif isinstance(output, (tuple, list)):
                # Some layers return tuples (e.g., attention mechanisms)
                self.features[name] = tuple(o.detach().cpu() if isinstance(o, torch.Tensor) else o
                                           for o in output)
            else:
                self.features[name] = output
        return hook

    def _get_module_by_name(self, name: str) -> Optional[nn.Module]:
        """Get a module by its hierarchical name.

        Args:
            name: Dot-separated name (e.g., 'backbone_conv.layer1')

        Returns:
            The module if found, None otherwise
        """
        try:
            parts = name.split('.')
            module = self.model
            for part in parts:
                module = getattr(module, part)
            return module
        except AttributeError:
            return None

    def get_features(self) -> Dict[str, torch.Tensor]:
        """Get the extracted features.

        Returns:
            Dictionary mapping layer names to feature tensors
        """
        return self.features

    def clear_features(self):
        """Clear stored features to free memory."""
        self.features.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()


class LandmarkFeatureExtractor(FeatureExtractor):
    """Specialized feature extractor for ResNet18Landmarks model.

    Automatically hooks the most important layers for visualization:
    - backbone_conv.layer1 through layer4 (ResNet feature hierarchy)
    - coord_attention (if present)
    - avgpool (global pooled features)
    - head (final regression layers)
    """

    DEFAULT_LAYERS = [
        'backbone_conv.4',  # layer1
        'backbone_conv.5',  # layer2
        'backbone_conv.6',  # layer3
        'backbone_conv.7',  # layer4
        'coord_attention',
        'avgpool',
        'head',
    ]

    def __init__(self, model: nn.Module, additional_layers: Optional[List[str]] = None):
        """Initialize landmark-specific feature extractor.

        Args:
            model: ResNet18Landmarks model
            additional_layers: Extra layers to hook beyond defaults
        """
        layers = self.DEFAULT_LAYERS.copy()
        if additional_layers:
            layers.extend(additional_layers)

        super().__init__(model, target_layers=layers)

    def get_backbone_features(self) -> Dict[str, torch.Tensor]:
        """Get only the ResNet backbone features (layer1-4).

        Returns:
            Dictionary with layer1 through layer4 features (backbone_conv.4 through backbone_conv.7)
        """
        return {
            name: feat for name, feat in self.features.items()
            if name in ['backbone_conv.4', 'backbone_conv.5', 'backbone_conv.6', 'backbone_conv.7']
        }

    def get_attention_maps(self) -> Optional[torch.Tensor]:
        """Get coordinate attention maps if available.

        Returns:
            Attention tensor or None if not present
        """
        return self.features.get('coord_attention')


class ClassifierFeatureExtractor(FeatureExtractor):
    """Specialized feature extractor for ImageClassifier model.

    Automatically hooks the ResNet backbone layers for visualization.
    """

    DEFAULT_LAYERS = [
        'backbone.layer1',
        'backbone.layer2',
        'backbone.layer3',
        'backbone.layer4',
        'backbone.avgpool',
        'fc',
    ]

    def __init__(self, model: nn.Module, additional_layers: Optional[List[str]] = None):
        """Initialize classifier-specific feature extractor.

        Args:
            model: ImageClassifier model
            additional_layers: Extra layers to hook beyond defaults
        """
        layers = self.DEFAULT_LAYERS.copy()
        if additional_layers:
            layers.extend(additional_layers)

        super().__init__(model, target_layers=layers)

    def get_backbone_features(self) -> Dict[str, torch.Tensor]:
        """Get only the ResNet backbone features (layer1-4).

        Returns:
            Dictionary with layer1 through layer4 features
        """
        return {
            name: feat for name, feat in self.features.items()
            if name.startswith('backbone.layer')
        }


def extract_features_from_batch(
    model: nn.Module,
    images: torch.Tensor,
    target_layers: List[str],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Extract features from a batch of images using a model.

    Convenience function for one-shot feature extraction.

    Args:
        model: PyTorch model
        images: Input tensor (B, C, H, W)
        target_layers: List of layer names to extract
        device: Device to run inference on

    Returns:
        Tuple of (model_output, features_dict)
    """
    model = model.to(device)
    model.eval()

    extractor = FeatureExtractor(model, target_layers)

    with torch.no_grad():
        images = images.to(device)
        output = model(images)
        features = extractor.get_features()

    extractor.remove_hooks()

    return output, features


def get_available_layers(model: nn.Module, max_depth: int = 3) -> List[str]:
    """Get list of all available layer names in a model.

    Useful for discovering what layers can be hooked.

    Args:
        model: PyTorch model
        max_depth: Maximum nesting depth to show

    Returns:
        List of layer names
    """
    layers = []
    for name, module in model.named_modules():
        if name:  # Skip root module
            depth = name.count('.') + 1
            if depth <= max_depth:
                layers.append(name)
    return sorted(layers)
