"""
Model manager singleton for loading and caching models.

Implements lazy loading to avoid reloading models on every inference.
"""
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from src_v2.models.resnet_landmark import create_model
from src_v2.models.classifier import load_classifier_checkpoint
from src_v2.processing.warp import piecewise_affine_warp, scale_landmarks_from_centroid
from src_v2.constants import SYMMETRIC_PAIRS

from .config import (
    LANDMARK_MODELS,
    CANONICAL_SHAPE,
    DELAUNAY_TRIANGLES,
    CLASSIFIER_CHECKPOINT,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_SIZE,
    MARGIN_SCALE,
    USE_FULL_COVERAGE,
    TTA_ENABLED,
    DEVICE_PREFERENCE,
)
from .gradcam_utils import generate_gradcam


def _apply_clahe_numpy(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 4
) -> np.ndarray:
    """
    Apply CLAHE to numpy grayscale image.

    Args:
        image: Grayscale image (H, W)
        clip_limit: Contrast limit
        tile_size: Tile grid size (tile_size x tile_size)

    Returns:
        CLAHE-enhanced image
    """
    # Ensure grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

    # Apply CLAHE
    return clahe.apply(image)


class ModelManager:
    """
    Singleton manager for all models and components.

    Provides:
    - Lazy loading of landmark models (ensemble of 4)
    - Canonical shape and triangulation loading
    - Classifier loading
    - Prediction with TTA
    - Warping functionality
    - Classification with GradCAM
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self, verbose: bool = True):
        """
        Initialize all models and components (lazy loading).

        Args:
            verbose: Print loading progress
        """
        if self._initialized:
            return

        # Determine device
        self.device = torch.device(
            'cuda' if DEVICE_PREFERENCE == 'cuda' and torch.cuda.is_available() else 'cpu'
        )

        if verbose:
            print(f"Dispositivo: {self.device}")
            if self.device.type == 'cuda':
                print(f"  GPU: {torch.cuda.get_device_name(0)}")

        # Load landmark ensemble
        if verbose:
            print("\nCargando ensemble de landmarks (4 modelos)...")
        self.landmark_models = self._load_landmark_ensemble(verbose=verbose)

        # Load canonical shape and triangulation
        if verbose:
            print("\nCargando forma canónica y triangulación...")
        self.canonical_shape, self.triangles = self._load_canonical_data()

        # Load classifier
        if verbose:
            print("\nCargando clasificador...")
        self.classifier, self.class_names = self._load_classifier()

        self._initialized = True

        if verbose:
            print("\n✓ Todos los modelos cargados exitosamente")

    def _load_landmark_ensemble(self, verbose: bool = True) -> List[nn.Module]:
        """Load ensemble of 4 landmark detection models."""
        models = []

        for i, model_path in enumerate(LANDMARK_MODELS, 1):
            if verbose:
                print(f"  [{i}/4] Cargando {model_path.name}...")

            if not model_path.exists():
                raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

            # Create model with same architecture as training
            model = create_model(
                pretrained=True,
                freeze_backbone=True,
                dropout_rate=0.3,
                hidden_dim=768,
                use_coord_attention=True,
                deep_head=True,
                device=self.device
            )

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            models.append(model)

        return models

    def _load_canonical_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load canonical shape and Delaunay triangulation.

        Returns:
            canonical_shape: (15, 2) coordinates in pixels
            triangles: (18, 3) triangle vertex indices
        """
        # Load canonical shape
        if not CANONICAL_SHAPE.exists():
            raise FileNotFoundError(f"Canonical shape no encontrado: {CANONICAL_SHAPE}")

        with open(CANONICAL_SHAPE, 'r') as f:
            canonical_data = json.load(f)

        canonical_shape = np.array(canonical_data['canonical_shape_pixels'])

        # Load triangulation
        if not DELAUNAY_TRIANGLES.exists():
            raise FileNotFoundError(f"Triangulación no encontrada: {DELAUNAY_TRIANGLES}")

        with open(DELAUNAY_TRIANGLES, 'r') as f:
            triangulation_data = json.load(f)

        triangles = np.array(triangulation_data['triangles'])

        return canonical_shape, triangles

    def _load_classifier(self) -> Tuple[nn.Module, List[str]]:
        """
        Load classifier model.

        Returns:
            model: Trained classifier
            class_names: List of class names
        """
        if not CLASSIFIER_CHECKPOINT.exists():
            raise FileNotFoundError(f"Clasificador no encontrado: {CLASSIFIER_CHECKPOINT}")

        model, metadata = load_classifier_checkpoint(
            str(CLASSIFIER_CHECKPOINT),
            device=self.device
        )

        model.eval()
        class_names = metadata.get('class_names', ['COVID', 'Normal', 'Viral_Pneumonia'])

        return model, class_names

    def predict_landmarks(
        self,
        image: np.ndarray,
        use_tta: bool = TTA_ENABLED,
        use_clahe: bool = True
    ) -> np.ndarray:
        """
        Predict landmarks using ensemble with TTA.

        Args:
            image: Input image (H, W) grayscale or (H, W, 3) RGB
            use_tta: Use Test-Time Augmentation (horizontal flip)
            use_clahe: Apply CLAHE preprocessing

        Returns:
            landmarks: (15, 2) coordinates in pixel space [0, 224]
        """
        self.initialize(verbose=False)

        # Ensure grayscale
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image.copy()

        # Apply CLAHE if requested
        if use_clahe:
            image_processed = _apply_clahe_numpy(
                image_gray,
                clip_limit=CLAHE_CLIP_LIMIT,
                tile_size=CLAHE_TILE_SIZE[0] if isinstance(CLAHE_TILE_SIZE, tuple) else CLAHE_TILE_SIZE
            )
        else:
            image_processed = image_gray

        # Resize to 224x224 if needed
        if image_processed.shape != (224, 224):
            image_processed = cv2.resize(image_processed, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Convert to tensor and normalize (ImageNet stats)
        # Convert grayscale to RGB for ResNet
        image_rgb = cv2.cvtColor(image_processed, cv2.COLOR_GRAY2RGB)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # Predict with ensemble
        predictions = []
        for model in self.landmark_models:
            if use_tta:
                pred = self._predict_with_tta(model, image_tensor)
            else:
                with torch.no_grad():
                    pred = model(image_tensor)

            predictions.append(pred)

        # Average ensemble predictions
        landmarks_norm = torch.stack(predictions).mean(dim=0)  # (1, 30)
        landmarks_norm = landmarks_norm.view(15, 2).cpu().numpy()

        # Denormalize to pixel coordinates
        landmarks_px = landmarks_norm * 224

        return landmarks_px

    def _predict_with_tta(self, model: nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Predict with Test-Time Augmentation (horizontal flip).

        Args:
            model: Landmark detection model
            image_tensor: Input tensor (1, 3, 224, 224)

        Returns:
            Averaged prediction (1, 30)
        """
        model.eval()

        with torch.no_grad():
            # Original prediction
            pred1 = model(image_tensor)

            # Flipped prediction
            image_flip = torch.flip(image_tensor, dims=[3])  # Horizontal flip
            pred2 = model(image_flip)

            # Correct flipped landmarks
            pred2 = pred2.view(-1, 15, 2)
            pred2[:, :, 0] = 1 - pred2[:, :, 0]  # Flip x coordinates

            # Swap symmetric pairs
            for left, right in SYMMETRIC_PAIRS:
                pred2[:, [left, right]] = pred2[:, [right, left]]

            pred2 = pred2.view(-1, 30)

            # Average
            return (pred1 + pred2) / 2

    def warp_image(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        margin_scale: float = MARGIN_SCALE,
        use_full_coverage: bool = USE_FULL_COVERAGE
    ) -> np.ndarray:
        """
        Apply piecewise affine warping to normalize geometry.

        Args:
            image: Input image (H, W) grayscale
            landmarks: Source landmarks (15, 2) in pixels
            margin_scale: Expansion margin from centroid
            use_full_coverage: Whether to use full coverage mode

        Returns:
            warped: Warped image (224, 224) grayscale
        """
        self.initialize(verbose=False)

        # Ensure grayscale
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image.copy()

        # Resize to 224x224 if needed
        if image_gray.shape != (224, 224):
            image_gray = cv2.resize(image_gray, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Scale landmarks from centroid
        scaled_landmarks = scale_landmarks_from_centroid(
            landmarks,
            scale=margin_scale
        )

        # Apply warping
        warped = piecewise_affine_warp(
            image=image_gray,
            source_landmarks=scaled_landmarks,
            target_landmarks=self.canonical_shape,
            triangles=self.triangles,
            use_full_coverage=use_full_coverage
        )

        return warped

    def classify_with_gradcam(
        self,
        warped_image: np.ndarray,
        target_class: Optional[int] = None,
        resize_heatmap: bool = True
    ) -> Tuple[dict, np.ndarray, int]:
        """
        Classify warped image and generate GradCAM heatmap.

        Args:
            warped_image: Warped image (224, 224) grayscale
            target_class: Target class for GradCAM (None = predicted class)
            resize_heatmap: Resize heatmap to match input size

        Returns:
            probabilities: Dict mapping class names to probabilities
            heatmap: GradCAM heatmap (224, 224) normalized [0, 1]
            predicted_class_idx: Index of predicted class
        """
        self.initialize(verbose=False)

        # Prepare image for classifier
        image_tensor = self._prepare_image_for_classifier(warped_image)

        # Generate GradCAM
        heatmap, predicted_class_idx, logits = generate_gradcam(
            self.classifier,
            image_tensor,
            target_layer='layer4',
            target_class=target_class,
            resize_to=(224, 224) if resize_heatmap else None
        )

        # Get probabilities
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

        # Map to class names
        probabilities = {
            name: float(prob)
            for name, prob in zip(self.class_names, probs)
        }

        return probabilities, heatmap, predicted_class_idx

    def _prepare_image_for_classifier(self, image: np.ndarray) -> torch.Tensor:
        """
        Prepare image tensor for classifier.

        Args:
            image: Input image (H, W) grayscale

        Returns:
            tensor: (1, 3, 224, 224) normalized tensor
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image.copy()

        # Resize if needed
        if image_gray.shape != (224, 224):
            image_gray = cv2.resize(image_gray, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Convert to RGB (classifier expects 3 channels)
        image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)

        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        return image_tensor

    def get_status(self) -> dict:
        """
        Get initialization status and model information.

        Returns:
            Status dictionary
        """
        status = {
            'initialized': self._initialized,
            'device': str(self.device) if self._initialized else 'N/A',
            'num_landmark_models': len(self.landmark_models) if self._initialized else 0,
            'classifier_loaded': self.classifier is not None if self._initialized else False,
            'class_names': self.class_names if self._initialized else [],
        }

        return status


# Convenience singleton instance
_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """Get singleton instance of ModelManager."""
    return _manager
