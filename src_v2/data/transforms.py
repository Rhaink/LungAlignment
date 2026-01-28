"""
Transformaciones y augmentations para landmarks.

IMPORTANTE: El flip horizontal debe intercambiar indices de pares simetricos,
no solo reflejar coordenadas.

SESION 7: Agregado CLAHE para mejorar visibilidad en imagenes COVID.
"""

import logging
import numpy as np
import torch
from PIL import Image
from typing import Tuple, Dict, Any, Optional
import torchvision.transforms.functional as TF
import random
import cv2

from src_v2.constants import (
    SYMMETRIC_PAIRS,
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEFAULT_CLAHE_CLIP_LIMIT,
    DEFAULT_CLAHE_TILE_SIZE,
    DEFAULT_FLIP_PROB,
    DEFAULT_ROTATION_DEGREES,
)

logger = logging.getLogger(__name__)


def apply_clahe(
    image: Image.Image,
    clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
    tile_grid_size: Tuple[int, int] = (DEFAULT_CLAHE_TILE_SIZE, DEFAULT_CLAHE_TILE_SIZE)
) -> Image.Image:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).

    CLAHE mejora el contraste local, especialmente util para:
    - Radiografias con consolidaciones pulmonares (COVID)
    - Zonas con bajo contraste donde los landmarks son dificiles de ver

    Args:
        image: Imagen PIL en RGB
        clip_limit: Limite de contraste (2.0 es estandar, mayor = mas contraste)
        tile_grid_size: Tamano de tiles para ecualizacion local

    Returns:
        Imagen PIL con CLAHE aplicado
    """
    # Convertir PIL a numpy
    img_array = np.array(image)

    # Convertir a LAB (mejor que aplicar a cada canal RGB)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # RGB -> LAB
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Aplicar CLAHE solo al canal L (luminancia)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_clahe = clahe.apply(l_channel)

        # Recombinar canales
        lab_clahe = cv2.merge([l_clahe, a_channel, b_channel])

        # LAB -> RGB
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    else:
        # Imagen en escala de grises
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        result = clahe.apply(img_array)
        # Convertir a RGB replicando canales
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(result)


class LandmarkTransform:
    """
    Transformacion base para imagen + landmarks.
    Las coordenadas se mantienen normalizadas en [0, 1].
    """

    def __init__(
        self,
        output_size: int = DEFAULT_IMAGE_SIZE,
        normalize_mean: Tuple[float, float, float] = IMAGENET_MEAN,
        normalize_std: Tuple[float, float, float] = IMAGENET_STD,
        use_clahe: bool = False,
        clahe_clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
        clahe_tile_size: int = DEFAULT_CLAHE_TILE_SIZE,
    ):
        self.output_size = output_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = (clahe_tile_size, clahe_tile_size)

    def resize_image(self, image: Image.Image) -> Image.Image:
        """Redimensiona imagen a output_size x output_size."""
        return image.resize((self.output_size, self.output_size), Image.BILINEAR)

    def apply_clahe_if_enabled(self, image: Image.Image) -> Image.Image:
        """Aplica CLAHE si esta habilitado."""
        if self.use_clahe:
            return apply_clahe(image, self.clahe_clip_limit, self.clahe_tile_size)
        return image

    def normalize_coords(
        self,
        landmarks: np.ndarray,
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Normaliza coordenadas de [0, original_size] a [0, 1].

        Args:
            landmarks: Array (15, 2) en pixeles originales
            original_size: (width, height) de imagen original

        Returns:
            Array (15, 2) normalizado a [0, 1]
        """
        landmarks = landmarks.copy().astype(np.float32)
        landmarks[:, 0] /= original_size[0]  # x / width
        landmarks[:, 1] /= original_size[1]  # y / height
        return np.clip(landmarks, 0, 1)

    def image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convierte imagen PIL a tensor normalizado."""
        tensor = TF.to_tensor(image)  # [0, 255] -> [0, 1]
        tensor = TF.normalize(tensor, self.normalize_mean, self.normalize_std)
        return tensor

    def landmarks_to_tensor(self, landmarks: np.ndarray) -> torch.Tensor:
        """Convierte landmarks a tensor flat (30,)."""
        return torch.tensor(landmarks.flatten(), dtype=torch.float32)


class TrainTransform(LandmarkTransform):
    """
    Transformaciones para entrenamiento con augmentation.
    """

    def __init__(
        self,
        output_size: int = DEFAULT_IMAGE_SIZE,
        flip_prob: float = DEFAULT_FLIP_PROB,
        rotation_degrees: float = DEFAULT_ROTATION_DEGREES,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        use_clahe: bool = False,
        clahe_clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
        clahe_tile_size: int = DEFAULT_CLAHE_TILE_SIZE,
        **kwargs
    ):
        super().__init__(
            output_size,
            use_clahe=use_clahe,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_size=clahe_tile_size,
            **kwargs
        )
        self.flip_prob = flip_prob
        self.rotation_degrees = rotation_degrees
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def horizontal_flip(
        self,
        image: Image.Image,
        landmarks: np.ndarray
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Flip horizontal CORRECTO para landmarks.

        1. Flip imagen
        2. Reflejar coordenadas X: new_x = 1 - x (en coords normalizadas)
        3. INTERCAMBIAR indices de pares simetricos

        Args:
            image: Imagen PIL
            landmarks: Array (15, 2) en coordenadas normalizadas [0, 1]

        Returns:
            Imagen y landmarks transformados
        """
        # 1. Flip imagen
        image = TF.hflip(image)

        # 2. Reflejar coordenadas X
        landmarks = landmarks.copy()
        landmarks[:, 0] = 1.0 - landmarks[:, 0]

        # 3. INTERCAMBIAR pares simetricos
        for left, right in SYMMETRIC_PAIRS:
            landmarks[left], landmarks[right] = landmarks[right].copy(), landmarks[left].copy()

        return image, landmarks

    def rotate(
        self,
        image: Image.Image,
        landmarks: np.ndarray,
        angle: float
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Rotacion de imagen y landmarks.

        Args:
            image: Imagen PIL
            landmarks: Array (15, 2) en coordenadas normalizadas [0, 1]
            angle: Angulo en grados (positivo = antihorario)

        Returns:
            Imagen y landmarks rotados
        """
        # Rotar imagen
        image = TF.rotate(image, angle, fill=0)

        # Rotar landmarks
        # Centro de rotacion: (0.5, 0.5) en coords normalizadas
        landmarks = landmarks.copy()
        cx, cy = 0.5, 0.5
        angle_rad = np.radians(-angle)  # Negativo porque rotamos coords inversamente

        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        # Trasladar al origen, rotar, trasladar de vuelta
        x = landmarks[:, 0] - cx
        y = landmarks[:, 1] - cy

        new_x = x * cos_a - y * sin_a + cx
        new_y = x * sin_a + y * cos_a + cy

        landmarks[:, 0] = np.clip(new_x, 0, 1)
        landmarks[:, 1] = np.clip(new_y, 0, 1)

        return image, landmarks

    def color_jitter(self, image: Image.Image) -> Image.Image:
        """Aplica variaciones de brillo y contraste."""
        brightness = random.uniform(*self.brightness_range)
        contrast = random.uniform(*self.contrast_range)

        image = TF.adjust_brightness(image, brightness)
        image = TF.adjust_contrast(image, contrast)

        return image

    def __call__(
        self,
        image: Image.Image,
        landmarks: np.ndarray,
        original_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aplica transformaciones de entrenamiento.

        Args:
            image: Imagen PIL original
            landmarks: Array (15, 2) en pixeles originales
            original_size: (width, height) original

        Returns:
            (image_tensor, landmarks_tensor)
        """
        # Normalizar coordenadas a [0, 1] basado en tamano original
        landmarks = self.normalize_coords(landmarks, original_size)

        # Aplicar CLAHE ANTES del resize (mejor calidad con resolucion original)
        image = self.apply_clahe_if_enabled(image)

        # Redimensionar imagen
        image = self.resize_image(image)

        # Augmentations
        if random.random() < self.flip_prob:
            image, landmarks = self.horizontal_flip(image, landmarks)

        if self.rotation_degrees > 0:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            image, landmarks = self.rotate(image, landmarks, angle)

        image = self.color_jitter(image)

        # Convertir a tensores
        image_tensor = self.image_to_tensor(image)
        landmarks_tensor = self.landmarks_to_tensor(landmarks)

        return image_tensor, landmarks_tensor


class ValTransform(LandmarkTransform):
    """
    Transformaciones para validacion/test (sin augmentation).
    """

    def __init__(
        self,
        output_size: int = DEFAULT_IMAGE_SIZE,
        use_clahe: bool = False,
        clahe_clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
        clahe_tile_size: int = DEFAULT_CLAHE_TILE_SIZE,
        **kwargs
    ):
        super().__init__(
            output_size,
            use_clahe=use_clahe,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_size=clahe_tile_size,
            **kwargs
        )

    def __call__(
        self,
        image: Image.Image,
        landmarks: np.ndarray,
        original_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aplica transformaciones de validacion.

        Args:
            image: Imagen PIL original
            landmarks: Array (15, 2) en pixeles originales
            original_size: (width, height) original

        Returns:
            (image_tensor, landmarks_tensor)
        """
        # Normalizar coordenadas a [0, 1]
        landmarks = self.normalize_coords(landmarks, original_size)

        # Aplicar CLAHE si esta habilitado
        image = self.apply_clahe_if_enabled(image)

        # Redimensionar imagen
        image = self.resize_image(image)

        # Convertir a tensores
        image_tensor = self.image_to_tensor(image)
        landmarks_tensor = self.landmarks_to_tensor(landmarks)

        return image_tensor, landmarks_tensor


def get_train_transforms(
    output_size: int = DEFAULT_IMAGE_SIZE,
    flip_prob: float = DEFAULT_FLIP_PROB,
    rotation_degrees: float = DEFAULT_ROTATION_DEGREES,
    use_clahe: bool = False,
    clahe_clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
    clahe_tile_size: int = DEFAULT_CLAHE_TILE_SIZE,
    **kwargs
) -> TrainTransform:
    """Factory para transformaciones de entrenamiento."""
    return TrainTransform(
        output_size=output_size,
        flip_prob=flip_prob,
        rotation_degrees=rotation_degrees,
        use_clahe=use_clahe,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_size=clahe_tile_size,
        **kwargs
    )


def get_val_transforms(
    output_size: int = DEFAULT_IMAGE_SIZE,
    use_clahe: bool = False,
    clahe_clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
    clahe_tile_size: int = DEFAULT_CLAHE_TILE_SIZE,
    **kwargs
) -> ValTransform:
    """Factory para transformaciones de validacion."""
    return ValTransform(
        output_size=output_size,
        use_clahe=use_clahe,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_size=clahe_tile_size,
        **kwargs
    )
