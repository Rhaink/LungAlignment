"""
Dataset de landmarks para radiografias de torax

SESION 7: Agregado soporte para pesos por categoria (COVID oversampling)
"""

import logging

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import random
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Callable, List, Dict
from sklearn.model_selection import train_test_split

from src_v2.constants import DEFAULT_CATEGORY_WEIGHTS, ORIGINAL_IMAGE_SIZE
from .utils import load_coordinates_csv, get_image_path, get_landmarks_array
from .transforms import get_train_transforms, get_val_transforms


logger = logging.getLogger(__name__)


def compute_sample_weights(
    df: pd.DataFrame,
    category_weights: Optional[Dict[str, float]] = None
) -> torch.Tensor:
    """
    Calcula pesos por muestra basados en su categoria.

    Args:
        df: DataFrame con columna 'category'
        category_weights: Dict con peso por categoria. Si None, usa DEFAULT_CATEGORY_WEIGHTS.

    Returns:
        Tensor de pesos (una por muestra)
    """
    if category_weights is None:
        category_weights = DEFAULT_CATEGORY_WEIGHTS

    weights = []
    for _, row in df.iterrows():
        category = row['category']
        weight = category_weights.get(category, 1.0)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float)


class LandmarkDataset(Dataset):
    """
    Dataset de imagenes con 15 landmarks.

    Cada muestra contiene:
    - Imagen: tensor (3, H, W) normalizado
    - Landmarks: tensor (30,) en coordenadas normalizadas [0, 1]

    Atributos:
        df: DataFrame con datos
        data_root: Ruta raiz de datos
        transform: Transformacion a aplicar
        original_size: Tamano de imagenes originales (por defecto 299)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str,
        transform: Optional[Callable] = None,
        original_size: int = ORIGINAL_IMAGE_SIZE
    ):
        """
        Args:
            df: DataFrame con columnas image_name, category, L1_x, L1_y, ...
            data_root: Directorio raiz que contiene carpeta 'dataset/'
            transform: Transformacion a aplicar (None = validacion)
            original_size: Tamano de imagenes originales
        """
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform if transform else get_val_transforms()
        self.original_size = (original_size, original_size)
        self._size_warning_emitted = False

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Retorna imagen, landmarks y metadata.

        Returns:
            image: Tensor (3, 224, 224)
            landmarks: Tensor (30,) en [0, 1]
            meta: Dict con image_name y category

        Raises:
            FileNotFoundError: Si la imagen no existe
            IOError: Si la imagen esta corrupta o no se puede leer
        """
        row = self.df.iloc[idx]
        image_name = row['image_name']
        category = row['category']

        # Cargar imagen con manejo de errores
        image_path = get_image_path(image_name, category, self.data_root)

        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            logger.error("Image not found: %s (idx=%d)", image_path, idx)
            raise FileNotFoundError(f"Image not found: {image_path}")
        except (OSError, IOError) as e:
            logger.error("Corrupt or unreadable image: %s (idx=%d): %s", image_path, idx, e)
            raise IOError(f"Corrupt or unreadable image: {image_path}") from e

        # Obtener landmarks como array (15, 2)
        landmarks = get_landmarks_array(row)

        # Usar tamano real de la imagen para normalizar; alertar si difiere del esperado
        actual_size = image.size  # (width, height)
        if actual_size != self.original_size and not self._size_warning_emitted:
            logger.warning(
                "Image size mismatch (expected %s, got %s); using actual size for normalization",
                self.original_size,
                actual_size
            )
            self._size_warning_emitted = True

        # Aplicar transformaciones
        image_tensor, landmarks_tensor = self.transform(
            image, landmarks, actual_size
        )

        meta = {
            'image_name': image_name,
            'category': category,
            'idx': idx,
            'original_size': actual_size,
        }

        return image_tensor, landmarks_tensor, meta


def create_dataloaders(
    csv_path: str,
    data_root: str,
    batch_size: int = 16,
    val_split: float = 0.15,
    test_split: float = 0.10,
    num_workers: int = 4,
    random_state: int = 42,
    output_size: int = 224,
    flip_prob: float = 0.5,
    rotation_degrees: float = 10.0,
    use_clahe: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: int = 4,
    use_category_weights: bool = False,
    category_weights: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
    deterministic: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crea DataLoaders para train, val y test.

    Args:
        csv_path: Ruta al CSV de coordenadas
        data_root: Directorio raiz de datos
        batch_size: Tamano de batch
        val_split: Fraccion para validacion
        test_split: Fraccion para test
        num_workers: Workers para DataLoader
        random_state: Semilla para reproducibilidad
        output_size: Tamano de salida de imagenes
        flip_prob: Probabilidad de flip horizontal
        rotation_degrees: Rango de rotacion en grados
        use_clahe: Usar CLAHE para mejorar contraste local
        clahe_clip_limit: Limite de contraste para CLAHE (2.0 es estandar)
        clahe_tile_size: Tamano de tiles para CLAHE (4 es el estandar del proyecto)
        use_category_weights: Usar WeightedRandomSampler para sobremuestrear COVID
        category_weights: Pesos por categoria (si None, usa DEFAULT_CATEGORY_WEIGHTS)
        seed: Semilla para orden/augmentations en DataLoader (opcional)
        deterministic: Habilitar seeding determinista en workers y shuffle

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Cargar datos
    df = load_coordinates_csv(csv_path)

    # Split estratificado por categoria
    try:
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_split + test_split),
            stratify=df['category'],
            random_state=random_state
        )
    except ValueError as exc:
        logger.warning("Split train/temp no estratificado por counts bajos (%s)", exc)
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_split + test_split),
            stratify=None,
            random_state=random_state
        )

    val_ratio = val_split / (val_split + test_split)
    try:
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio),
            stratify=temp_df['category'],
            random_state=random_state
        )
    except ValueError as exc:
        logger.warning("Split val/test no estratificado por counts bajos (%s)", exc)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio),
            stratify=None,
            random_state=random_state
        )

    logger.info(
        "Dataset split: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
        len(train_df), len(train_df) / len(df) * 100,
        len(val_df), len(val_df) / len(df) * 100,
        len(test_df), len(test_df) / len(df) * 100
    )

    # Transformaciones
    train_transform = get_train_transforms(
        output_size=output_size,
        flip_prob=flip_prob,
        rotation_degrees=rotation_degrees,
        use_clahe=use_clahe,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_size=clahe_tile_size,
    )
    val_transform = get_val_transforms(
        output_size=output_size,
        use_clahe=use_clahe,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_size=clahe_tile_size,
    )

    # Datasets
    train_dataset = LandmarkDataset(train_df, data_root, train_transform)
    val_dataset = LandmarkDataset(val_df, data_root, val_transform)
    test_dataset = LandmarkDataset(test_df, data_root, val_transform)

    # Custom collate para incluir metadata
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        landmarks = torch.stack([item[1] for item in batch])
        metas = [item[2] for item in batch]
        return images, landmarks, metas

    worker_init_fn = None
    generator = None
    if deterministic and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

        def seed_worker(worker_id: int) -> None:
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        worker_init_fn = seed_worker

    # Crear sampler si se usan pesos por categoria
    train_sampler = None
    train_shuffle = True

    if use_category_weights:
        sample_weights = compute_sample_weights(train_df, category_weights)
        if generator is not None:
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_df),
                replacement=True,  # Permitir muestreo con reemplazo
                generator=generator
            )
        else:
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_df),
                replacement=True  # Permitir muestreo con reemplazo
            )
        train_shuffle = False  # No usar shuffle con sampler
        weights_dict = category_weights if category_weights else DEFAULT_CATEGORY_WEIGHTS
        weight_info = ", ".join(
            f"{cat}(w={w}, n={len(train_df[train_df['category'] == cat])})"
            for cat, w in weights_dict.items()
        )
        logger.info("Using WeightedRandomSampler: %s", weight_info)

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": collate_fn,
    }
    if worker_init_fn is not None:
        loader_kwargs["worker_init_fn"] = worker_init_fn
    if generator is not None:
        loader_kwargs["generator"] = generator

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs
    )

    return train_loader, val_loader, test_loader


def get_dataframe_splits(
    csv_path: str,
    val_split: float = 0.15,
    test_split: float = 0.10,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Retorna DataFrames separados para train, val, test.

    Util para analisis sin crear DataLoaders completos.
    Usa split estratificado por categoria para mantener proporciones.

    Args:
        csv_path: Ruta al archivo CSV de coordenadas
        val_split: Fraccion para validacion (default: 0.15)
        test_split: Fraccion para test (default: 0.10)
        random_state: Semilla para reproducibilidad (default: 42)

    Returns:
        Tupla (train_df, val_df, test_df) con DataFrames estratificados
    """
    df = load_coordinates_csv(csv_path)

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_split + test_split),
        stratify=df['category'],
        random_state=random_state
    )

    val_ratio = val_split / (val_split + test_split)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio),
        stratify=temp_df['category'],
        random_state=random_state
    )

    return train_df, val_df, test_df
