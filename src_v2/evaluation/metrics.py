"""
Metricas de evaluacion para landmark prediction
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src_v2.constants import (
    LANDMARK_NAMES,
    SYMMETRIC_PAIRS,
    DEFAULT_IMAGE_SIZE,
)


logger = logging.getLogger(__name__)


def _prepare_image_size_tensor(
    image_size: Any,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Normaliza image_size a un tensor (B, 1, 2) para escalar predicciones.

    Args:
        image_size: Puede ser int/float (lado unico), (W, H), lista de tamaños
            por muestra, o tensor ya existente.
        batch_size: Tamaño del batch para broadcasting.
        device: Dispositivo destino.
        dtype: Tipo de dato para el tensor final.

    Returns:
        Tensor de tamaño (B, 1, 2) con anchura y altura por muestra.
    """
    if image_size is None:
        size_tensor = torch.tensor(
            [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE], device=device, dtype=dtype
        )
        return size_tensor.view(1, 1, 2).expand(batch_size, 1, 2)

    if isinstance(image_size, (int, float)):
        size_tensor = torch.tensor(
            [image_size, image_size], device=device, dtype=dtype
        )
        return size_tensor.view(1, 1, 2).expand(batch_size, 1, 2)

    size_tensor = torch.as_tensor(image_size, device=device, dtype=dtype)

    if size_tensor.ndim == 0:
        size_tensor = size_tensor.repeat(2)

    if size_tensor.ndim == 1:
        if size_tensor.numel() == 1:
            size_tensor = size_tensor.repeat(2)
        if size_tensor.numel() != 2:
            raise ValueError("image_size vector debe tener 2 elementos (W, H)")
        size_tensor = size_tensor.view(1, 1, 2).expand(batch_size, 1, 2)
    elif size_tensor.ndim == 2:
        if size_tensor.shape[1] != 2:
            raise ValueError("image_size de forma (B, 2) esperado")
        if size_tensor.shape[0] == 1 and batch_size > 1:
            size_tensor = size_tensor.expand(batch_size, size_tensor.shape[1])
        elif size_tensor.shape[0] != batch_size:
            raise ValueError("image_size debe tener batch_size filas")
        size_tensor = size_tensor.unsqueeze(1)
    else:
        raise ValueError("image_size debe ser escalar, (2,) o (B, 2)")

    return size_tensor


def compute_pixel_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    image_size: Any = DEFAULT_IMAGE_SIZE
) -> torch.Tensor:
    """
    Calcula error euclidiano en pixeles.

    Args:
        pred: Predicciones (B, 30) o (B, 15, 2) en [0, 1]
        target: Ground truth misma forma
        image_size: Tamano de imagen para desnormalizar. Puede ser:
            - int/float (lado unico, asume imagen cuadrada)
            - Tupla/lista (width, height)
            - Lista/array/tensor de tamaños por muestra (B, 2)
            - None => usa DEFAULT_IMAGE_SIZE

    Returns:
        Tensor (B, 15) con error por landmark en pixeles
    """
    B = pred.shape[0]
    pred = pred.view(B, 15, 2)
    target = target.view(B, 15, 2)

    size_tensor = _prepare_image_size_tensor(
        image_size=image_size,
        batch_size=B,
        device=pred.device,
        dtype=pred.dtype,
    )

    pred = pred * size_tensor
    target = target * size_tensor

    errors = torch.norm(pred - target, dim=-1)  # (B, 15) o (B, 1, 15) si image_size tenia dims extra
    if errors.ndim == 3 and errors.shape[1] == 1:
        errors = errors.squeeze(1)
    return errors


def compute_error_per_landmark(
    pred: torch.Tensor,
    target: torch.Tensor,
    image_size: Any = DEFAULT_IMAGE_SIZE
) -> Dict[str, float]:
    """
    Calcula error promedio por landmark.

    Returns:
        Dict {'L1': error, 'L2': error, ...}
    """
    errors = compute_pixel_error(pred, target, image_size)  # (B, 15)
    mean_errors = errors.mean(dim=0).cpu().numpy()

    return {name: float(mean_errors[i]) for i, name in enumerate(LANDMARK_NAMES)}


def _get_batch_image_sizes(
    metas: List[dict],
    default_size: Any = DEFAULT_IMAGE_SIZE
) -> List[Tuple[float, float]]:
    """
    Extrae tamaños de imagen por muestra desde el metadata del DataLoader.

    Args:
        metas: Lista de metadata (de LandmarkDataset)
        default_size: Fallback si no hay original_size en meta

    Returns:
        Lista de tuplas (width, height) por muestra
    """
    if isinstance(default_size, (int, float)):
        fallback = (float(default_size), float(default_size))
    elif default_size is None:
        fallback = (float(DEFAULT_IMAGE_SIZE), float(DEFAULT_IMAGE_SIZE))
    else:
        fallback = tuple(default_size)

    sizes = []
    for meta in metas:
        if meta is None:
            sizes.append(fallback)
            continue
        size = meta.get("original_size")
        if size and len(size) == 2:
            sizes.append(tuple(size))
        else:
            sizes.append(fallback)
    return sizes


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    image_size: Any = DEFAULT_IMAGE_SIZE
) -> Dict[str, Any]:
    """
    Evaluacion completa del modelo.

    Args:
        model: Modelo a evaluar
        data_loader: DataLoader de test
        device: Dispositivo
        image_size: Tamano de imagen (fallback si meta no incluye original_size)

    Returns:
        Dict con metricas detalladas
    """
    model.eval()

    all_errors = []
    all_preds = []
    all_targets = []
    category_errors = defaultdict(list)

    for images, landmarks, metas in data_loader:
        images = images.to(device)
        landmarks = landmarks.to(device)

        outputs = model(images)

        # Error por muestra y landmark
        batch_sizes = _get_batch_image_sizes(metas, image_size)
        errors = compute_pixel_error(outputs, landmarks, batch_sizes)  # (B, 15)
        all_errors.append(errors.cpu())
        all_preds.append(outputs.cpu())
        all_targets.append(landmarks.cpu())

        # Agrupar por categoria
        for i, meta in enumerate(metas):
            category = meta['category']
            category_errors[category].append(errors[i].cpu())

    # Concatenar
    all_errors = torch.cat(all_errors, dim=0)  # (N, 15)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Metricas globales
    mean_error = all_errors.mean().item()
    std_error = all_errors.std().item()
    median_error = all_errors.median().item()

    # Error por landmark
    per_landmark = {
        name: {
            'mean': all_errors[:, i].mean().item(),
            'std': all_errors[:, i].std().item(),
            'median': all_errors[:, i].median().item(),
            'max': all_errors[:, i].max().item(),
        }
        for i, name in enumerate(LANDMARK_NAMES)
    }

    # Error por categoria
    per_category = {}
    for category, errors_list in category_errors.items():
        cat_errors = torch.stack(errors_list)  # (N_cat, 15)
        per_category[category] = {
            'mean': cat_errors.mean().item(),
            'std': cat_errors.std().item(),
            'count': len(errors_list)
        }

    # Percentiles
    flat_errors = all_errors.flatten()
    percentiles = {
        'p50': torch.quantile(flat_errors, 0.50).item(),
        'p75': torch.quantile(flat_errors, 0.75).item(),
        'p90': torch.quantile(flat_errors, 0.90).item(),
        'p95': torch.quantile(flat_errors, 0.95).item(),
    }

    return {
        'overall': {
            'mean': mean_error,
            'std': std_error,
            'median': median_error,
        },
        'per_landmark': per_landmark,
        'per_category': per_category,
        'percentiles': percentiles,
        'raw_errors': all_errors,
        'predictions': all_preds,
        'targets': all_targets,
    }


def compute_error_per_category(
    pred: torch.Tensor,
    target: torch.Tensor,
    categories: List[str],
    image_size: Any = DEFAULT_IMAGE_SIZE
) -> Dict[str, Dict[str, float]]:
    """
    Calcula error promedio por categoria.

    Args:
        pred: Predicciones (B, 30)
        target: Ground truth (B, 30)
        categories: Lista de categorias por muestra
        image_size: Tamano de imagen (escalar o por muestra)

    Returns:
        Dict {category: {'mean': error, 'std': std, 'count': n}}
    """
    errors = compute_pixel_error(pred, target, image_size)  # (B, 15)
    mean_per_sample = errors.mean(dim=1)  # (B,)

    category_stats = defaultdict(list)
    for error, cat in zip(mean_per_sample.tolist(), categories):
        category_stats[cat].append(error)

    results = {}
    for cat, cat_errors in category_stats.items():
        results[cat] = {
            'mean': np.mean(cat_errors),
            'std': np.std(cat_errors),
            'count': len(cat_errors)
        }

    return results


def generate_evaluation_report(metrics: Dict) -> str:
    """
    Genera reporte de evaluacion en texto.

    Args:
        metrics: Dict retornado por evaluate_model

    Returns:
        String con reporte formateado
    """
    lines = []
    lines.append("=" * 60)
    lines.append("EVALUATION REPORT")
    lines.append("=" * 60)

    # Metricas globales
    lines.append("\n--- Overall Metrics ---")
    overall = metrics['overall']
    lines.append(f"Mean Error:   {overall['mean']:.2f} px")
    lines.append(f"Std Error:    {overall['std']:.2f} px")
    lines.append(f"Median Error: {overall['median']:.2f} px")

    # Percentiles
    lines.append("\n--- Percentiles ---")
    for p, val in metrics['percentiles'].items():
        lines.append(f"  {p}: {val:.2f} px")

    # Por landmark
    lines.append("\n--- Error per Landmark ---")
    per_landmark = metrics['per_landmark']

    # Ordenar por error medio
    sorted_landmarks = sorted(per_landmark.items(), key=lambda x: x[1]['mean'])

    lines.append(f"{'Landmark':<8} {'Mean':>8} {'Std':>8} {'Median':>8} {'Max':>8}")
    lines.append("-" * 44)
    for name, stats in sorted_landmarks:
        lines.append(f"{name:<8} {stats['mean']:>8.2f} {stats['std']:>8.2f} "
                    f"{stats['median']:>8.2f} {stats['max']:>8.2f}")

    # Mejores y peores
    lines.append(f"\nBest:  {sorted_landmarks[0][0]} ({sorted_landmarks[0][1]['mean']:.2f} px)")
    lines.append(f"Worst: {sorted_landmarks[-1][0]} ({sorted_landmarks[-1][1]['mean']:.2f} px)")

    # Por categoria
    lines.append("\n--- Error per Category ---")
    for cat, stats in metrics['per_category'].items():
        lines.append(f"{cat}: {stats['mean']:.2f} +/- {stats['std']:.2f} px (n={stats['count']})")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


def compute_success_rate(
    errors: torch.Tensor,
    thresholds: List[float] = [5, 8, 10, 15]
) -> Dict[float, float]:
    """
    Calcula porcentaje de predicciones bajo cada umbral.

    Args:
        errors: Tensor de errores (N, 15) o (N*15,)
        thresholds: Lista de umbrales en pixeles

    Returns:
        Dict {threshold: percentage}
    """
    errors = errors.flatten()
    total = len(errors)

    return {
        thresh: (errors < thresh).sum().item() / total * 100
        for thresh in thresholds
    }


def _flip_landmarks_horizontal(landmarks: torch.Tensor) -> torch.Tensor:
    """
    Flip horizontal de landmarks normalizados [0,1].

    Args:
        landmarks: (B, 30) o (B, 15, 2) en [0,1]

    Returns:
        Landmarks con x reflejado e indices de pares intercambiados
    """
    B = landmarks.shape[0]
    landmarks = landmarks.view(B, 15, 2).clone()

    # Reflejar coordenada X
    landmarks[:, :, 0] = 1.0 - landmarks[:, :, 0]

    # Intercambiar pares simetricos
    for left, right in SYMMETRIC_PAIRS:
        tmp = landmarks[:, left].clone()
        landmarks[:, left] = landmarks[:, right]
        landmarks[:, right] = tmp

    return landmarks.view(B, 30)


@torch.no_grad()
def predict_with_tta(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    use_flip: bool = True
) -> torch.Tensor:
    """
    Prediccion con Test-Time Augmentation.

    Promedia predicciones de imagen original y flip horizontal.

    Args:
        model: Modelo
        images: Batch de imagenes (B, 3, H, W)
        device: Dispositivo
        use_flip: Usar flip horizontal

    Returns:
        Predicciones promediadas (B, 30) en [0,1]
    """
    model.eval()
    images = images.to(device)

    # Prediccion original
    pred_original = model(images)  # (B, 30)

    if not use_flip:
        return pred_original

    # Prediccion con flip
    images_flipped = torch.flip(images, dims=[3])  # Flip horizontal (W dimension)
    pred_flipped = model(images_flipped)

    # Revertir flip en predicciones
    pred_flipped = _flip_landmarks_horizontal(pred_flipped)

    # Promediar
    return (pred_original + pred_flipped) / 2


@torch.no_grad()
def evaluate_model_with_tta(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    image_size: Any = DEFAULT_IMAGE_SIZE
) -> Dict[str, Any]:
    """
    Evaluacion completa del modelo con TTA.

    Args:
        model: Modelo a evaluar
        data_loader: DataLoader de test
        device: Dispositivo
        image_size: Tamano de imagen (fallback si meta no incluye original_size)

    Returns:
        Dict con metricas detalladas
    """
    model.eval()

    all_errors = []
    all_preds = []
    all_targets = []
    category_errors = defaultdict(list)

    for images, landmarks, metas in data_loader:
        images = images.to(device)
        landmarks = landmarks.to(device)

        # Prediccion con TTA
        outputs = predict_with_tta(model, images, device, use_flip=True)

        # Error por muestra y landmark
        batch_sizes = _get_batch_image_sizes(metas, image_size)
        errors = compute_pixel_error(outputs, landmarks, batch_sizes)  # (B, 15)
        all_errors.append(errors.cpu())
        all_preds.append(outputs.cpu())
        all_targets.append(landmarks.cpu())

        # Agrupar por categoria
        for i, meta in enumerate(metas):
            category = meta['category']
            category_errors[category].append(errors[i].cpu())

    # Concatenar
    all_errors = torch.cat(all_errors, dim=0)  # (N, 15)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Metricas globales
    mean_error = all_errors.mean().item()
    std_error = all_errors.std().item()
    median_error = all_errors.median().item()

    # Error por landmark
    per_landmark = {
        name: {
            'mean': all_errors[:, i].mean().item(),
            'std': all_errors[:, i].std().item(),
            'median': all_errors[:, i].median().item(),
            'max': all_errors[:, i].max().item(),
        }
        for i, name in enumerate(LANDMARK_NAMES)
    }

    # Error por categoria
    per_category = {}
    for category, errors_list in category_errors.items():
        cat_errors = torch.stack(errors_list)  # (N_cat, 15)
        per_category[category] = {
            'mean': cat_errors.mean().item(),
            'std': cat_errors.std().item(),
            'count': len(errors_list)
        }

    # Percentiles
    flat_errors = all_errors.flatten()
    percentiles = {
        'p50': torch.quantile(flat_errors, 0.50).item(),
        'p75': torch.quantile(flat_errors, 0.75).item(),
        'p90': torch.quantile(flat_errors, 0.90).item(),
        'p95': torch.quantile(flat_errors, 0.95).item(),
    }

    return {
        'overall': {
            'mean': mean_error,
            'std': std_error,
            'median': median_error,
        },
        'per_landmark': per_landmark,
        'per_category': per_category,
        'percentiles': percentiles,
        'raw_errors': all_errors,
        'predictions': all_preds,
        'targets': all_targets,
    }
