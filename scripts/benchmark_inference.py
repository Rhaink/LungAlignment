#!/usr/bin/env python3
"""
Script de benchmark para medir tiempos de inferencia del pipeline completo.

Mide el tiempo de cada módulo:
1. Preprocesamiento (CLAHE/SAHS)
2. Predicción de landmarks (individual y ensemble+TTA)
3. Normalización geométrica (warping)
4. Clasificación

Genera reporte con estadísticas (media, mediana, std, min, max).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src_v2.constants import (
    DEFAULT_CLAHE_CLIP_LIMIT,
    DEFAULT_CLAHE_TILE_SIZE,
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SYMMETRIC_PAIRS,
)
from src_v2.models import create_model, load_classifier_checkpoint
from src_v2.processing.warp import (
    piecewise_affine_warp,
    scale_landmarks_from_centroid,
)
def apply_clahe_numpy(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 4) -> np.ndarray:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) sobre numpy array.

    Parameters:
        image: Imagen RGB o escala de grises
        clip_limit: Limite de contraste para evitar amplificacion de ruido
        tile_size: Tamano de la cuadricula de tiles (tile_size x tile_size)

    Returns:
        Imagen con CLAHE aplicado
    """
    # Convertir a escala de grises si es RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Aplicar CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(gray)

    # Convertir de vuelta a RGB si es necesario
    if len(image.shape) == 3:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    return enhanced


def apply_sahs(image: np.ndarray) -> np.ndarray:
    """
    Aplica el algoritmo SAHS (Statistical Asymmetrical Histogram Stretching).

    SAHS calcula limites de estiramiento asimetricos basados en la media:
    - Factor 2.5 para el limite superior
    - Factor 2.0 para el limite inferior
    """
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    pixels = gray_image.astype(np.float64).ravel()
    gray_mean = np.mean(pixels)

    above_mean = pixels[pixels > gray_mean]
    below_or_equal_mean = pixels[pixels <= gray_mean]

    max_value = gray_mean
    min_value = gray_mean

    if above_mean.size > 0:
        std_above = np.sqrt(np.mean((above_mean - gray_mean) ** 2))
        max_value = gray_mean + 2.5 * std_above

    if below_or_equal_mean.size > 0:
        std_below = np.sqrt(np.mean((below_or_equal_mean - gray_mean) ** 2))
        min_value = gray_mean - 2.0 * std_below

    if max_value != min_value:
        enhanced = (255 / (max_value - min_value)) * (gray_image.astype(np.float64) - min_value)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    else:
        enhanced = gray_image

    # Convertir de vuelta a RGB si es necesario
    if len(image.shape) > 2:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    return enhanced


class Timer:
    """Contexto para medir tiempo de ejecución."""

    def __init__(self):
        self.times = []
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start_time
        self.times.append(elapsed)

    def stats(self):
        """Retorna estadísticas de tiempos medidos."""
        if not self.times:
            return {}
        times_ms = np.array(self.times) * 1000  # Convertir a ms
        return {
            "mean": float(np.mean(times_ms)),
            "median": float(np.median(times_ms)),
            "std": float(np.std(times_ms)),
            "min": float(np.min(times_ms)),
            "max": float(np.max(times_ms)),
            "total": float(np.sum(times_ms)),
            "count": len(times_ms)
        }


def load_models(ensemble_config_path: Path, device: str):
    """Carga modelos del ensemble desde config."""
    with open(ensemble_config_path) as f:
        config = json.load(f)

    models = []
    for ckpt_path in config["models"]:
        # Crear modelo con architecture correcta (CoordAtention + deep head)
        model = create_model(
            pretrained=False,
            use_coord_attention=True,
            deep_head=True
        )

        # Cargar checkpoint
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Cargar estado (probar diferentes estructuras)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)

        model = model.to(device)
        model.eval()
        models.append(model)

    return models


def load_classifier(classifier_path: Path, device: str):
    """Carga el clasificador."""
    device_obj = torch.device(device)
    model, metadata = load_classifier_checkpoint(str(classifier_path), device=device_obj)
    model.eval()
    return model


def load_canonical_shape(gpa_output_dir: Path):
    """Carga la forma canónica desde GPA."""
    shape_path = gpa_output_dir / "canonical_shape.npy"
    return np.load(shape_path)


def preprocess_clahe(image: np.ndarray) -> np.ndarray:
    """Aplica CLAHE."""
    return apply_clahe_numpy(
        image,
        clip_limit=DEFAULT_CLAHE_CLIP_LIMIT,
        tile_size=DEFAULT_CLAHE_TILE_SIZE
    )


def preprocess_sahs(image: np.ndarray) -> np.ndarray:
    """Aplica SAHS."""
    return apply_sahs(image)


def predict_landmarks_single(model, image: np.ndarray, device: str) -> np.ndarray:
    """Predice landmarks con un modelo."""
    # Normalizar con ImageNet stats
    img_normalized = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        coords = model(img_tensor)

    coords = coords.cpu().numpy().reshape(15, 2)
    return coords * DEFAULT_IMAGE_SIZE  # Desnormalizar


def predict_landmarks_ensemble_tta(
    models,
    image: np.ndarray,
    device: str
) -> np.ndarray:
    """Predice landmarks con ensemble + TTA."""
    all_preds = []

    # Original
    for model in models:
        coords = predict_landmarks_single(model, image, device)
        all_preds.append(coords)

    # Flipped (TTA)
    image_flipped = np.fliplr(image).copy()
    for model in models:
        coords = predict_landmarks_single(model, image_flipped, device)
        # Flip back coordinates
        coords[:, 0] = DEFAULT_IMAGE_SIZE - coords[:, 0]
        # Swap symmetric pairs
        for left, right in SYMMETRIC_PAIRS:
            coords[[left, right]] = coords[[right, left]]
        all_preds.append(coords)

    # Promedio
    return np.mean(all_preds, axis=0)


def warp_image(
    image: np.ndarray,
    landmarks: np.ndarray,
    canonical_shape: np.ndarray,
    margin_scale: float = 1.05
) -> np.ndarray:
    """Aplica warping."""
    landmarks_scaled = scale_landmarks_from_centroid(landmarks, margin_scale)
    canonical_scaled = scale_landmarks_from_centroid(canonical_shape, margin_scale)

    warped = piecewise_affine_warp(
        image,
        landmarks_scaled,
        canonical_scaled,
        output_size=DEFAULT_IMAGE_SIZE
    )
    return warped


def classify_image(model, image: np.ndarray, device: str) -> np.ndarray:
    """Clasifica imagen."""
    img_normalized = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)

    return probs.cpu().numpy()[0]


def benchmark_pipeline(
    sample_images: list[Path],
    ensemble_models: list,
    classifier_model,
    canonical_shape: np.ndarray,
    device: str,
    warmup_runs: int = 5
):
    """Ejecuta benchmark completo del pipeline."""

    # Timers para cada etapa
    timer_clahe = Timer()
    timer_sahs = Timer()
    timer_landmark_single = Timer()
    timer_landmark_ensemble_tta = Timer()
    timer_warp = Timer()
    timer_classify = Timer()
    timer_total = Timer()

    print(f"\n🔥 Warmup: {warmup_runs} iteraciones...")
    # Warmup para estabilizar GPU
    warmup_image = cv2.imread(str(sample_images[0]))
    warmup_image = cv2.cvtColor(warmup_image, cv2.COLOR_BGR2RGB)
    warmup_image = cv2.resize(warmup_image, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))

    for _ in range(warmup_runs):
        img_clahe = preprocess_clahe(warmup_image)
        coords = predict_landmarks_ensemble_tta(ensemble_models, img_clahe, device)
        warped = warp_image(warmup_image, coords, canonical_shape)
        img_sahs = preprocess_sahs(warped)
        _ = classify_image(classifier_model, img_sahs, device)

    print(f"\n📊 Benchmarking sobre {len(sample_images)} imágenes...\n")

    for img_path in tqdm(sample_images, desc="Procesando"):
        # Leer imagen
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))

        with timer_total:
            # 1. CLAHE (para landmarks)
            with timer_clahe:
                img_clahe = preprocess_clahe(image)

            # 2. Predicción landmarks (single model)
            with timer_landmark_single:
                coords_single = predict_landmarks_single(
                    ensemble_models[0],
                    img_clahe,
                    device
                )

            # 3. Predicción landmarks (ensemble + TTA)
            with timer_landmark_ensemble_tta:
                coords_ensemble = predict_landmarks_ensemble_tta(
                    ensemble_models,
                    img_clahe,
                    device
                )

            # 4. Warping
            with timer_warp:
                warped = warp_image(image, coords_ensemble, canonical_shape)

            # 5. SAHS (para clasificación)
            with timer_sahs:
                img_sahs = preprocess_sahs(warped)

            # 6. Clasificación
            with timer_classify:
                probs = classify_image(classifier_model, img_sahs, device)

    # Recopilar estadísticas
    results = {
        "system_info": {
            "device": device,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "num_images": len(sample_images),
            "warmup_runs": warmup_runs
        },
        "timings": {
            "preprocessing_clahe_ms": timer_clahe.stats(),
            "preprocessing_sahs_ms": timer_sahs.stats(),
            "landmarks_single_model_ms": timer_landmark_single.stats(),
            "landmarks_ensemble_tta_ms": timer_landmark_ensemble_tta.stats(),
            "warping_ms": timer_warp.stats(),
            "classification_ms": timer_classify.stats(),
            "total_pipeline_ms": timer_total.stats()
        }
    }

    # Calcular throughput
    total_time_sec = results["timings"]["total_pipeline_ms"]["total"] / 1000
    results["throughput"] = {
        "images_per_second": len(sample_images) / total_time_sec if total_time_sec > 0 else 0,
        "seconds_per_image": total_time_sec / len(sample_images) if len(sample_images) > 0 else 0
    }

    return results


def print_report(results: dict):
    """Imprime reporte formateado."""
    print("\n" + "="*70)
    print("📊 REPORTE DE BENCHMARK - PIPELINE DE INFERENCIA")
    print("="*70)

    print(f"\n🖥️  SISTEMA:")
    print(f"  Device: {results['system_info']['device']}")
    print(f"  PyTorch: {results['system_info']['torch_version']}")
    print(f"  CUDA available: {results['system_info']['cuda_available']}")
    print(f"  Imágenes procesadas: {results['system_info']['num_images']}")

    print(f"\n⏱️  TIEMPOS POR MÓDULO (ms):")
    print(f"  {'Módulo':<35} {'Media':<10} {'Mediana':<10} {'Std':<10}")
    print(f"  {'-'*65}")

    timings = results["timings"]
    modules = [
        ("Preprocesamiento CLAHE", "preprocessing_clahe_ms"),
        ("Predicción Landmarks (1 modelo)", "landmarks_single_model_ms"),
        ("Predicción Landmarks (Ensemble+TTA)", "landmarks_ensemble_tta_ms"),
        ("Normalización Geométrica", "warping_ms"),
        ("Preprocesamiento SAHS", "preprocessing_sahs_ms"),
        ("Clasificación", "classification_ms"),
        ("PIPELINE COMPLETO", "total_pipeline_ms"),
    ]

    for name, key in modules:
        stats = timings[key]
        print(f"  {name:<35} {stats['mean']:>8.2f}  {stats['median']:>8.2f}  {stats['std']:>8.2f}")

    print(f"\n🚀 THROUGHPUT:")
    print(f"  Imágenes/segundo: {results['throughput']['images_per_second']:.2f}")
    print(f"  Segundos/imagen:  {results['throughput']['seconds_per_image']:.3f}")

    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark de tiempos de inferencia del pipeline completo"
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=Path("data/dataset/COVID-19_Radiography_Dataset/test"),
        help="Directorio con imágenes de muestra"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Número de imágenes a procesar para el benchmark"
    )
    parser.add_argument(
        "--ensemble-config",
        type=Path,
        default=Path("configs/ensemble_best.json"),
        help="Config del ensemble de landmarks"
    )
    parser.add_argument(
        "--classifier-path",
        type=Path,
        default=Path("outputs/classifier_warped_lung_best/best_classifier.pt"),
        help="Path al clasificador entrenado"
    )
    parser.add_argument(
        "--gpa-output-dir",
        type=Path,
        default=Path("outputs/shape_analysis"),
        help="Directorio con la forma canónica (GPA)"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/benchmark_results.json"),
        help="Path para guardar resultados JSON"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Número de iteraciones de warmup"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)"
    )

    args = parser.parse_args()

    print(f"\n🔧 Configuración del benchmark:")
    print(f"  Device: {args.device}")
    print(f"  Muestras: {args.num_samples}")
    print(f"  Warmup: {args.warmup}")

    # Buscar imágenes de muestra
    print(f"\n📁 Buscando imágenes en {args.sample_dir}...")

    all_images = []
    for class_dir in args.sample_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.png"))
            all_images.extend(images)

    if len(all_images) < args.num_samples:
        print(f"⚠️  Solo se encontraron {len(all_images)} imágenes, usando todas.")
        sample_images = all_images
    else:
        import random
        random.seed(42)
        sample_images = random.sample(all_images, args.num_samples)

    print(f"✓ Usando {len(sample_images)} imágenes para benchmark")

    # Cargar modelos
    print(f"\n🔄 Cargando modelos...")
    ensemble_models = load_models(args.ensemble_config, args.device)
    print(f"✓ Ensemble cargado: {len(ensemble_models)} modelos")

    classifier = load_classifier(args.classifier_path, args.device)
    print(f"✓ Clasificador cargado")

    canonical_shape = load_canonical_shape(args.gpa_output_dir)
    print(f"✓ Forma canónica cargada: {canonical_shape.shape}")

    # Ejecutar benchmark
    results = benchmark_pipeline(
        sample_images,
        ensemble_models,
        classifier,
        canonical_shape,
        args.device,
        warmup_runs=args.warmup
    )

    # Imprimir reporte
    print_report(results)

    # Guardar resultados
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Resultados guardados en: {args.output_json}")


if __name__ == "__main__":
    main()
