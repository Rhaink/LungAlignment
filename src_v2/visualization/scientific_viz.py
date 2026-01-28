"""
Generación de visualizaciones científicas para landmarks y warping.

Este módulo crea visualizaciones de calidad publicable que muestran:
- Radiografías originales a 299x299 (sin preprocesamiento)
- Predicciones exactas del ensemble best (3.61 px) usadas en el warping
- Malla de triangulación de Delaunay sobre landmarks predichos
- Splits idénticos al clasificador (train/val/test con seed=42)

Uso:
    python -m src_v2 generate-viz-dataset \\
        data/dataset/COVID-19_Radiography_Dataset \\
        outputs/viz_dataset
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from tqdm import tqdm

from src_v2.data.dataset import get_dataframe_splits

logger = logging.getLogger(__name__)


def load_prediction_cache(npz_path: Path) -> Dict[str, np.ndarray]:
    """
    Cargar cache NPZ de predicciones y crear mapping image_name -> landmarks.

    Args:
        npz_path: Ruta al archivo .npz con predicciones del ensemble

    Returns:
        Dict mapeando image_name (ej: "COVID-1") a landmarks (15, 2) en espacio 224x224

    Raises:
        FileNotFoundError: Si el archivo NPZ no existe
        KeyError: Si el NPZ no tiene las claves esperadas
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Cache de predicciones no encontrado: {npz_path}")

    logger.info(f"Cargando cache de predicciones desde {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    # Verificar claves necesarias
    if "landmarks" not in data or "image_paths" not in data:
        raise KeyError(f"El NPZ debe contener 'landmarks' y 'image_paths'. Claves: {list(data.keys())}")

    landmarks = data["landmarks"]  # Shape: (N, 15, 2)
    image_paths = data["image_paths"]  # Array de paths

    logger.info(f"Cargadas {len(landmarks)} predicciones")

    # Crear mapping image_name -> landmarks
    predictions_dict = {}
    for i, path in enumerate(image_paths):
        # Extraer nombre de imagen sin extensión
        # Path puede ser: "COVID/images/COVID-1.png" -> "COVID-1"
        image_name = Path(path).stem
        predictions_dict[image_name] = landmarks[i]

    logger.info(f"Creado mapping para {len(predictions_dict)} imágenes")
    return predictions_dict


def scale_landmarks_to_viz(
    landmarks_224: np.ndarray,
    viz_size: int = 299,
    model_size: int = 224
) -> np.ndarray:
    """
    Escalar landmarks del espacio del modelo (224x224) al espacio de visualización (299x299).

    Args:
        landmarks_224: Array (15, 2) con coordenadas en espacio 224x224
        viz_size: Tamaño de imagen de visualización (default: 299)
        model_size: Tamaño usado por el modelo (default: 224)

    Returns:
        Array (15, 2) con coordenadas escaladas a viz_size x viz_size
    """
    scale_factor = viz_size / model_size
    return landmarks_224 * scale_factor


def create_scientific_visualization(
    image: np.ndarray,
    landmarks: np.ndarray,
    triangles: List[List[int]],
    canonical_shape: np.ndarray,
    output_path: Path,
    dpi: int = 300,
    show_numbers: bool = True
) -> None:
    """
    Crear visualización científica de landmarks y triangulación.

    La visualización muestra:
    - Imagen base en escala de grises
    - Malla de triangulación de Delaunay (cyan) sobre landmarks predichos
    - Landmarks predichos (rojo, círculos numerados 1-15)
    - Leyenda profesional

    Args:
        image: Imagen en escala de grises (H, W) o (H, W, 1)
        landmarks: Predicciones de landmarks (15, 2) en coordenadas de la imagen
        triangles: Lista de triángulos como índices de vértices [[v1, v2, v3], ...]
        canonical_shape: [IGNORADO - mantener por compatibilidad]
        output_path: Ruta de salida para guardar la visualización
        dpi: Resolución de la imagen (default: 300)
        show_numbers: Mostrar números de landmarks (default: True)
    """
    # Asegurar que la imagen sea 2D
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image.squeeze(2)
        else:
            # Si es RGB, convertir a escala de grises
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Crear figura
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)

    # Imagen base en escala de grises
    ax.imshow(image, cmap='gray', aspect='auto')

    # 1. Dibujar malla de triangulación (cyan) sobre landmarks predichos
    for triangle in triangles:
        v1, v2, v3 = triangle
        # Cerrar el triángulo conectando v3 con v1
        x_coords = [landmarks[v1, 0], landmarks[v2, 0], landmarks[v3, 0], landmarks[v1, 0]]
        y_coords = [landmarks[v1, 1], landmarks[v2, 1], landmarks[v3, 1], landmarks[v1, 1]]
        ax.plot(x_coords, y_coords, color='#00FFFF', alpha=0.6, linewidth=1.5,
                label='Triangulación' if triangle == triangles[0] else '')

    # 2. Dibujar landmarks predichos (círculos rojos con borde blanco)
    for i, (x, y) in enumerate(landmarks):
        circle = Circle((x, y), radius=3, color='#FF4444', fill=True,
                       edgecolor='white', linewidth=1.5, zorder=10)
        ax.add_patch(circle)

        if show_numbers:
            # Numerar landmarks (1-indexed para coincidir con nomenclatura L1-L15)
            ax.text(x, y, str(i + 1), color='white', fontsize=8,
                   ha='center', va='center', weight='bold', zorder=11)

    # Añadir entrada de leyenda para landmarks (solo una vez)
    ax.scatter([], [], s=50, color='#FF4444', edgecolor='white', label='Landmarks', zorder=10)

    # Configuración de ejes y leyenda
    ax.axis('off')
    ax.legend(loc='upper right', framealpha=0.8, fontsize=10)

    # Ajustar límites para que la imagen ocupe todo el espacio
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)  # Invertir eje Y para que (0,0) esté arriba-izquierda

    # Guardar con alta calidad
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def generate_viz_dataset(
    input_dir: Path,
    output_dir: Path,
    predictions_npz: Path,
    canonical_json: Path,
    triangles_json: Path,
    csv_path: Path,
    viz_size: int = 299,
    model_size: int = 224,
    seed: int = 42,
    splits: Tuple[float, float, float] = (0.75, 0.125, 0.125),
    max_per_split: Optional[int] = None,
    dpi: int = 300,
    show_numbers: bool = True
) -> Dict:
    """
    Generar dataset completo de visualizaciones científicas.

    Pipeline:
    1. Cargar cache de predicciones NPZ
    2. Cargar forma canónica y triangulación
    3. Crear splits con get_dataframe_splits() (seed=42)
    4. Para cada split (train/val/test):
       - Para cada imagen en el split:
         * Cargar imagen original 299x299 (sin CLAHE, sin normalización)
         * Obtener predicciones del cache
         * Escalar landmarks y canonical shape a 299x299
         * Crear visualización
         * Guardar en {output}/{split}/{class}/{name}_viz.png
    5. Generar metadata.json con estadísticas

    Args:
        input_dir: Directorio del dataset original (COVID-19_Radiography_Dataset)
        output_dir: Directorio de salida para visualizaciones
        predictions_npz: Cache NPZ de predicciones del ensemble
        canonical_json: JSON con forma canónica de GPA
        triangles_json: JSON con triangulación de Delaunay
        csv_path: CSV maestro con coordenadas para crear splits
        viz_size: Tamaño de imagen de visualización (default: 299)
        model_size: Tamaño usado por el modelo (default: 224)
        seed: Seed para reproducibilidad de splits (default: 42)
        splits: Tupla (train, val, test) con ratios (default: 0.75, 0.125, 0.125)
        max_per_split: Máximo de imágenes por split (None = todas)
        dpi: Resolución de visualizaciones (default: 300)
        show_numbers: Mostrar números en landmarks (default: True)

    Returns:
        Dict con metadata del proceso (conteos, paths, parámetros)
    """
    logger.info("="*80)
    logger.info("GENERACIÓN DE DATASET DE VISUALIZACIONES CIENTÍFICAS")
    logger.info("="*80)

    # 1. Cargar cache de predicciones
    logger.info("\n[1/6] Cargando cache de predicciones...")
    predictions_dict = load_prediction_cache(predictions_npz)

    # 2. Cargar forma canónica
    logger.info("\n[2/6] Cargando forma canónica...")
    if not canonical_json.exists():
        raise FileNotFoundError(f"Forma canónica no encontrada: {canonical_json}")

    with open(canonical_json, 'r') as f:
        canonical_data = json.load(f)

    canonical_224 = np.array(canonical_data['canonical_shape_pixels'])  # (15, 2) en 224x224
    canonical_viz = scale_landmarks_to_viz(canonical_224, viz_size, model_size)
    logger.info(f"Forma canónica escalada de {model_size}x{model_size} a {viz_size}x{viz_size}")

    # 3. Cargar triangulación
    logger.info("\n[3/6] Cargando triangulación de Delaunay...")
    if not triangles_json.exists():
        raise FileNotFoundError(f"Triangulación no encontrada: {triangles_json}")

    with open(triangles_json, 'r') as f:
        triangles_data = json.load(f)

    triangles = triangles_data['triangles']
    logger.info(f"Cargados {len(triangles)} triángulos")

    # 4. Crear splits
    logger.info("\n[4/6] Creando splits estratificados...")
    train_split, val_split, test_split = splits
    train_df, val_df, test_df = get_dataframe_splits(
        str(csv_path),
        val_split=val_split,
        test_split=test_split,
        random_state=seed
    )

    splits_dict = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

    logger.info(f"Splits creados con seed={seed}:")
    logger.info(f"  Train: {len(train_df)} imágenes")
    logger.info(f"  Val:   {len(val_df)} imágenes")
    logger.info(f"  Test:  {len(test_df)} imágenes")

    # 5. Procesar cada split
    logger.info("\n[5/6] Generando visualizaciones...")

    metadata = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "predictions_source": {
            "path": str(predictions_npz),
        },
        "canonical_source": str(canonical_json),
        "triangles_source": str(triangles_json),
        "coordinate_spaces": {
            "predictions_original": f"{model_size}x{model_size}",
            "visualization": f"{viz_size}x{viz_size}",
            "scale_factor": viz_size / model_size
        },
        "splits": {
            "seed": seed,
            "ratios": {"train": train_split, "val": val_split, "test": test_split}
        },
        "visualization_style": {
            "image_size": viz_size,
            "dpi": dpi,
            "triangulation_color": "cyan",
            "triangulation_alpha": 0.6,
            "triangulation_linewidth": 1.5,
            "landmarks_color": "red",
            "landmarks_radius": 3,
            "landmarks_edgewidth": 1.5,
            "show_numbers": show_numbers
        },
        "processing_stats": {}
    }

    scale_factor = viz_size / model_size

    for split_name, split_df in splits_dict.items():
        logger.info(f"\n  Procesando split '{split_name}'...")

        # Limitar número de imágenes si se especifica
        if max_per_split is not None:
            split_df = split_df.head(max_per_split)
            logger.info(f"    Limitado a {max_per_split} imágenes")

        split_output = output_dir / split_name

        # Contadores por categoría
        category_counts = {}
        processed = 0
        skipped = 0

        # Procesar cada imagen
        for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"    {split_name}"):
            image_name = row['image_name']  # ej: "COVID-1"
            category = row['category']       # ej: "COVID"

            # Inicializar contador de categoría
            if category not in category_counts:
                category_counts[category] = 0

            # Path a imagen original
            # Estructura: {input_dir}/{category}/images/{image_name}.png
            image_path = input_dir / category / "images" / f"{image_name}.png"

            if not image_path.exists():
                logger.warning(f"    Imagen no encontrada: {image_path}")
                skipped += 1
                continue

            # Verificar que existan predicciones
            if image_name not in predictions_dict:
                logger.warning(f"    Sin predicciones para: {image_name}")
                skipped += 1
                continue

            # Cargar imagen original SIN preprocesamiento
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"    Error al cargar: {image_path}")
                skipped += 1
                continue

            # Resize a viz_size (solo resize, NO CLAHE)
            img_viz = cv2.resize(img, (viz_size, viz_size))

            # Obtener predicciones del cache (en espacio 224x224)
            landmarks_224 = predictions_dict[image_name]  # (15, 2)

            # Escalar a espacio de visualización
            landmarks_viz = scale_landmarks_to_viz(landmarks_224, viz_size, model_size)

            # Path de salida
            output_path = split_output / category / f"{image_name}_viz.png"

            # Crear visualización
            try:
                create_scientific_visualization(
                    img_viz,
                    landmarks_viz,
                    triangles,
                    canonical_viz,
                    output_path,
                    dpi=dpi,
                    show_numbers=show_numbers
                )
                category_counts[category] += 1
                processed += 1
            except Exception as e:
                logger.error(f"    Error al crear viz para {image_name}: {e}")
                skipped += 1
                continue

        # Guardar estadísticas del split
        metadata["processing_stats"][split_name] = {
            "total": processed,
            "skipped": skipped,
            "by_category": category_counts
        }

        logger.info(f"    Procesadas: {processed}, Omitidas: {skipped}")
        logger.info(f"    Por categoría: {category_counts}")

    # 6. Guardar metadata
    logger.info("\n[6/6] Guardando metadata...")
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata guardado en: {metadata_path}")

    # Resumen final
    logger.info("\n" + "="*80)
    logger.info("RESUMEN FINAL")
    logger.info("="*80)
    total_processed = sum(stats['total'] for stats in metadata["processing_stats"].values())
    total_skipped = sum(stats['skipped'] for stats in metadata["processing_stats"].values())
    logger.info(f"Total procesadas: {total_processed}")
    logger.info(f"Total omitidas:   {total_skipped}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*80)

    return metadata


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Configuración de ejemplo
    input_dir = Path("data/dataset/COVID-19_Radiography_Dataset")
    output_dir = Path("outputs/viz_dataset_test")
    predictions_npz = Path("outputs/landmark_predictions/session_warping/predictions.npz")
    canonical_json = Path("outputs/shape_analysis/canonical_shape_gpa.json")
    triangles_json = Path("outputs/shape_analysis/canonical_delaunay_triangles.json")
    csv_path = Path("data/coordenadas/coordenadas_maestro.csv")

    # Generar muestra pequeña para testing
    metadata = generate_viz_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        predictions_npz=predictions_npz,
        canonical_json=canonical_json,
        triangles_json=triangles_json,
        csv_path=csv_path,
        max_per_split=10  # Solo 10 por split para testing
    )

    print("\nMetadata generado:")
    print(json.dumps(metadata, indent=2))
