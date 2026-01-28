#!/usr/bin/env python3
"""
Script para extraer predicciones del ensemble y generar triangulacion de Delaunay.

Sesion 18: Extraccion de Predicciones y Triangulacion de Delaunay

Este script:
1. Carga los 4 modelos del ensemble final
2. Extrae predicciones para TODAS las imagenes del test set (96 imagenes)
3. Aplica TTA (original + flip horizontal)
4. Guarda predicciones en multiples formatos (CSV, JSON, NPZ)
5. Genera triangulacion de Delaunay
6. Crea visualizaciones

Uso:
    python scripts/extract_predictions.py
    python scripts/extract_predictions.py --output-dir outputs/predictions_custom
    python scripts/extract_predictions.py --visualize-only
    python scripts/extract_predictions.py --config configs/ensemble_best.json
    python scripts/extract_predictions.py --models m1.pt m2.pt m3.pt m4.pt
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from src_v2.data.dataset import create_dataloaders
from src_v2.models.resnet_landmark import create_model
from src_v2.data.utils import get_image_path


# ==============================================================================
# CONFIGURACION
# ==============================================================================

# Checkpoints del ensemble (4 modelos)
DEFAULT_ENSEMBLE_CHECKPOINTS = [
    'checkpoints/session10/ensemble/seed123/final_model.pt',
    'checkpoints/session10/ensemble/seed456/final_model.pt',
    'checkpoints/session13/seed321/final_model.pt',
    'checkpoints/session13/seed789/final_model.pt',
]

# Pares simetricos (indices 0-based)
SYMMETRIC_PAIRS = [(2, 3), (4, 5), (6, 7), (11, 12), (13, 14)]

# Nombres de landmarks
LANDMARK_NAMES = [
    'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8',
    'L9', 'L10', 'L11', 'L12', 'L13', 'L14', 'L15'
]

# Colores por categoria
CATEGORY_COLORS = {
    'Normal': '#2ecc71',       # Verde
    'COVID': '#e74c3c',        # Rojo
    'Viral_Pneumonia': '#3498db'  # Azul
}


# ==============================================================================
# FUNCIONES DE CARGA DE MODELOS
# ==============================================================================

def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Carga un modelo desde checkpoint."""
    model = create_model(
        pretrained=True,
        freeze_backbone=True,
        dropout_rate=0.3,
        hidden_dim=768,
        use_coord_attention=True,
        deep_head=True,
        device=device
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_ensemble(checkpoints: list[str], device: torch.device) -> list:
    """Carga todos los modelos del ensemble."""
    models = []
    print("\n=== Cargando modelos del ensemble ===")
    for i, checkpoint_path in enumerate(checkpoints):
        full_path = resolve_repo_path(checkpoint_path)
        if not full_path.exists():
            raise FileNotFoundError(f"No se encontro checkpoint: {full_path}")
        print(f"  [{i+1}/{len(checkpoints)}] Cargando: {checkpoint_path}")
        model = load_model(full_path, device)
        models.append(model)
    print(f"  Ensemble cargado: {len(models)} modelos")
    return models


# ==============================================================================
# FUNCIONES DE PREDICCION CON TTA
# ==============================================================================

@torch.no_grad()
def predict_with_tta(model: torch.nn.Module, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Prediccion con Test-Time Augmentation (original + flip horizontal)."""
    model.eval()

    # Prediccion original
    pred1 = model(image)

    # Prediccion con flip horizontal
    image_flip = torch.flip(image, dims=[3])
    pred2 = model(image_flip)

    # Invertir flip en predicciones
    pred2 = pred2.view(-1, 15, 2)
    pred2[:, :, 0] = 1 - pred2[:, :, 0]  # Invertir X

    # Intercambiar pares simetricos
    for left, right in SYMMETRIC_PAIRS:
        pred2[:, [left, right]] = pred2[:, [right, left]]

    pred2 = pred2.view(-1, 30)

    # Promediar
    return (pred1 + pred2) / 2


@torch.no_grad()
def predict_ensemble(models: list, images: torch.Tensor) -> torch.Tensor:
    """Prediccion del ensemble sin TTA."""
    preds = [model(images) for model in models]
    return torch.stack(preds).mean(dim=0)


@torch.no_grad()
def predict_ensemble_tta(models: list, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Prediccion del ensemble con TTA."""
    preds = []
    for model in models:
        pred = predict_with_tta(model, images, device)
        preds.append(pred)

    # Promediar predicciones de todos los modelos
    ensemble_pred = torch.stack(preds).mean(dim=0)
    return ensemble_pred


# ==============================================================================
# EXTRACCION DE PREDICCIONES
# ==============================================================================

def extract_all_predictions(
    models: list,
    test_loader,
    device: torch.device,
    image_size: int = 224,
    use_tta: bool = True,
) -> dict:
    """
    Extrae predicciones del ensemble para todas las imagenes del test set.

    Returns:
        dict con:
            - predictions: (N, 15, 2) coordenadas predichas en pixeles
            - ground_truth: (N, 15, 2) coordenadas GT en pixeles
            - errors: (N, 15) error por landmark en pixeles
            - image_names: lista de nombres de imagen
            - categories: lista de categorias
            - metadata: lista de dicts con info adicional
    """
    all_preds = []
    all_targets = []
    all_names = []
    all_categories = []
    all_metadata = []

    print("\n=== Extrayendo predicciones ===")
    for batch in tqdm(test_loader, desc="Procesando imagenes"):
        images = batch[0].to(device)
        targets = batch[1].to(device)
        metadata = batch[2]

        # Prediccion del ensemble
        if use_tta:
            preds = predict_ensemble_tta(models, images, device)
        else:
            preds = predict_ensemble(models, images)

        # Convertir a shape (B, 15, 2)
        preds = preds.view(-1, 15, 2)
        targets = targets.view(-1, 15, 2)

        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

        for meta in metadata:
            all_names.append(meta['image_name'])
            all_categories.append(meta['category'])
            all_metadata.append(meta)

    # Concatenar
    predictions = torch.cat(all_preds, dim=0)  # (N, 15, 2)
    ground_truth = torch.cat(all_targets, dim=0)  # (N, 15, 2)

    # Convertir a pixeles
    predictions_px = predictions * image_size
    ground_truth_px = ground_truth * image_size

    # Calcular errores
    errors = torch.norm(predictions_px - ground_truth_px, dim=-1)  # (N, 15)

    return {
        'predictions': predictions_px.numpy(),  # (N, 15, 2) en pixeles
        'ground_truth': ground_truth_px.numpy(),  # (N, 15, 2) en pixeles
        'predictions_normalized': predictions.numpy(),  # (N, 15, 2) en [0,1]
        'ground_truth_normalized': ground_truth.numpy(),  # (N, 15, 2) en [0,1]
        'errors': errors.numpy(),  # (N, 15)
        'image_names': all_names,
        'categories': all_categories,
        'metadata': all_metadata,
    }


# ==============================================================================
# GUARDADO DE PREDICCIONES
# ==============================================================================

def save_predictions_csv(data: dict, output_path: Path):
    """Guarda predicciones en formato CSV."""
    rows = []

    for i in range(len(data['image_names'])):
        row = {
            'image_name': data['image_names'][i],
            'category': data['categories'][i],
        }

        # Coordenadas predichas (en pixeles)
        for j in range(15):
            row[f'pred_L{j+1}_x'] = data['predictions'][i, j, 0]
            row[f'pred_L{j+1}_y'] = data['predictions'][i, j, 1]

        # Coordenadas GT (en pixeles)
        for j in range(15):
            row[f'gt_L{j+1}_x'] = data['ground_truth'][i, j, 0]
            row[f'gt_L{j+1}_y'] = data['ground_truth'][i, j, 1]

        # Errores por landmark
        for j in range(15):
            row[f'error_L{j+1}'] = data['errors'][i, j]

        # Error promedio
        row['error_mean'] = data['errors'][i].mean()

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  CSV guardado: {output_path}")


def save_predictions_json(data: dict, output_path: Path, metadata: dict):
    """Guarda predicciones en formato JSON detallado."""
    output = {
        'metadata': metadata,
        'landmark_names': LANDMARK_NAMES,
        'predictions': []
    }

    for i in range(len(data['image_names'])):
        pred_entry = {
            'image': data['image_names'][i],
            'category': data['categories'][i],
            'landmarks_pred': data['predictions'][i].tolist(),
            'landmarks_gt': data['ground_truth'][i].tolist(),
            'errors': data['errors'][i].tolist(),
            'error_mean': float(data['errors'][i].mean()),
        }
        output['predictions'].append(pred_entry)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  JSON guardado: {output_path}")


def save_predictions_npz(data: dict, output_path: Path):
    """Guarda predicciones en formato NPZ (NumPy)."""
    np.savez(
        output_path,
        predictions=data['predictions'],
        ground_truth=data['ground_truth'],
        predictions_normalized=data['predictions_normalized'],
        ground_truth_normalized=data['ground_truth_normalized'],
        errors=data['errors'],
        image_names=np.array(data['image_names']),
        categories=np.array(data['categories']),
    )
    print(f"  NPZ guardado: {output_path}")


# ==============================================================================
# TRIANGULACION DE DELAUNAY
# ==============================================================================

def compute_delaunay_triangulation(landmarks: np.ndarray) -> Delaunay:
    """
    Computa triangulacion de Delaunay para un conjunto de landmarks.

    Args:
        landmarks: (15, 2) coordenadas de landmarks

    Returns:
        Objeto Delaunay con la triangulacion
    """
    return Delaunay(landmarks)


def get_canonical_triangulation(data: dict) -> dict:
    """
    Obtiene la triangulacion canonica usando el promedio de todos los landmarks.
    Esta triangulacion se usara para todas las imagenes.
    """
    # Promedio de GT de todas las imagenes
    mean_landmarks = data['ground_truth'].mean(axis=0)  # (15, 2)

    # Triangulacion
    tri = compute_delaunay_triangulation(mean_landmarks)

    return {
        'simplices': tri.simplices.tolist(),  # Lista de triangulos [(i, j, k), ...]
        'num_triangles': len(tri.simplices),
        'mean_landmarks': mean_landmarks.tolist(),
    }


def compute_triangle_areas(landmarks: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """
    Calcula el area de cada triangulo.

    Args:
        landmarks: (15, 2) coordenadas
        triangles: (M, 3) indices de vertices

    Returns:
        (M,) areas de triangulos
    """
    areas = []
    for tri in triangles:
        p1, p2, p3 = landmarks[tri]
        # Area usando producto cruz
        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        areas.append(area)
    return np.array(areas)


def compute_delaunay_metrics(data: dict, triangles: np.ndarray) -> pd.DataFrame:
    """
    Calcula metricas de triangulacion para cada imagen.

    Returns:
        DataFrame con areas de triangulos por imagen
    """
    rows = []

    for i in range(len(data['image_names'])):
        row = {
            'image_name': data['image_names'][i],
            'category': data['categories'][i],
        }

        # Areas de triangulos predichos
        pred_landmarks = data['predictions'][i]  # (15, 2)
        pred_areas = compute_triangle_areas(pred_landmarks, triangles)

        # Areas de triangulos GT
        gt_landmarks = data['ground_truth'][i]
        gt_areas = compute_triangle_areas(gt_landmarks, triangles)

        # Guardar areas
        for j, area in enumerate(pred_areas):
            row[f'pred_triangle_{j}_area'] = area

        for j, area in enumerate(gt_areas):
            row[f'gt_triangle_{j}_area'] = area

        # Totales
        row['pred_total_area'] = pred_areas.sum()
        row['gt_total_area'] = gt_areas.sum()
        row['area_diff'] = pred_areas.sum() - gt_areas.sum()
        row['area_diff_percent'] = (row['area_diff'] / gt_areas.sum()) * 100

        rows.append(row)

    return pd.DataFrame(rows)


def save_delaunay_triangles(triangulation: dict, output_path: Path):
    """Guarda definicion de triangulos en JSON."""
    # Crear nombres descriptivos para triangulos
    triangle_names = []
    for tri in triangulation['simplices']:
        name = f"L{tri[0]+1}-L{tri[1]+1}-L{tri[2]+1}"
        triangle_names.append(name)

    output = {
        'num_triangles': triangulation['num_triangles'],
        'triangles': triangulation['simplices'],
        'triangle_names': triangle_names,
        'mean_landmarks': triangulation['mean_landmarks'],
        'description': 'Triangulacion de Delaunay sobre 15 landmarks anatomicos',
        'landmark_names': LANDMARK_NAMES,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Triangulos Delaunay guardados: {output_path}")


# ==============================================================================
# VISUALIZACIONES
# ==============================================================================

def visualize_delaunay(
    image_path: Path,
    landmarks: np.ndarray,
    triangles: np.ndarray,
    output_path: Path,
    title: str = None,
    show_indices: bool = True
):
    """
    Visualiza landmarks y triangulacion de Delaunay sobre una imagen.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Cargar y mostrar imagen en escala de grises
    img = Image.open(image_path)
    img_array = np.array(img)
    ax.imshow(img_array, cmap='gray', vmin=0, vmax=255)

    # Dibujar triangulos
    for tri in triangles:
        pts = landmarks[tri]
        triangle = plt.Polygon(pts, fill=False, edgecolor='yellow', linewidth=1.5, alpha=0.8)
        ax.add_patch(triangle)

    # Dibujar landmarks
    ax.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=100, zorder=5, edgecolors='white', linewidths=2)

    # Etiquetas
    if show_indices:
        for i, (x, y) in enumerate(landmarks):
            ax.annotate(f'L{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, color='white', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.7))

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def visualize_delaunay_comparison(
    image_path: Path,
    gt_landmarks: np.ndarray,
    pred_landmarks: np.ndarray,
    triangles: np.ndarray,
    output_path: Path,
    title: str = None
):
    """
    Visualiza comparacion de triangulacion GT vs Prediccion.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Cargar y mostrar imagen en escala de grises
    img = Image.open(image_path)
    img_array = np.array(img)
    ax.imshow(img_array, cmap='gray', vmin=0, vmax=255)

    # Triangulos GT (verde)
    for tri in triangles:
        pts = gt_landmarks[tri]
        triangle = plt.Polygon(pts, fill=False, edgecolor='green', linewidth=2, alpha=0.8)
        ax.add_patch(triangle)

    # Triangulos Pred (rojo)
    for tri in triangles:
        pts = pred_landmarks[tri]
        triangle = plt.Polygon(pts, fill=False, edgecolor='red', linewidth=2, alpha=0.8, linestyle='--')
        ax.add_patch(triangle)

    # Landmarks GT
    ax.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], c='green', s=80, zorder=5,
               edgecolors='white', linewidths=1.5, label='Ground Truth')

    # Landmarks Pred
    ax.scatter(pred_landmarks[:, 0], pred_landmarks[:, 1], c='red', s=80, zorder=5,
               edgecolors='white', linewidths=1.5, marker='^', label='Prediccion')

    ax.legend(loc='upper right', fontsize=10)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_category_visualizations(
    data: dict,
    triangles: np.ndarray,
    data_root: Path,
    output_dir: Path
):
    """
    Genera visualizaciones de ejemplo por categoria.
    """
    print("\n=== Generando visualizaciones por categoria ===")

    categories = ['Normal', 'COVID', 'Viral_Pneumonia']

    for category in categories:
        # Encontrar un ejemplo de esta categoria
        for i, cat in enumerate(data['categories']):
            if cat == category:
                image_name = data['image_names'][i]
                landmarks = data['predictions'][i]
                gt_landmarks = data['ground_truth'][i]

                # Obtener path de imagen
                # Ajustar el tamaño de landmarks de 224 a 299 (tamaño original)
                scale = 299 / 224
                landmarks_scaled = landmarks * scale
                gt_scaled = gt_landmarks * scale

                image_path = get_image_path(image_name, category, data_root)

                # Generar visualizacion
                output_path = output_dir / f'delaunay_{category.lower()}.png'
                visualize_delaunay(
                    image_path, landmarks_scaled, triangles, output_path,
                    title=f'Triangulacion Delaunay - {category}'
                )
                print(f"  {category}: {output_path}")
                break


def generate_comparison_visualization(
    data: dict,
    triangles: np.ndarray,
    data_root: Path,
    output_dir: Path
):
    """
    Genera visualizacion de comparacion GT vs Prediccion.
    """
    print("\n=== Generando comparacion GT vs Prediccion ===")

    # Encontrar un caso con error cercano al promedio
    mean_errors = data['errors'].mean(axis=1)
    median_idx = np.argsort(mean_errors)[len(mean_errors)//2]

    image_name = data['image_names'][median_idx]
    category = data['categories'][median_idx]
    landmarks = data['predictions'][median_idx]
    gt_landmarks = data['ground_truth'][median_idx]
    error = mean_errors[median_idx]

    # Escalar para imagen original
    scale = 299 / 224
    landmarks_scaled = landmarks * scale
    gt_scaled = gt_landmarks * scale

    image_path = get_image_path(image_name, category, data_root)
    output_path = output_dir / 'delaunay_comparison.png'

    visualize_delaunay_comparison(
        image_path, gt_scaled, landmarks_scaled, triangles, output_path,
        title=f'Comparacion GT (verde) vs Prediccion (rojo) - Error: {error:.2f} px'
    )
    print(f"  Comparacion guardada: {output_path}")


def generate_area_heatmap(
    metrics_df: pd.DataFrame,
    triangles: np.ndarray,
    output_path: Path
):
    """
    Genera heatmap de areas de triangulos por categoria.
    """
    print("\n=== Generando heatmap de areas ===")

    num_triangles = len(triangles)
    categories = ['Normal', 'COVID', 'Viral_Pneumonia']

    # Calcular areas promedio por categoria
    area_data = np.zeros((len(categories), num_triangles))

    for i, cat in enumerate(categories):
        cat_df = metrics_df[metrics_df['category'] == cat]
        for j in range(num_triangles):
            col = f'pred_triangle_{j}_area'
            if col in cat_df.columns:
                area_data[i, j] = cat_df[col].mean()

    # Crear heatmap
    fig, ax = plt.subplots(figsize=(14, 6))

    im = ax.imshow(area_data, cmap='YlOrRd', aspect='auto')

    # Etiquetas
    ax.set_xticks(range(num_triangles))
    ax.set_xticklabels([f'T{i}' for i in range(num_triangles)], fontsize=8)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)

    ax.set_xlabel('Triangulo', fontsize=12)
    ax.set_ylabel('Categoria', fontsize=12)
    ax.set_title('Area Promedio de Triangulos por Categoria', fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Area (px²)', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Heatmap guardado: {output_path}")


# ==============================================================================
# FUNCION PRINCIPAL
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Extraer predicciones del ensemble y generar triangulacion de Delaunay')
    parser.add_argument('--config', type=str, default=None,
                        help='JSON con lista de modelos y defaults (opcional)')
    parser.add_argument('--name', type=str, default=None,
                        help='Nombre opcional de la corrida (solo metadata)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Lista de checkpoints del ensemble (override config)')
    parser.add_argument('--data-root', type=str, default='data/',
                        help='Directorio raiz de datos')
    parser.add_argument('--csv-path', type=str,
                        default='data/coordenadas/coordenadas_maestro.csv',
                        help='CSV de coordenadas')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Tamano de batch')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Workers para DataLoader')
    parser.add_argument('--split-seed', type=int, default=42,
                        help='Semilla para split train/val/test')
    parser.add_argument('--clahe', dest='clahe', action='store_true',
                        help='Usar CLAHE')
    parser.add_argument('--no-clahe', dest='clahe', action='store_false',
                        help='Desactivar CLAHE')
    parser.add_argument('--clahe-clip', type=float, default=2.0,
                        help='CLAHE clip limit')
    parser.add_argument('--clahe-tile', type=int, default=4,
                        help='CLAHE tile size')
    parser.add_argument('--tta', dest='tta', action='store_true',
                        help='Usar Test-Time Augmentation')
    parser.add_argument('--no-tta', dest='tta', action='store_false',
                        help='Desactivar TTA')
    parser.add_argument('--output-dir', type=str, default='outputs/predictions',
                        help='Directorio de salida')
    parser.add_argument('--visualize-only', action='store_true',
                        help='Solo generar visualizaciones (requiere predicciones existentes)')
    parser.add_argument('--no-viz', action='store_true',
                        help='No generar visualizaciones')
    parser.set_defaults(clahe=None, tta=None)

    args, _ = parser.parse_known_args()
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_file():
            parser.error(f"Config file not found: {config_path}")
        with config_path.open('r') as handle:
            config_data = json.load(handle)
        if not isinstance(config_data, dict):
            parser.error("Config file must be a JSON object with flat key/value pairs.")
        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(config_data) - valid_keys)
        if unknown_keys:
            parser.error(f"Unknown config keys: {', '.join(unknown_keys)}")
        parser.set_defaults(**config_data)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # Crear directorio de salida
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / 'viz'
    viz_dir.mkdir(exist_ok=True)

    # Si solo visualizaciones, cargar datos existentes
    if args.visualize_only:
        npz_path = output_dir / 'test_predictions.npz'
        if not npz_path.exists():
            print(f"ERROR: No se encontro {npz_path}")
            print("Ejecuta primero sin --visualize-only para generar predicciones")
            sys.exit(1)

        print(f"\n=== Cargando predicciones existentes de {npz_path} ===")
        npz_data = np.load(npz_path, allow_pickle=True)
        data = {
            'predictions': npz_data['predictions'],
            'ground_truth': npz_data['ground_truth'],
            'errors': npz_data['errors'],
            'image_names': npz_data['image_names'].tolist(),
            'categories': npz_data['categories'].tolist(),
        }
    else:
        use_tta = args.tta if args.tta is not None else True
        use_clahe = args.clahe if args.clahe is not None else True
        models_list = args.models or DEFAULT_ENSEMBLE_CHECKPOINTS

        # Cargar modelos
        models = load_ensemble(models_list, device)

        # Cargar datos de test
        print("\n=== Cargando datos de test ===")
        _, _, test_loader = create_dataloaders(
            csv_path=str(resolve_repo_path(args.csv_path)),
            data_root=str(resolve_repo_path(args.data_root)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            random_state=args.split_seed,
            use_clahe=use_clahe,
            clahe_clip_limit=args.clahe_clip,
            clahe_tile_size=args.clahe_tile,
        )

        # Extraer predicciones
        data = extract_all_predictions(
            models,
            test_loader,
            device,
            use_tta=use_tta,
        )

        # Mostrar estadisticas
        print(f"\n=== Estadisticas de Predicciones ===")
        print(f"  Total de imagenes: {len(data['image_names'])}")
        print(f"  Error promedio: {data['errors'].mean():.2f} px")
        print(f"  Error std: {data['errors'].std():.2f} px")
        print(f"  Error mediana: {np.median(data['errors']):.2f} px")

        # Guardar predicciones
        print("\n=== Guardando predicciones ===")
        metadata = {
            'model': 'ensemble',
            'models': models_list,
            'tta': use_tta,
            'clahe': use_clahe,
            'clahe_clip': args.clahe_clip,
            'clahe_tile': args.clahe_tile,
            'split_seed': args.split_seed,
            'error_mean': float(data['errors'].mean()),
            'error_std': float(data['errors'].std()),
            'error_median': float(np.median(data['errors'])),
            'num_samples': len(data['image_names']),
            'image_size': 224,
            'timestamp': datetime.now().isoformat(),
        }
        if args.name:
            metadata['name'] = args.name
        if args.config:
            metadata['config'] = args.config

        save_predictions_csv(data, output_dir / 'test_predictions.csv')
        save_predictions_json(data, output_dir / 'test_predictions.json', metadata)
        save_predictions_npz(data, output_dir / 'test_predictions.npz')

    # Triangulacion de Delaunay
    print("\n=== Calculando triangulacion de Delaunay ===")
    triangulation = get_canonical_triangulation(data)
    print(f"  Numero de triangulos: {triangulation['num_triangles']}")

    triangles = np.array(triangulation['simplices'])
    save_delaunay_triangles(triangulation, output_dir / 'delaunay_triangles.json')

    # Metricas de triangulacion
    print("\n=== Calculando metricas de triangulacion ===")
    metrics_df = compute_delaunay_metrics(data, triangles)
    metrics_df.to_csv(output_dir / 'delaunay_metrics.csv', index=False)
    print(f"  Metricas guardadas: {output_dir / 'delaunay_metrics.csv'}")

    # Visualizaciones
    if not args.no_viz:
        data_root = PROJECT_ROOT / 'data'

        generate_category_visualizations(data, triangles, data_root, viz_dir)
        generate_comparison_visualization(data, triangles, data_root, viz_dir)
        generate_area_heatmap(metrics_df, triangles, viz_dir / 'delaunay_area_heatmap.png')

    # Resumen final
    print("\n" + "=" * 60)
    print("EXTRACCION COMPLETADA")
    print("=" * 60)
    print(f"\nArchivos generados en {output_dir}:")
    print(f"  - test_predictions.csv       (CSV estructurado)")
    print(f"  - test_predictions.json      (JSON detallado)")
    print(f"  - test_predictions.npz       (NumPy arrays)")
    print(f"  - delaunay_triangles.json    (Definicion de triangulos)")
    print(f"  - delaunay_metrics.csv       (Metricas por imagen)")
    if not args.no_viz:
        print(f"\nVisualizaciones en {viz_dir}:")
        print(f"  - delaunay_normal.png")
        print(f"  - delaunay_covid.png")
        print(f"  - delaunay_viral_pneumonia.png")
        print(f"  - delaunay_comparison.png")
        print(f"  - delaunay_area_heatmap.png")

    print(f"\n*** Error promedio del ensemble: {data['errors'].mean():.2f} px ***")


if __name__ == '__main__':
    main()
