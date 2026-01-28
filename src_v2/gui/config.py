"""
Configuration file for the GUI demonstration.

Contains paths, validated metrics, colors, and interface text.
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Check if running from installed demo package or PyInstaller frozen mode
MODELS_DIR = os.environ.get('COVID_DEMO_MODELS_DIR')
IS_FROZEN = os.environ.get('COVID_DEMO_FROZEN', '0') == '1'
PROJECT_ROOT = Path(__file__).parent.parent.parent

if MODELS_DIR:
    # Deployment mode: models in simplified structure
    MODELS_BASE = Path(MODELS_DIR)
elif IS_FROZEN and hasattr(sys, '_MEIPASS'):
    # PyInstaller frozen mode fallback
    import sys
    MODELS_BASE = Path(sys._MEIPASS) / 'models'
    MODELS_DIR = str(MODELS_BASE)
elif (PROJECT_ROOT / 'python' / 'python.exe').exists():
    # Portable mode: Python embeddable detected
    MODELS_BASE = PROJECT_ROOT / 'models'
    MODELS_DIR = str(MODELS_BASE)
    IS_PORTABLE = True
    print(f"[INFO] Portable mode detected. Models directory: {MODELS_DIR}")
else:
    # Development mode: original structure
    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    CONFIGS_DIR = PROJECT_ROOT / "configs"
    IS_PORTABLE = False

DATA_DIR = PROJECT_ROOT / "data"

# ============================================================================
# MODEL PATHS (from configs/ensemble_best.json and GROUND_TRUTH.json)
# ============================================================================

if MODELS_DIR:
    # Deployment mode: simplified paths (PyInstaller or release package)
    CANONICAL_SHAPE = MODELS_BASE / "shape_analysis/canonical_shape_gpa.json"
    DELAUNAY_TRIANGLES = MODELS_BASE / "shape_analysis/canonical_delaunay_triangles.json"

    # PyInstaller frozen mode: models are renamed during build to standardized names
    # (see scripts/prepare_models_for_build.py)
    LANDMARK_MODELS = [
        MODELS_BASE / "landmarks/resnet18_seed123_best.pt",
        MODELS_BASE / "landmarks/resnet18_seed321_best.pt",
        MODELS_BASE / "landmarks/resnet18_seed111_best.pt",
        MODELS_BASE / "landmarks/resnet18_seed666_best.pt",
    ]

    CLASSIFIER_CHECKPOINT = MODELS_BASE / "classifier/best_classifier.pt"
else:
    # Development mode: original paths
    ENSEMBLE_CONFIG = CONFIGS_DIR / "ensemble_best.json"
    CANONICAL_SHAPE = OUTPUTS_DIR / "shape_analysis/canonical_shape_gpa.json"
    DELAUNAY_TRIANGLES = OUTPUTS_DIR / "shape_analysis/canonical_delaunay_triangles.json"

    LANDMARK_MODELS = [
        CHECKPOINTS_DIR / "session10/ensemble/seed123/final_model.pt",
        CHECKPOINTS_DIR / "session13/seed321/final_model.pt",
        CHECKPOINTS_DIR / "repro_split111/session14/seed111/final_model.pt",
        CHECKPOINTS_DIR / "repro_split666/session16/seed666/final_model.pt",
    ]

    CLASSIFIER_PATH = OUTPUTS_DIR / "classifier_warped_lung_best/sweeps_2026-01-12/lr2e-4_seed321_on"
    CLASSIFIER_CHECKPOINT = CLASSIFIER_PATH / "best_classifier.pt"

# ============================================================================
# EXAMPLE IMAGES
# ============================================================================

EXAMPLES_DIR = PROJECT_ROOT / "examples"

# Will be populated after examples are created
EXAMPLES: List[Tuple[str, str]] = []

# ============================================================================
# VALIDATED METRICS (from GROUND_TRUTH.json v2.1.0)
# ============================================================================

VALIDATED_METRICS = {
    # Landmark detection (ensemble_4_models_tta_best_20260111)
    'landmark_error_px': 3.61,
    'landmark_std_px': 2.48,
    'landmark_median_px': 3.07,

    # Classification - Cross-Validation (5-fold, más robusto)
    'classification_cv_accuracy_mean': 98.60,
    'classification_cv_accuracy_std': 0.26,
    'classification_cv_f1_macro_mean': 98.00,
    'classification_cv_f1_macro_std': 0.36,
    'classification_cv_f1_weighted_mean': 98.60,
    'classification_cv_f1_weighted_std': 0.25,

    # Para retrocompatibilidad (mantener por ahora):
    'classification_accuracy': 98.60,  # Ahora apunta a CV mean
    'classification_f1_macro': 98.00,
    'classification_f1_weighted': 98.60,

    # Preprocessing parameters
    'clahe_clip': 2.0,
    'clahe_tile': 4,
    'margin_scale': 1.05,

    # Dataset info
    'model_input_size': 224,
    'fill_rate': 47,
}

# Per-landmark errors (from GROUND_TRUTH.json)
PER_LANDMARK_ERRORS = {
    'L1': 3.22, 'L2': 3.96, 'L3': 3.18, 'L4': 3.65,
    'L5': 2.88, 'L6': 2.94, 'L7': 3.29, 'L8': 3.50,
    'L9': 2.76, 'L10': 2.44, 'L11': 2.94, 'L12': 5.43,
    'L13': 5.35, 'L14': 4.39, 'L15': 4.29,
}

# ============================================================================
# LANDMARK VISUALIZATION (from scripts/visualize_predictions.py)
# ============================================================================

LANDMARK_COLORS = {
    'axis': '#00FF00',      # L1, L2 - verde
    'central': '#00FFFF',   # L9, L10, L11 - cyan
    'lateral': '#FFFF00',   # L3-L8 - amarillo
    'border': '#FF00FF',    # L12, L13 - magenta
    'costal': '#FF0000',    # L14, L15 - rojo
}

LANDMARK_GROUPS = {
    0: 'axis', 1: 'axis',                                      # L1, L2
    2: 'lateral', 3: 'lateral', 4: 'lateral', 5: 'lateral',   # L3-L6
    6: 'lateral', 7: 'lateral',                                # L7, L8
    8: 'central', 9: 'central', 10: 'central',                 # L9, L10, L11
    11: 'border', 12: 'border',                                # L12, L13
    13: 'costal', 14: 'costal',                                # L14, L15
}

LANDMARK_LABELS_ES = {
    'axis': 'Eje Vertical (Vértice y Base Pulmonar)',
    'central': 'Región Mediastínica Central',
    'lateral': 'Contorno Lateral Pulmonar',
    'border': 'Ápices Pulmonares Superiores',
    'costal': 'Ángulos Costofrénicos',
}

# ============================================================================
# CLASSIFICATION
# ============================================================================

CLASS_NAMES = ['COVID', 'Normal', 'Viral_Pneumonia']
CLASS_NAMES_ES = ['COVID-19', 'Normal', 'Neumonía Viral']

CLASS_COLORS = {
    'COVID': '#FF6B6B',         # Rojo
    'Normal': '#51CF66',        # Verde
    'Viral_Pneumonia': '#FFD43B',  # Amarillo
}


def get_class_color_es(class_name_es: str) -> str:
    """
    Obtiene el color hexadecimal para una clase (nombre en español).

    Args:
        class_name_es: Nombre de clase en español ("COVID-19", "Normal", "Neumonía Viral")

    Returns:
        Color hex string (e.g., "#FF6B6B")
    """
    # Mapeo de nombres en español a claves en CLASS_COLORS
    mapping = {
        'COVID-19': 'COVID',
        'Normal': 'Normal',
        'Neumonía Viral': 'Viral_Pneumonia',
    }

    class_key = mapping.get(class_name_es, 'COVID')  # Default a COVID si no se encuentra
    return CLASS_COLORS.get(class_key, '#FF6B6B')  # Default rojo


# ============================================================================
# INTERFACE TEXT (Spanish)
# ============================================================================

TITLE = "Sistema de Detección de COVID-19 mediante Puntos de Referencia Anatómicos"

SUBTITLE = f"""
**Sistema Validado**: Ensemble de Detección de Landmarks (15 puntos) |
Accuracy Clasificación: {VALIDATED_METRICS['classification_cv_accuracy_mean']:.2f}% ± {VALIDATED_METRICS['classification_cv_accuracy_std']:.2f}% (5-fold CV)
"""

ABOUT_TEXT = f"""
## Metodología

Este sistema combina tres componentes principales para la detección de COVID-19 en radiografías de tórax:

### 1. Detección de Puntos de Referencia Anatómicos
Ensemble de 4 modelos ResNet-18 con Coordinate Attention que predicen 15 puntos de referencia
en el contorno pulmonar:
- **Arquitectura**: ResNet-18 con módulo de Coordinate Attention
- **Test-Time Augmentation**: Flip horizontal con corrección de pares simétricos
- **Preprocesamiento**: CLAHE (clip={VALIDATED_METRICS['clahe_clip']}, tile={VALIDATED_METRICS['clahe_tile']}×{VALIDATED_METRICS['clahe_tile']})
- **Landmarks**: 15 puntos anatómicos en 5 grupos funcionales

Los 15 puntos de referencia definen el contorno pulmonar en 5 grupos:
- **Eje (verde)**: L1, L2 - Puntos superior e inferior del eje central
- **Central (cyan)**: L9, L10, L11 - Puntos intermedios del eje central
- **Lateral (amarillo)**: L3-L8 - Contornos laterales izquierdo y derecho
- **Borde (magenta)**: L12, L13 - Puntos de borde superior
- **Costal (rojo)**: L14, L15 - Puntos costales inferiores

### 2. Normalización Geométrica
Warping piecewise affine que alinea todas las radiografías a una forma canónica consenso:
- **Análisis Procrustes Generalizado (GPA)**: Calcula la forma consenso de los pulmones
- **Triangulación de Delaunay**: 18 triángulos para transformación por regiones
- **Margen óptimo**: {VALIDATED_METRICS['margin_scale']} (expansión desde centroide)
- **Objetivo**: Eliminar variabilidad en posicionamiento del paciente

### 3. Clasificación COVID-19
Clasificador ResNet-18 entrenado sobre imágenes normalizadas:
- **Accuracy**: {VALIDATED_METRICS['classification_accuracy']}%
- **F1-Score Macro**: {VALIDATED_METRICS['classification_f1_macro']}%
- **F1-Score Weighted**: {VALIDATED_METRICS['classification_f1_weighted']}%
- **Clases**: COVID-19, Normal, Neumonía Viral
- **Explicabilidad**: GradCAM muestra regiones de atención del modelo

## Arquitectura del Modelo

### Detector de Puntos de Referencia
```
ResNet-18 (pretrained ImageNet)
    ↓
Coordinate Attention Module
    ↓
Deep Head (FC 512→768→30 coords)
    ↓
15 puntos de referencia (x, y) normalizados [0,1]
```

### Clasificador
```
Imagen Warped (224×224 grayscale)
    ↓
ResNet-18 (finetuned)
    ↓
Global Average Pooling
    ↓
FC Layer → 3 clases
    ↓
Softmax → Probabilidades
```

## Dataset

- **Fuente**: COVID-19 Radiography Database (Kaggle)
- **Clases**: COVID-19, Normal, Neumonía Viral
- **Splits**: Train/Val/Test con seed fijo para reproducibilidad
- **Anotaciones**: 15 puntos de referencia manuales en 100 imágenes para entrenamiento

## Resultados Detallados

### Distribución de Puntos de Referencia
Los 15 puntos de referencia se distribuyen estratégicamente para capturar la geometría pulmonar:
- **Puntos centrales** (L9, L10, L11): Eje mediastínico y mediastino
- **Contornos laterales** (L3-L8): Bordes izquierdo y derecho de ambos pulmones
- **Puntos axiales** (L1, L2): Vértice superior y base inferior
- **Ápices y ángulos** (L12-L15): Regiones de mayor variabilidad anatómica

### Cross-Validation (5-fold)
- **Val Accuracy**: {VALIDATED_METRICS.get('cv_val_accuracy_mean', 'N/A')}%
- **Val F1-Macro**: {VALIDATED_METRICS.get('cv_val_f1_macro_mean', 'N/A')}%

## Tecnologías

- **Deep Learning**: PyTorch 2.x
- **Arquitectura**: ResNet-18 con modificaciones (Coordinate Attention)
- **Preprocesamiento**: OpenCV CLAHE, numpy
- **Geometría**: Delaunay triangulation, piecewise affine warping
- **Visualización**: Matplotlib, Gradio
- **Explicabilidad**: GradCAM (Gradient-weighted Class Activation Mapping)

## Referencias

### Publicaciones Base
- He et al. (2016): "Deep Residual Learning for Image Recognition"
- Hou et al. (2021): "Coordinate Attention for Efficient Mobile Network Design"
- Selvaraju et al. (2017): "Grad-CAM: Visual Explanations from Deep Networks"

### Dataset
- Chowdhury et al. (2020): "Can AI help in screening Viral and COVID-19 pneumonia?"
- COVID-19 Radiography Database: https://www.kaggle.com/tawsifurrahman/covid19-radiography-database

### Metodología Geométrica
- Goodall (1991): "Procrustes methods in the statistical analysis of shape"
- Bookstein (1989): "Principal Warps: Thin-Plate Splines and Decomposition of Deformations"

## Limitaciones

1. **Domain Shift**: El modelo está entrenado en un dataset específico. Para uso clínico en nuevos
   hospitales se requiere domain adaptation o fine-tuning local.

2. **Landmarks Anatómicos**: Los 15 puntos definen contornos generales, no estructuras anatómicas
   específicas (e.g., hilios, costillas).

3. **Fill Rate**: ~{VALIDATED_METRICS['fill_rate']}% de la imagen warped contiene información
   (resto es fondo negro por la transformación geométrica).

## Autor

[Agregar información del investigador/tesista]

## Contacto

[Agregar email o información de contacto]

---

**Versión**: 1.0.13
**Última actualización**: Enero 2026
**Framework**: Gradio {4}
**Python**: 3.8+
"""

# ============================================================================
# UI TEXT SNIPPETS
# ============================================================================

PROCESSING_MESSAGE = "Procesando imagen... Esto puede tomar 1-2 segundos."
ERROR_INVALID_FORMAT = "⚠️ Formato no soportado. Use imágenes PNG o JPG."
ERROR_TOO_SMALL = "⚠️ Imagen demasiado pequeña. Tamaño mínimo: 100×100 píxeles."
ERROR_MODEL_NOT_FOUND = "❌ Error: Modelos no encontrados. Verifique las rutas en config.py."
ERROR_GPU_OOM = "⚠️ GPU sin memoria suficiente. Ejecutando en CPU (puede ser más lento)."
ERROR_WARPING_FAILED = "⚠️ Error durante warping. Mostrando imagen original."

SUCCESS_EXPORT = "✅ Resultados exportados correctamente."

# ============================================================================
# GRADIO THEME
# ============================================================================

THEME = "soft"  # Options: soft, default, monochrome, glass

# ============================================================================
# INFERENCE SETTINGS
# ============================================================================

DEVICE_PREFERENCE = "cuda"  # Will fallback to "cpu" if CUDA not available
BATCH_SIZE = 1
NUM_WORKERS = 0  # For DataLoader (0 for single-image inference)

# TTA settings
TTA_ENABLED = True  # Horizontal flip with symmetric pairs correction

# CLAHE settings (from GROUND_TRUTH.json)
CLAHE_CLIP_LIMIT = VALIDATED_METRICS['clahe_clip']
CLAHE_TILE_SIZE = (VALIDATED_METRICS['clahe_tile'], VALIDATED_METRICS['clahe_tile'])

# Warping settings
MARGIN_SCALE = VALIDATED_METRICS['margin_scale']
USE_FULL_COVERAGE = False  # Current best configuration

# SAHS (Statistical Asymmetrical Histogram Stretching) settings
SAHS_THRESHOLD = 10  # Threshold to separate lung region from background

# GradCAM settings (not used in GUI v1.0.6, maintained for reference)
GRADCAM_TARGET_LAYER = 'layer4'  # Last conv block in ResNet18
GRADCAM_ALPHA = 0.4  # Overlay transparency

# ============================================================================
# EXPORT SETTINGS
# ============================================================================

EXPORT_DPI = 150
EXPORT_FORMAT = 'pdf'  # Options: pdf, png
EXPORT_FILENAME_TEMPLATE = "resultado_{timestamp}.{ext}"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_landmark_color(landmark_idx: int) -> str:
    """Get color for a specific landmark index (0-14)."""
    group = LANDMARK_GROUPS.get(landmark_idx, 'lateral')
    return LANDMARK_COLORS[group]


def get_landmark_label_es(landmark_idx: int) -> str:
    """Get Spanish label for a landmark group."""
    group = LANDMARK_GROUPS.get(landmark_idx, 'lateral')
    return LANDMARK_LABELS_ES[group]


def get_class_name_es(class_name: str) -> str:
    """Convert English class name to Spanish."""
    mapping = {
        'COVID': 'COVID-19',
        'Normal': 'Normal',
        'Viral_Pneumonia': 'Neumonía Viral',
    }
    return mapping.get(class_name, class_name)


def populate_examples():
    """Populate examples list after verifying files exist."""
    global EXAMPLES

    example_files = [
        (EXAMPLES_DIR / "covid_example.png", "COVID-19"),
        (EXAMPLES_DIR / "normal_example.png", "Normal"),
        (EXAMPLES_DIR / "viral_example.png", "Neumonía Viral"),
    ]

    EXAMPLES = [
        (str(path), label)
        for path, label in example_files
        if path.exists()
    ]

    return EXAMPLES
