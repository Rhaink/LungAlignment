"""
Constantes centralizadas para el proyecto COVID-19 Detection via Anatomical Landmarks.

Este módulo define todas las constantes del dominio en un solo lugar
para evitar duplicación y facilitar el mantenimiento.

ESTRUCTURA GEOMÉTRICA DE LOS 15 LANDMARKS:
==========================================
Los landmarks definen el CONTORNO de los pulmones, NO son puntos anatómicos específicos.
Fueron colocados manualmente para dibujar la silueta pulmonar en 15 puntos.

Estructura:
- Eje central vertical: L1 (superior) → L9 → L10 → L11 → L2 (inferior)
  L9, L10, L11 dividen el eje en 4 partes iguales (t=0.25, 0.50, 0.75)

- Contorno pulmón izquierdo: L12 → L3 → L5 → L7 → L14
- Contorno pulmón derecho:   L13 → L4 → L6 → L8 → L15

- 5 pares simétricos (izq-der): L3-L4, L5-L6, L7-L8, L12-L13, L14-L15

Parámetro t: posición relativa sobre el eje L1-L2
- t=0.00: L1, L12, L13 (zona superior)
- t=0.25: L9, L3, L4
- t=0.50: L10, L5, L6
- t=0.75: L11, L7, L8
- t=1.00: L2, L14, L15 (zona inferior)

Uso:
    from src_v2.constants import SYMMETRIC_PAIRS, CENTRAL_LANDMARKS, DEFAULT_IMAGE_SIZE
"""

from typing import Tuple, List

# =============================================================================
# LANDMARKS - Configuración de puntos anatómicos
# =============================================================================

# Número de landmarks anatómicos
NUM_LANDMARKS: int = 15

# Número de coordenadas (x, y por cada landmark)
NUM_COORDINATES: int = NUM_LANDMARKS * 2  # 30

# Nombres de landmarks (L1 a L15)
# Los 15 landmarks definen el CONTORNO de los pulmones, NO son puntos anatómicos específicos.
# Fueron colocados manualmente para dibujar la silueta pulmonar.
LANDMARK_NAMES: List[str] = [
    'L1',   # Eje central - punto superior (t=0.00)
    'L2',   # Eje central - punto inferior (t=1.00)
    'L3',   # Contorno izquierdo - zona superior (t≈0.25)
    'L4',   # Contorno derecho - zona superior (t≈0.25)
    'L5',   # Contorno izquierdo - zona media (t≈0.50)
    'L6',   # Contorno derecho - zona media (t≈0.50)
    'L7',   # Contorno izquierdo - zona inferior (t≈0.75)
    'L8',   # Contorno derecho - zona inferior (t≈0.75)
    'L9',   # Eje central - cuarto superior (t=0.25)
    'L10',  # Eje central - punto medio (t=0.50)
    'L11',  # Eje central - cuarto inferior (t=0.75)
    'L12',  # Borde superior izquierdo (t≈0.00)
    'L13',  # Borde superior derecho (t≈0.00)
    'L14',  # Esquina inferior izquierda (t≈1.00)
    'L15',  # Esquina inferior derecha (t≈1.00)
]

# Pares de landmarks simétricos (índices 0-based)
# Cada par tiene un punto en el contorno izquierdo y otro en el derecho
SYMMETRIC_PAIRS: List[Tuple[int, int]] = [
    (2, 3),   # L3-L4: Contorno zona superior
    (4, 5),   # L5-L6: Contorno zona media
    (6, 7),   # L7-L8: Contorno zona inferior
    (11, 12), # L12-L13: Borde superior
    (13, 14), # L14-L15: Esquinas inferiores
]

# Landmarks centrales que están sobre el eje L1-L2
# Dividen el eje en 4 partes iguales (t=0.25, 0.50, 0.75)
CENTRAL_LANDMARKS: List[int] = [8, 9, 10]  # L9, L10, L11

# Landmarks centrales con sus posiciones t teóricas sobre el eje L1-L2
# Útil para modelos jerárquicos que predicen posición relativa
CENTRAL_LANDMARKS_T: List[Tuple[int, float]] = [
    (8, 0.25),   # L9
    (9, 0.50),   # L10
    (10, 0.75),  # L11
]

# Landmarks del eje vertical (definen la línea central)
AXIS_LANDMARKS: List[int] = [0, 1]  # L1 (superior), L2 (inferior)

# =============================================================================
# DIMENSIONES DE IMAGEN
# =============================================================================

# Tamaño de imagen por defecto para el modelo
DEFAULT_IMAGE_SIZE: int = 224

# Tamaño original de las imágenes del dataset
ORIGINAL_IMAGE_SIZE: int = 299

# =============================================================================
# NORMALIZACIÓN
# =============================================================================

# Valores de normalización ImageNet (para modelos preentrenados)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# =============================================================================
# CATEGORÍAS DEL DATASET
# =============================================================================

# Categorías de clasificación
CATEGORIES: List[str] = ['COVID', 'Normal', 'Viral_Pneumonia']

# Número de clases
NUM_CLASSES: int = len(CATEGORIES)

# Pesos por defecto por categoría (para balanceo de clases)
DEFAULT_CATEGORY_WEIGHTS = {
    'COVID': 2.0,           # Más peso por ser clase minoritaria importante
    'Normal': 1.0,          # Peso base
    'Viral_Pneumonia': 1.2  # Peso intermedio
}

# =============================================================================
# CLASIFICADOR COVID-19
# =============================================================================

# Backbone por defecto para clasificador
DEFAULT_CLASSIFIER_BACKBONE: str = "resnet18"

# Clases del clasificador (alias de CATEGORIES para claridad)
CLASSIFIER_CLASSES: List[str] = CATEGORIES

# =============================================================================
# MODELO
# =============================================================================

# Dimensión de features del backbone ResNet-18
BACKBONE_FEATURE_DIM: int = 512

# Dimensión oculta de la cabeza de regresión
DEFAULT_HIDDEN_DIM: int = 768

# Dropout por defecto
DEFAULT_DROPOUT_RATE: float = 0.3

# =============================================================================
# ENTRENAMIENTO
# =============================================================================

# Batch size por defecto
DEFAULT_BATCH_SIZE: int = 16

# Learning rates por defecto
DEFAULT_PHASE1_LR: float = 1e-3
DEFAULT_PHASE2_BACKBONE_LR: float = 2e-5
DEFAULT_PHASE2_HEAD_LR: float = 2e-4

# Épocas por defecto
DEFAULT_PHASE1_EPOCHS: int = 15
DEFAULT_PHASE2_EPOCHS: int = 100

# =============================================================================
# LOSS
# =============================================================================

# Parámetros de Wing Loss (en píxeles, se normalizan automáticamente)
DEFAULT_WING_OMEGA: float = 10.0
DEFAULT_WING_EPSILON: float = 2.0

# Margen de simetría permitida (en píxeles)
DEFAULT_SYMMETRY_MARGIN: float = 6.0

# =============================================================================
# DATA AUGMENTATION
# =============================================================================

# Probabilidad de flip horizontal
DEFAULT_FLIP_PROB: float = 0.5

# Grados de rotación máxima
DEFAULT_ROTATION_DEGREES: float = 10.0

# Parámetros CLAHE
# Nota: tile_size=4 produce mejores resultados que 8 (usado en el modelo 3.71 px ensemble)
DEFAULT_CLAHE_CLIP_LIMIT: float = 2.0
DEFAULT_CLAHE_TILE_SIZE: int = 4

# =============================================================================
# QUICK MODE - Configuración para pruebas rápidas
# =============================================================================

# Límites de datos para modo quick (optimize-margin, etc.)
# Estos valores permiten pruebas rápidas manteniendo representatividad estadística
QUICK_MODE_MAX_TRAIN: int = 500  # Máximo de imágenes de entrenamiento
QUICK_MODE_MAX_VAL: int = 100   # Máximo de imágenes de validación
QUICK_MODE_MAX_TEST: int = 100  # Máximo de imágenes de test

# Épocas reducidas para modo quick
QUICK_MODE_EPOCHS_OPTIMIZE: int = 3   # Para optimize-margin
QUICK_MODE_EPOCHS_COMPARE: int = 5    # Para compare-architectures

# =============================================================================
# WARPING - Configuración de normalización geométrica
# =============================================================================

# Margen óptimo encontrado experimentalmente (Session 25 optimize-margin)
# Grid search evaluó márgenes: [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]
# Criterio de selección: minimizar error de warping (distancia entre puntos
# reales y predichos después del warping)
# Resultado: 1.05 fue óptimo, balanceando:
#   - Expansión suficiente para incluir estructura pulmonar completa
#   - Sin sobre-expansión que cause artefactos de warping
# 1.05 = 5% de expansión desde el centroide de landmarks
# NOTA: GROUND_TRUTH.json usa margin_scale_optimal=1.05
OPTIMAL_MARGIN_SCALE: float = 1.05

# Margen por defecto (conservador/legacy)
# 1.25 = 25% de expansión desde el centroide
# Se mantiene para compatibilidad con experimentos anteriores
DEFAULT_MARGIN_SCALE: float = 1.25

# =============================================================================
# COMBINED LOSS - Pesos para pérdida combinada
# =============================================================================

# Pesos para CombinedLandmarkLoss (basados en análisis geométrico)
# central_weight: peso para landmarks del eje central (L1, L2, L9, L10, L11)
# symmetry_weight: peso para penalización de asimetría bilateral
DEFAULT_CENTRAL_WEIGHT: float = 1.0
DEFAULT_SYMMETRY_WEIGHT: float = 0.5

# =============================================================================
# HIERARCHICAL LOSS - Parámetros para modelo jerárquico
# =============================================================================

# Escala para offsets de landmarks centrales (dt)
HIERARCHICAL_DT_SCALE: float = 0.1

# Escala para offset t bilateral
HIERARCHICAL_T_SCALE: float = 0.2

# Distancia máxima bilateral (en escala normalizada)
HIERARCHICAL_D_MAX: float = 0.7

# =============================================================================
# ENTRENAMIENTO - Parámetros adicionales
# =============================================================================

# Weight decay por defecto para Adam/AdamW
DEFAULT_WEIGHT_DECAY: float = 0.01

# Learning rate para fine-tuning
DEFAULT_FINE_TUNE_LR: float = 1e-4

# =============================================================================
# DATA AUGMENTATION - Parámetros adicionales
# =============================================================================

# Rango de brillo para augmentation
DEFAULT_BRIGHTNESS_RANGE: Tuple[float, float] = (0.8, 1.2)

# Rango de contraste para augmentation
DEFAULT_CONTRAST_RANGE: Tuple[float, float] = (0.8, 1.2)

# =============================================================================
# BILATERAL LANDMARKS - Posiciones T
# =============================================================================

# Posiciones t base para landmarks bilaterales sobre el eje L1-L2
# Corresponden a los pares: (L3,L4), (L5,L6), (L7,L8), (L12,L13), (L14,L15)
# - L3,L4 en t=0.25, L5,L6 en t=0.50, L7,L8 en t=0.75
# - L12,L13 en t=0.00 (en L1), L14,L15 en t=1.00 (en L2)
BILATERAL_T_POSITIONS: List[float] = [0.25, 0.50, 0.75, 0.00, 1.00]

# =============================================================================
# WARP - Parámetros por defecto
# =============================================================================

# Margen mínimo desde bordes de imagen para landmarks
DEFAULT_WARP_MARGIN: int = 2

# Tamaño máximo de imagen para warping (coincide con DEFAULT_IMAGE_SIZE)
DEFAULT_WARP_MAX_SIZE: int = 224

# =============================================================================
# HIERARCHICAL MODEL - Arquitectura
# =============================================================================

# Dimensión oculta para cabezas del modelo jerárquico
HIERARCHICAL_HIDDEN_DIM: int = 512

# Número de grupos para GroupNorm en modelo jerárquico
HIERARCHICAL_NUM_GROUPS: int = 32

# Número de grupos para GroupNorm en cabeza relativa (hidden_dim // 2)
HIERARCHICAL_NUM_GROUPS_HALF: int = 16
