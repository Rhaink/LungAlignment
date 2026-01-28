"""
Arquitectura Jerarquica para Prediccion de Landmarks

La idea es explotar la estructura geometrica del proceso de etiquetado:
1. Predecir el eje central (L1, L2) primero
2. Predecir los parametros relativos al eje:
   - t_i para landmarks centrales (L9, L10, L11) - posicion en el eje
   - distancias perpendiculares para landmarks bilaterales

Estructura del etiquetado descubierta:
- L9, L10, L11 dividen el eje L1-L2 en exactamente 4 partes (t = 0.25, 0.50, 0.75)
- Los landmarks bilaterales estan a distancias perpendiculares del eje
"""

import logging

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional

from src_v2.constants import (
    SYMMETRIC_PAIRS,
    CENTRAL_LANDMARKS,
    BACKBONE_FEATURE_DIM,
    HIERARCHICAL_DT_SCALE,
    HIERARCHICAL_T_SCALE,
    HIERARCHICAL_D_MAX,
    BILATERAL_T_POSITIONS,
    HIERARCHICAL_HIDDEN_DIM,
    HIERARCHICAL_NUM_GROUPS,
    HIERARCHICAL_NUM_GROUPS_HALF,
    DEFAULT_PHASE2_BACKBONE_LR,
    DEFAULT_PHASE2_HEAD_LR,
)
from src_v2.utils.geometry import compute_perpendicular_vector


logger = logging.getLogger(__name__)


class HierarchicalLandmarkModel(nn.Module):
    """
    Modelo jerarquico que predice landmarks en dos etapas:
    1. Predecir eje central (L1, L2)
    2. Predecir parametros relativos al eje

    Landmarks (0-indexed):
    - L1 (0): Superior del eje
    - L2 (1): Inferior del eje
    - L3 (2), L4 (3): Par bilateral superior
    - L5 (4), L6 (5): Par bilateral medio
    - L7 (6), L8 (7): Par bilateral inferior
    - L9 (8): Centro superior (t=0.25)
    - L10 (9): Centro medio (t=0.50)
    - L11 (10): Centro inferior (t=0.75)
    - L12 (11), L13 (12): Par bilateral apices
    - L14 (13), L15 (14): Par bilateral costofrenicos
    """

    # Pares bilaterales (izq, der) - usando constante de constants.py
    # Equivalente a: [(2,3), (4,5), (6,7), (11,12), (13,14)]
    BILATERAL_PAIRS = SYMMETRIC_PAIRS

    # Parametros t teoricos para landmarks centrales
    CENTRAL_T = {
        8: 0.25,   # L9
        9: 0.50,   # L10
        10: 0.75,  # L11
    }

    def __init__(
        self,
        pretrained: bool = True,
        hidden_dim: int = HIERARCHICAL_HIDDEN_DIM,
        dropout: float = 0.3,
        learn_t_offsets: bool = True,  # Aprender desviaciones de t teorico
    ):
        super().__init__()

        self.learn_t_offsets = learn_t_offsets

        # Backbone compartido
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = resnet.avgpool

        # Cabeza para predecir eje (L1, L2) - 4 valores
        self.axis_head = nn.Sequential(
            nn.Linear(BACKBONE_FEATURE_DIM, hidden_dim),
            nn.GroupNorm(HIERARCHICAL_NUM_GROUPS, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),  # L1_x, L1_y, L2_x, L2_y
            nn.Sigmoid()
        )

        # Cabeza para predecir parametros relativos
        # Para cada landmark bilateral: t (posicion en eje), d_left, d_right (distancias perpendiculares)
        # Para centrales: dt (offset del t teorico)
        # Total: 5 pares * 3 = 15 + 3 offsets centrales = 18
        # Pero mejor predecir directamente:
        #   - 3 valores para centrales (dt_9, dt_10, dt_11)
        #   - 5*3 = 15 valores para bilaterales (t, d_left, d_right para cada par)

        relative_dim = 3 + 5 * 3  # 3 dt + 15 (t, d_l, d_r) = 18

        self.relative_head = nn.Sequential(
            nn.Linear(BACKBONE_FEATURE_DIM + 4, hidden_dim),  # Features + axis info
            nn.GroupNorm(HIERARCHICAL_NUM_GROUPS, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GroupNorm(HIERARCHICAL_NUM_GROUPS_HALF, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, relative_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass que devuelve 30 coordenadas normalizadas [0,1]

        Args:
            x: Batch de imagenes (B, 3, 224, 224)

        Returns:
            landmarks: (B, 30) coordenadas normalizadas
        """
        batch_size = x.size(0)

        # Extraer features
        features = self.backbone(x)
        features = self.avgpool(features)
        features = features.view(batch_size, -1)  # (B, 512)

        # Predecir eje
        axis = self.axis_head(features)  # (B, 4)
        L1 = axis[:, :2]  # (B, 2)
        L2 = axis[:, 2:]  # (B, 2)

        # Concatenar features con info del eje
        features_with_axis = torch.cat([features, axis], dim=1)  # (B, 516)

        # Predecir parametros relativos
        relative_params = self.relative_head(features_with_axis)  # (B, 18)

        # Reconstruir landmarks desde parametros
        landmarks = self._reconstruct_landmarks(L1, L2, relative_params)

        return landmarks

    def _reconstruct_landmarks(
        self,
        L1: torch.Tensor,
        L2: torch.Tensor,
        params: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruir coordenadas de landmarks desde parametros relativos.

        Args:
            L1: (B, 2) punto superior del eje
            L2: (B, 2) punto inferior del eje
            params: (B, 18) parametros relativos

        Returns:
            landmarks: (B, 30) coordenadas de todos los landmarks
        """
        batch_size = L1.size(0)
        device = L1.device

        # Inicializar output
        landmarks = torch.zeros(batch_size, 15, 2, device=device)

        # L1 y L2 directamente
        landmarks[:, 0] = L1
        landmarks[:, 1] = L2

        # Vector del eje
        axis_vec = L2 - L1  # (B, 2)
        axis_len = torch.norm(axis_vec, dim=1, keepdim=True) + 1e-8  # (B, 1)

        # Vector perpendicular (rotacion 90 grados) usando función centralizada
        perp_unit = compute_perpendicular_vector(axis_vec)  # (B, 2)

        # Extraer parametros
        dt_centrales = torch.tanh(params[:, :3]) * HIERARCHICAL_DT_SCALE  # Offsets pequenos
        bilateral_params = params[:, 3:]  # (B, 15)

        # Landmarks centrales (L9, L10, L11)
        # Nota: t_base se define inline en el loop, no se necesita tensor separado
        for i, (landmark_idx, t_base) in enumerate([(8, 0.25), (9, 0.50), (10, 0.75)]):
            t = t_base + dt_centrales[:, i]  # (B,)
            landmarks[:, landmark_idx] = L1 + t.unsqueeze(1) * axis_vec

        # Landmarks bilaterales
        # Usar posiciones t base desde constantes (CORREGIDO basado en análisis de GT)
        # L12,L13 están en t=0.0 (en L1), L14,L15 están en t=1.0 (en L2)

        for pair_idx, (left_idx, right_idx) in enumerate(self.BILATERAL_PAIRS):
            # Extraer parametros para este par: t, d_left, d_right
            p_start = pair_idx * 3
            t_offset = torch.tanh(bilateral_params[:, p_start]) * HIERARCHICAL_T_SCALE  # Offset de t
            # CORREGIDO: aumentar rango para permitir distancias mayores
            d_left = torch.sigmoid(bilateral_params[:, p_start + 1]) * HIERARCHICAL_D_MAX
            d_right = torch.sigmoid(bilateral_params[:, p_start + 2]) * HIERARCHICAL_D_MAX

            # Posicion base en el eje
            t = BILATERAL_T_POSITIONS[pair_idx] + t_offset
            base_point = L1 + t.unsqueeze(1) * axis_vec  # (B, 2)

            # CORREGIDO: invertir signos - en las coordenadas de imagen,
            # "izquierda" anatómica está en dirección +perp_unit, "derecha" en -perp_unit
            landmarks[:, left_idx] = base_point + d_left.unsqueeze(1) * perp_unit * axis_len
            landmarks[:, right_idx] = base_point - d_right.unsqueeze(1) * perp_unit * axis_len

        # Reshape a (B, 30)
        landmarks = landmarks.view(batch_size, -1)

        # Clamp a [0, 1]
        landmarks = torch.clamp(landmarks, 0, 1)

        return landmarks

    def freeze_backbone(self):
        """Congelar backbone para Phase 1 training."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Descongelar backbone para Phase 2 fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_trainable_params(
        self,
        backbone_lr: float = DEFAULT_PHASE2_BACKBONE_LR,
        head_lr: float = DEFAULT_PHASE2_HEAD_LR
    ):
        """Obtener grupos de parametros con LR diferenciado."""
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.axis_head.parameters()) + list(self.relative_head.parameters())

        return [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': head_lr}
        ]


class AxisLoss(nn.Module):
    """
    Loss adicional que penaliza errores en la prediccion del eje.
    El eje (L1, L2) es critico porque todos los demas landmarks dependen de el.
    """

    def __init__(self, weight: float = 2.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calcular loss del eje.

        Args:
            pred: (B, 30) predicciones
            target: (B, 30) ground truth

        Returns:
            loss: scalar
        """
        # Extraer L1 y L2 (indices 0,1 y 2,3 de las 30 coordenadas)
        pred_L1 = pred[:, :2]
        pred_L2 = pred[:, 2:4]
        target_L1 = target[:, :2]
        target_L2 = target[:, 2:4]

        # Error euclidiano
        error_L1 = torch.sqrt(((pred_L1 - target_L1) ** 2).sum(dim=1))
        error_L2 = torch.sqrt(((pred_L2 - target_L2) ** 2).sum(dim=1))

        return self.weight * (error_L1.mean() + error_L2.mean())


class CentralAlignmentLossHierarchical(nn.Module):
    """
    Loss que fuerza L9, L10, L11 a estar sobre el eje predicho.
    Dado que el modelo jerarquico ya reconstruye estos puntos sobre el eje,
    este loss es mas para verificacion que para entrenamiento.
    """

    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Verificar alineacion de centrales.

        Args:
            pred: (B, 30) predicciones

        Returns:
            loss: scalar (deberia ser ~0 si el modelo funciona bien)
        """
        # Reshape a (B, 15, 2)
        pred = pred.view(-1, 15, 2)

        L1 = pred[:, 0]
        L2 = pred[:, 1]

        axis = L2 - L1
        axis_len = torch.norm(axis, dim=1, keepdim=True) + 1e-8
        axis_unit = axis / axis_len

        total_dist = 0
        for idx in [8, 9, 10]:  # L9, L10, L11
            vec = pred[:, idx] - L1
            proj = (vec * axis_unit).sum(dim=1, keepdim=True) * axis_unit
            perp = vec - proj
            dist = torch.norm(perp, dim=1)
            total_dist = total_dist + dist

        return self.weight * total_dist.mean() / 3


if __name__ == "__main__":
    # Test - configurar logging para ver salida
    logging.basicConfig(level=logging.DEBUG)

    # Test
    model = HierarchicalLandmarkModel()
    x = torch.randn(4, 3, 224, 224)

    with torch.no_grad():
        out = model(x)

    logger.debug("Input shape: %s", x.shape)
    logger.debug("Output shape: %s", out.shape)
    logger.debug("Output range: [%.3f, %.3f]", out.min(), out.max())

    # Verificar que landmarks centrales estan alineados
    out_reshaped = out.view(-1, 15, 2)
    L1 = out_reshaped[:, 0]
    L2 = out_reshaped[:, 1]
    L10 = out_reshaped[:, 9]

    axis = L2 - L1
    axis_len = torch.norm(axis, dim=1, keepdim=True)
    axis_unit = axis / (axis_len + 1e-8)

    # Distancia perpendicular de L10 al eje
    vec = L10 - L1
    proj = (vec * axis_unit).sum(dim=1, keepdim=True) * axis_unit
    perp = vec - proj
    dist = torch.norm(perp, dim=1)
    logger.debug("L10 perpendicular distance to axis: %.6f (should be ~0)", dist.mean())
