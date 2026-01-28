"""
Funciones de perdida para landmark prediction

Incluye:
- WingLoss: Loss robusto para landmarks
- WeightedWingLoss: Con pesos por landmark
- CentralAlignmentLoss: Forzar L9,L10,L11 sobre eje L1-L2
- SoftSymmetryLoss: Penaliza asimetria > margen
- CombinedLandmarkLoss: Combinacion ponderada
"""

import logging

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict

from src_v2.constants import (
    SYMMETRIC_PAIRS,
    CENTRAL_LANDMARKS,
    DEFAULT_IMAGE_SIZE,
)
from src_v2.utils.geometry import compute_perpendicular_vector


logger = logging.getLogger(__name__)


class WingLoss(nn.Module):
    """
    Wing Loss para landmark localization.

    Ref: Wing Loss for Robust Facial Landmark Localisation with CNNs (CVPR 2018)

    Para errores pequenos (< omega): comportamiento logaritmico (mayor gradiente)
    Para errores grandes (>= omega): comportamiento lineal (estable)

    IMPORTANTE: Si las coordenadas estan normalizadas a [0,1], usar normalized=True
    para escalar omega y epsilon automaticamente.
    """

    def __init__(self, omega: float = 10.0, epsilon: float = 2.0,
                 normalized: bool = True, image_size: int = DEFAULT_IMAGE_SIZE):
        """
        Args:
            omega: Umbral para cambio de regimen (en pixeles si normalized=True)
            epsilon: Curvatura de la parte logaritmica (en pixeles si normalized=True)
            normalized: Si True, escala omega/epsilon para coordenadas en [0,1]
            image_size: Tamano de imagen para escalar (solo si normalized=True)
        """
        super().__init__()

        # Si las coordenadas estan normalizadas, escalar omega y epsilon
        if normalized:
            self.omega = omega / image_size
            self.epsilon = epsilon / image_size
        else:
            self.omega = omega
            self.epsilon = epsilon

        # Constante para continuidad
        self.C = self.omega - self.omega * np.log(1 + self.omega / self.epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calcula Wing Loss.

        Args:
            pred: Predicciones (B, 30) o (B, 15, 2)
            target: Ground truth con misma forma

        Returns:
            Loss escalar
        """
        diff = torch.abs(pred - target)

        # Wing Loss por elemento
        loss = torch.where(
            diff < self.omega,
            self.omega * torch.log(1 + diff / self.epsilon),
            diff - self.C
        )

        return loss.mean()


class WeightedWingLoss(nn.Module):
    """
    Wing Loss con pesos diferenciados por landmark.

    Permite dar mas importancia a landmarks especificos (ej: L14, L15 mas dificiles).
    """

    def __init__(
        self,
        omega: float = 10.0,
        epsilon: float = 2.0,
        weights: Optional[torch.Tensor] = None,
        normalized: bool = True,
        image_size: int = DEFAULT_IMAGE_SIZE
    ):
        """
        Args:
            omega: Umbral para Wing Loss (en pixeles si normalized=True)
            epsilon: Curvatura (en pixeles si normalized=True)
            weights: Tensor (15,) de pesos por landmark. None = pesos iguales.
            normalized: Si True, escala omega/epsilon para coordenadas en [0,1]
            image_size: Tamano de imagen para escalar
        """
        super().__init__()

        # Si las coordenadas estan normalizadas, escalar omega y epsilon
        if normalized:
            self.omega = omega / image_size
            self.epsilon = epsilon / image_size
        else:
            self.omega = omega
            self.epsilon = epsilon

        self.C = self.omega - self.omega * np.log(1 + self.omega / self.epsilon)

        # Pesos por defecto (sugeridos en DESCUBRIMIENTOS_GEOMETRICOS.md)
        if weights is None:
            weights = torch.ones(15)
        self.register_buffer('weights', weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calcula Weighted Wing Loss.

        Args:
            pred: Predicciones (B, 30)
            target: Ground truth (B, 30)

        Returns:
            Loss escalar ponderado
        """
        B = pred.shape[0]

        # Reshape a (B, 15, 2)
        pred = pred.view(B, 15, 2)
        target = target.view(B, 15, 2)

        diff = torch.abs(pred - target)

        # Wing Loss por elemento
        loss = torch.where(
            diff < self.omega,
            self.omega * torch.log(1 + diff / self.epsilon),
            diff - self.C
        )

        # Promedio sobre x,y: (B, 15)
        loss = loss.mean(dim=-1)

        # Aplicar pesos: (B, 15) * (15,) -> (B, 15)
        loss = loss * self.weights.unsqueeze(0)

        return loss.mean()


class CentralAlignmentLoss(nn.Module):
    """
    Penaliza que L9, L10, L11 no esten sobre el eje L1-L2.

    Basado en DESCUBRIMIENTO: Los puntos centrales estan a solo ~1.3 px
    del eje perfecto en el Ground Truth. Esta es una restriccion casi perfecta.

    NOTA: Opera en espacio normalizado [0,1] para mantener escala similar
    al Wing Loss. La distancia de 1.3 px en 224 px = 0.0058 normalizado.
    """

    def __init__(self, image_size: int = DEFAULT_IMAGE_SIZE):
        """
        Args:
            image_size: Tamano de imagen (para referencia, no escala)
        """
        super().__init__()
        self.image_size = image_size

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Calcula distancia de L9,L10,L11 al eje L1-L2.

        Args:
            pred: Predicciones (B, 30) en [0, 1]

        Returns:
            Loss escalar (promedio de distancias perpendiculares en [0,1])
        """
        B = pred.shape[0]
        pred = pred.view(B, 15, 2)

        # Puntos del eje
        L1 = pred[:, 0]  # (B, 2)
        L2 = pred[:, 1]  # (B, 2)

        # Vector del eje
        eje = L2 - L1  # (B, 2)
        eje_len = torch.norm(eje, dim=1, keepdim=True) + 1e-8  # (B, 1)
        eje_unit = eje / eje_len  # (B, 2)

        # Calcular distancia perpendicular para cada punto central
        total_dist = 0.0
        for idx in CENTRAL_LANDMARKS:
            point = pred[:, idx]  # (B, 2)
            vec = point - L1  # Vector desde L1 al punto

            # Proyeccion sobre el eje
            proj_len = (vec * eje_unit).sum(dim=1, keepdim=True)  # (B, 1)
            proj = proj_len * eje_unit  # (B, 2)

            # Componente perpendicular
            perp = vec - proj  # (B, 2)
            dist = torch.norm(perp, dim=1)  # (B,)

            total_dist = total_dist + dist

        # Promedio sobre los puntos centrales
        # NO multiplicar por image_size - mantener en escala normalizada [0,1]
        loss = total_dist / len(CENTRAL_LANDMARKS)

        return loss.mean()


class SoftSymmetryLoss(nn.Module):
    """
    Penaliza SOLO asimetrias grandes (> margen).

    IMPORTANTE: El Ground Truth tiene asimetria natural de 5.5-7.9 px.
    NO debemos forzar simetria perfecta.

    NOTA: Opera en espacio normalizado [0,1] para mantener escala similar
    al Wing Loss. El margen de 6 px en 224 px = 0.027 normalizado.
    """

    def __init__(self, margin: float = 6.0, image_size: int = DEFAULT_IMAGE_SIZE):
        """
        Args:
            margin: Margen de asimetria permitida en pixeles
            image_size: Tamano de imagen para normalizar margen
        """
        super().__init__()
        # Normalizar margen a [0, 1]
        self.margin = margin / image_size
        self.image_size = image_size

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Calcula soft symmetry loss.

        Solo penaliza si |d_izq - d_der| > margin

        Args:
            pred: Predicciones (B, 30) en [0, 1]

        Returns:
            Loss escalar (en escala normalizada [0,1])
        """
        B = pred.shape[0]
        pred = pred.view(B, 15, 2)

        # Puntos del eje
        L1 = pred[:, 0]  # (B, 2)
        L2 = pred[:, 1]  # (B, 2)

        # Vector perpendicular al eje usando función centralizada
        eje = L2 - L1
        perp = compute_perpendicular_vector(eje)  # (B, 2)

        total_loss = 0.0
        for left_idx, right_idx in SYMMETRIC_PAIRS:
            left = pred[:, left_idx]  # (B, 2)
            right = pred[:, right_idx]  # (B, 2)

            # Distancias perpendiculares (con signo)
            d_left = (left - L1) * perp
            d_left = d_left.sum(dim=1)  # (B,)

            d_right = (right - L1) * perp
            d_right = d_right.sum(dim=1)  # (B,)

            # Asimetria = diferencia de magnitudes
            asim = torch.abs(torch.abs(d_left) - torch.abs(d_right))

            # Soft loss: max(0, asim - margin)^2
            loss = torch.relu(asim - self.margin) ** 2
            total_loss = total_loss + loss

        # Promedio sobre pares
        # NO multiplicar por image_size^2 - mantener en escala normalizada
        loss = total_loss / len(SYMMETRIC_PAIRS)

        return loss.mean()


class CombinedLandmarkLoss(nn.Module):
    """
    Loss combinado para landmark prediction.

    total_loss = wing_loss
               + alpha * central_alignment_loss
               + beta * soft_symmetry_loss

    IMPORTANTE: Todos los terminos operan en espacio normalizado [0,1]
    para mantener gradientes balanceados.
    """

    def __init__(
        self,
        wing_omega: float = 10.0,
        wing_epsilon: float = 2.0,
        landmark_weights: Optional[torch.Tensor] = None,
        central_weight: float = 1.0,
        symmetry_weight: float = 0.5,
        symmetry_margin: float = 6.0,
        image_size: int = DEFAULT_IMAGE_SIZE
    ):
        """
        Args:
            wing_omega: Parametro omega de Wing Loss (en pixeles, se normaliza)
            wing_epsilon: Parametro epsilon de Wing Loss (en pixeles, se normaliza)
            landmark_weights: Pesos por landmark (15,)
            central_weight: Peso del Central Alignment Loss (default=1.0)
            symmetry_weight: Peso del Soft Symmetry Loss (default=0.5)
            symmetry_margin: Margen de simetria en pixeles
            image_size: Tamano de imagen
        """
        super().__init__()

        self.wing_loss = WeightedWingLoss(
            omega=wing_omega,
            epsilon=wing_epsilon,
            weights=landmark_weights,
            normalized=True,
            image_size=image_size
        )
        self.central_loss = CentralAlignmentLoss(image_size=image_size)
        self.symmetry_loss = SoftSymmetryLoss(
            margin=symmetry_margin,
            image_size=image_size
        )

        self.central_weight = central_weight
        self.symmetry_weight = symmetry_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calcula loss combinado.

        Args:
            pred: Predicciones (B, 30)
            target: Ground truth (B, 30)

        Returns:
            Dict con 'total', 'wing', 'central', 'symmetry'
        """
        wing = self.wing_loss(pred, target)
        central = self.central_loss(pred)
        symmetry = self.symmetry_loss(pred)

        total = wing + self.central_weight * central + self.symmetry_weight * symmetry

        return {
            'total': total,
            'wing': wing,
            'central': central,
            'symmetry': symmetry
        }


def get_landmark_weights(strategy: str = 'inverse_variance') -> torch.Tensor:
    """
    Genera pesos por landmark segun estrategia.

    Args:
        strategy: 'uniform', 'inverse_variance', 'custom'

    Returns:
        Tensor (15,) de pesos normalizados
    """
    if strategy == 'uniform':
        return torch.ones(15)

    elif strategy == 'inverse_variance':
        # Pesos basados en variabilidad (σ) de cada landmark del Ground Truth.
        # Landmarks con mayor σ (mas dificiles) tienen menor peso para evitar
        # que el modelo se obsesione con ellos.
        # Fuente: docs/reportes/REPORTE_VERIFICACION_DESCUBRIMIENTOS_GEOMETRICOS.md, Seccion 7
        # Valores de σ: L9=19.2px, L14=35.2px, etc.
        # Peso = normalizado(1/σ) para que landmarks faciles contribuyan mas.
        weights = torch.tensor([
            1.16,  # L1
            0.79,  # L2
            1.07,  # L3
            1.11,  # L4
            1.00,  # L5
            1.04,  # L6
            0.85,  # L7
            0.89,  # L8
            1.30,  # L9 (central, mas facil, mayor peso)
            1.21,  # L10
            0.99,  # L11
            1.06,  # L12
            1.07,  # L13
            0.71,  # L14 (costofrenico, dificil, menor peso)
            0.74,  # L15
        ])
        return weights

    elif strategy == 'custom':
        # Pesos enfocados en landmarks criticos
        weights = torch.tensor([
            1.5,   # L1 (eje, critico)
            1.5,   # L2 (eje, critico)
            1.2,   # L3
            1.2,   # L4
            1.3,   # L5
            1.3,   # L6
            1.1,   # L7
            1.1,   # L8
            1.5,   # L9 (central)
            1.4,   # L10
            1.4,   # L11
            1.0,   # L12
            1.0,   # L13
            2.0,   # L14 (costofrenico, mas peso por dificultad)
            2.0,   # L15
        ])
        return weights

    else:
        return torch.ones(15)
