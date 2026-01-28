"""
Modelo ResNet-18 para prediccion de landmarks
"""

import logging

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, List, Tuple

from src_v2.constants import (
    NUM_LANDMARKS,
    DEFAULT_IMAGE_SIZE,
    BACKBONE_FEATURE_DIM,
    DEFAULT_HIDDEN_DIM,
)


logger = logging.getLogger(__name__)


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Module (CVPR 2021).

    Captura dependencias de largo alcance con informacion posicional.
    Paper: "Coordinate Attention for Efficient Mobile Network Design"
    """

    def __init__(self, in_channels: int, reduction: int = 32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mid_channels = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, h, w = x.size()

        # Pool along spatial dimensions
        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1)

        # Concatenate and process
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split and generate attention
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return x * a_h * a_w


class ResNet18Landmarks(nn.Module):
    """
    ResNet-18 adaptado para regresion de landmarks.

    Arquitectura:
    - Backbone: ResNet-18 pretrained (sin FC final)
    - Head: Capas FC con dropout para regresion

    Output: 30 valores (15 landmarks x 2 coordenadas) en [0, 1]
    """

    def __init__(
        self,
        num_landmarks: int = NUM_LANDMARKS,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.5,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        use_coord_attention: bool = False,
        deep_head: bool = False
    ):
        """
        Args:
            num_landmarks: Numero de landmarks (default NUM_LANDMARKS)
            pretrained: Usar pesos pretrained de ImageNet
            freeze_backbone: Congelar backbone inicialmente
            dropout_rate: Tasa de dropout en cabeza
            hidden_dim: Dimension de capa oculta
            use_coord_attention: Usar Coordinate Attention antes de avgpool
            deep_head: Usar cabeza mas profunda con BatchNorm
        """
        super().__init__()

        self.num_landmarks = num_landmarks
        self.output_dim = num_landmarks * 2
        self.use_coord_attention = use_coord_attention

        # Backbone ResNet-18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Remover FC original, mantener hasta layer4
        self.backbone_conv = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # Coordinate Attention (opcional)
        self.coord_attention = CoordinateAttention(BACKBONE_FEATURE_DIM, reduction=32) if use_coord_attention else None

        # Average pooling
        self.avgpool = resnet.avgpool

        # Dimension de features del backbone
        self.feature_dim = BACKBONE_FEATURE_DIM

        # Cabeza de regresion
        if deep_head:
            # Cabeza profunda con GroupNorm (estable con batch pequeño)
            # GroupNorm: normaliza sobre grupos de canales, no sobre batch
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.feature_dim, 512),
                nn.GroupNorm(num_groups=32, num_channels=512),  # 512/32=16 canales por grupo
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(512, hidden_dim),
                nn.GroupNorm(num_groups=min(16, hidden_dim // 16) if hidden_dim >= 16 else 1, num_channels=hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(hidden_dim, self.output_dim),
                nn.Sigmoid()  # Output en [0, 1]
            )
        else:
            # Cabeza original
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate * 0.5),  # Menor dropout en segunda capa
                nn.Linear(hidden_dim, self.output_dim),
                nn.Sigmoid()  # Output en [0, 1]
            )

        # Congelar backbone si se especifica
        if freeze_backbone:
            self.freeze_backbone()

    @property
    def backbone(self):
        """Propiedad para compatibilidad con codigo existente."""
        return self.backbone_conv

    def freeze_backbone(self):
        """Congela los parametros del backbone."""
        for param in self.backbone_conv.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Descongela los parametros del backbone."""
        for param in self.backbone_conv.parameters():
            param.requires_grad = True

    def freeze_coord_attention(self):
        """Congela los parametros de Coordinate Attention."""
        if self.coord_attention is not None:
            for param in self.coord_attention.parameters():
                param.requires_grad = False

    def unfreeze_coord_attention(self):
        """Descongela los parametros de Coordinate Attention."""
        if self.coord_attention is not None:
            for param in self.coord_attention.parameters():
                param.requires_grad = True

    def freeze_all_except_head(self):
        """
        Congela backbone y CoordAttention para Phase 1.
        Solo la cabeza de regresion queda entrenable.
        """
        self.freeze_backbone()
        self.freeze_coord_attention()

    def unfreeze_all(self):
        """
        Descongela todo para Phase 2 (fine-tuning completo).
        """
        self.unfreeze_backbone()
        self.unfreeze_coord_attention()

    def get_trainable_params(self) -> List[dict]:
        """
        Retorna grupos de parametros para optimizador con LR diferenciado.

        Returns:
            Lista de dicts para torch.optim (backbone_params, coord_attn_params, head_params)
        """
        backbone_params = [p for p in self.backbone_conv.parameters() if p.requires_grad]
        head_params = [p for p in self.head.parameters() if p.requires_grad]

        # CoordAttention tiene su propio grupo (se entrena con LR de backbone)
        coord_attn_params = []
        if self.coord_attention is not None:
            coord_attn_params = [p for p in self.coord_attention.parameters() if p.requires_grad]

        # Combinar backbone y coord_attention en un grupo (mismo LR bajo)
        # Head en otro grupo (LR mas alto)
        feature_params = backbone_params + coord_attn_params

        return [
            {'params': feature_params, 'name': 'features'},  # backbone + coord_attn
            {'params': head_params, 'name': 'head'}
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor de imagenes (B, 3, H, W)

        Returns:
            Tensor de coordenadas (B, 30) en [0, 1]
        """
        # Backbone convolutional
        features = self.backbone_conv(x)  # (B, 512, 7, 7)

        # Coordinate Attention (si esta habilitado)
        if self.coord_attention is not None:
            features = self.coord_attention(features)

        # Global average pooling
        features = self.avgpool(features)  # (B, 512, 1, 1)

        # Head de regresion
        output = self.head(features)
        return output

    def predict_landmarks(
        self,
        x: torch.Tensor,
        image_size: int = DEFAULT_IMAGE_SIZE
    ) -> torch.Tensor:
        """
        Prediccion con desnormalizacion a pixeles.

        Args:
            x: Tensor de imagenes (B, 3, H, W)
            image_size: Tamano de imagen para desnormalizar

        Returns:
            Tensor (B, 15, 2) en pixeles
        """
        output = self.forward(x)  # (B, 30)
        output = output.view(-1, 15, 2)  # (B, 15, 2)
        output = output * image_size  # Desnormalizar
        return output


def create_model(
    num_landmarks: int = NUM_LANDMARKS,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout_rate: float = 0.5,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    use_coord_attention: bool = False,
    deep_head: bool = False,
    device: Optional[torch.device] = None
) -> ResNet18Landmarks:
    """
    Factory para crear modelo.

    Args:
        num_landmarks: Numero de landmarks
        pretrained: Usar pesos pretrained
        freeze_backbone: Congelar backbone
        dropout_rate: Tasa de dropout
        hidden_dim: Dimension oculta
        use_coord_attention: Usar Coordinate Attention
        deep_head: Usar cabeza profunda con BatchNorm
        device: Dispositivo (auto-detecta si None)

    Returns:
        Modelo inicializado
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ResNet18Landmarks(
        num_landmarks=num_landmarks,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout_rate,
        hidden_dim=hidden_dim,
        use_coord_attention=use_coord_attention,
        deep_head=deep_head
    )

    return model.to(device)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Cuenta parametros totales y entrenables.

    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
