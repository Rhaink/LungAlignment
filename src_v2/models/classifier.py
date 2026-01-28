"""
Clasificador CNN para COVID-19/Normal/Viral_Pneumonia.

Arquitecturas soportadas:
- ResNet-18 (default)
- ResNet-50
- EfficientNet-B0
- DenseNet-121 (recomendado para mejor generalización)
- AlexNet
- VGG-16
- MobileNetV2

Basado en scripts/train_classifier.py (Sesion 22).
DenseNet-121 agregado en Sesion 18.
Arquitecturas adicionales agregadas en Sesion 20.
"""

from collections import Counter
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torchvision import models, transforms


class GrayscaleToRGB:
    """Convierte imagen grayscale a RGB (3 canales)."""

    def __call__(self, img):
        """
        Args:
            img: PIL Image

        Returns:
            PIL Image con 3 canales RGB
        """
        if img.mode == "RGB":
            return img
        return img.convert("RGB")


class ImageClassifier(nn.Module):
    """
    Clasificador CNN para COVID-19 usando transfer learning.

    Soporta multiples arquitecturas como backbone:
    - resnet18, resnet50
    - efficientnet_b0
    - densenet121
    - alexnet
    - vgg16
    - mobilenet_v2
    """

    SUPPORTED_BACKBONES = (
        "resnet18",
        "resnet50",
        "efficientnet_b0",
        "densenet121",
        "alexnet",
        "vgg16",
        "mobilenet_v2",
    )

    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        """
        Args:
            backbone: Arquitectura base. Ver SUPPORTED_BACKBONES para opciones validas.
            num_classes: Numero de clases de salida
            pretrained: Usar pesos preentrenados de ImageNet
            dropout: Tasa de dropout antes de capa final
        """
        super().__init__()

        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Backbone '{backbone}' no soportado. "
                f"Opciones: {self.SUPPORTED_BACKBONES}"
            )

        self.backbone_name = backbone
        self.num_classes = num_classes
        self.dropout_rate = dropout

        if backbone == "resnet18":
            if pretrained:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.resnet18(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, num_classes),
            )

        elif backbone == "efficientnet_b0":
            if pretrained:
                weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.efficientnet_b0(weights=weights)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, num_classes),
            )

        elif backbone == "densenet121":
            if pretrained:
                weights = models.DenseNet121_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.densenet121(weights=weights)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, num_classes),
            )

        elif backbone == "resnet50":
            if pretrained:
                weights = models.ResNet50_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, num_classes),
            )

        elif backbone == "alexnet":
            if pretrained:
                weights = models.AlexNet_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.alexnet(weights=weights)
            # AlexNet classifier: Sequential(Dropout, Linear(9216, 4096), ReLU, Dropout, Linear, ReLU, Linear)
            num_features = self.backbone.classifier[6].in_features
            self.backbone.classifier[6] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, num_classes),
            )

        elif backbone == "vgg16":
            if pretrained:
                weights = models.VGG16_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.vgg16(weights=weights)
            # VGG16 classifier: Sequential(Linear(25088, 4096), ReLU, Dropout, Linear, ReLU, Dropout, Linear)
            num_features = self.backbone.classifier[6].in_features
            self.backbone.classifier[6] = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, num_classes),
            )

        elif backbone == "mobilenet_v2":
            if pretrained:
                weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.mobilenet_v2(weights=weights)
            # MobileNetV2 classifier: Sequential(Dropout, Linear(1280, 1000))
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor de entrada (batch, 3, H, W)

        Returns:
            Logits (batch, num_classes)
        """
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predice probabilidades (softmax).

        Args:
            x: Tensor de entrada (batch, 3, H, W)

        Returns:
            Probabilidades (batch, num_classes)
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predice clase (argmax).

        Args:
            x: Tensor de entrada (batch, 3, H, W)

        Returns:
            Indices de clase predicha (batch,)
        """
        logits = self.forward(x)
        return logits.argmax(dim=1)


def create_classifier(
    backbone: str = "resnet18",
    num_classes: int = 3,
    pretrained: bool = True,
    dropout: float = 0.3,
    checkpoint: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> ImageClassifier:
    """
    Factory function para crear un clasificador.

    Args:
        backbone: Arquitectura base. Ver ImageClassifier.SUPPORTED_BACKBONES para opciones validas.
        num_classes: Numero de clases
        pretrained: Usar pesos ImageNet (ignorado si checkpoint se provee)
        dropout: Tasa de dropout
        checkpoint: Path a checkpoint guardado (opcional)
        device: Dispositivo para cargar el modelo

    Returns:
        Modelo ImageClassifier
    """
    if checkpoint is not None:
        # Cargar desde checkpoint
        ckpt = torch.load(checkpoint, map_location=device or "cpu", weights_only=False)

        # Detectar parametros del checkpoint
        if "model_name" in ckpt:
            backbone = ckpt["model_name"]
        if "class_names" in ckpt:
            num_classes = len(ckpt["class_names"])

        model = ImageClassifier(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=False,  # No necesario si cargamos pesos
            dropout=dropout,
        )

        # Cargar state dict
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        # Detectar formato antiguo (sin prefijo "backbone.")
        # El formato antiguo tiene claves como "conv1.weight", "layer1.0.conv1.weight"
        # El formato nuevo tiene claves como "backbone.conv1.weight"
        sample_key = next(iter(state_dict.keys()))
        if not sample_key.startswith("backbone."):
            # Convertir formato antiguo a nuevo
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = f"backbone.{key}"
                new_state_dict[new_key] = value
            state_dict = new_state_dict

        model.load_state_dict(state_dict)

    else:
        model = ImageClassifier(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )

    if device is not None:
        model = model.to(device)

    return model


def get_classifier_transforms(
    train: bool = False,
    img_size: int = 224,
) -> transforms.Compose:
    """
    Obtiene transformaciones para el clasificador.

    Args:
        train: Si True, incluye augmentaciones
        img_size: Tamano de imagen de salida

    Returns:
        transforms.Compose con las transformaciones
    """
    # Normalizacion ImageNet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if train:
        return transforms.Compose([
            GrayscaleToRGB(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            GrayscaleToRGB(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])


def get_class_weights(
    labels: List[int],
    num_classes: int = 3,
) -> torch.Tensor:
    """
    Calcula pesos de clase inversamente proporcionales a la frecuencia.

    Args:
        labels: Lista de etiquetas de clase
        num_classes: Numero total de clases

    Returns:
        Tensor de pesos por clase
    """
    class_counts = Counter(labels)
    n_samples = len(labels)

    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)  # Evitar division por cero
        weight = n_samples / (num_classes * count)
        weights.append(weight)

    return torch.FloatTensor(weights)


def load_classifier_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[ImageClassifier, dict]:
    """
    Carga un checkpoint de clasificador con metadatos.

    Args:
        checkpoint_path: Path al archivo .pt
        device: Dispositivo destino

    Returns:
        Tupla (modelo, metadatos)
    """
    ckpt = torch.load(
        checkpoint_path,
        map_location=device or "cpu",
        weights_only=False,
    )

    # Extraer metadatos
    metadata = {
        "backbone": ckpt.get("model_name", "resnet18"),
        "class_names": ckpt.get("class_names", ["COVID", "Normal", "Viral_Pneumonia"]),
        "best_val_f1": ckpt.get("best_val_f1", None),
    }

    # Crear modelo
    model = create_classifier(
        backbone=metadata["backbone"],
        num_classes=len(metadata["class_names"]),
        checkpoint=checkpoint_path,
        device=device,
    )

    return model, metadata
