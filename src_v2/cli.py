"""
CLI principal para COVID-19 Landmark Detection.

Uso:
    python -m src_v2 train --help
    python -m src_v2 evaluate --checkpoint model.pt
    python -m src_v2 predict --image xray.png --checkpoint model.pt
    python -m src_v2 warp --input-dir data/ --output-dir warped/
"""

import logging
import random
import sys

import numpy as np
from pathlib import Path
from typing import Optional

import typer

from src_v2.constants import (
    DEFAULT_IMAGE_SIZE,
    NUM_LANDMARKS,
    DEFAULT_WING_OMEGA,
    DEFAULT_WING_EPSILON,
    DEFAULT_SYMMETRY_MARGIN,
    DEFAULT_PHASE1_LR,
    DEFAULT_PHASE2_BACKBONE_LR,
    DEFAULT_PHASE2_HEAD_LR,
    DEFAULT_PHASE1_EPOCHS,
    DEFAULT_PHASE2_EPOCHS,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_FLIP_PROB,
    DEFAULT_ROTATION_DEGREES,
    DEFAULT_CLAHE_CLIP_LIMIT,
    DEFAULT_CLAHE_TILE_SIZE,
    ORIGINAL_IMAGE_SIZE,
    # Quick mode constants
    QUICK_MODE_MAX_TRAIN,
    QUICK_MODE_MAX_VAL,
    QUICK_MODE_MAX_TEST,
    QUICK_MODE_EPOCHS_OPTIMIZE,
    QUICK_MODE_EPOCHS_COMPARE,
    # Classifier classes
    CLASSIFIER_CLASSES,
)


# Configurar logging inicial
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def verbose_callback(ctx: typer.Context, verbose: bool):
    """Callback para configurar nivel de logging."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("src_v2").setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")


app = typer.Typer(
    name="src_v2",
    help="COVID-19 Detection via Anatomical Landmarks - CLI",
    add_completion=False,
)


@app.callback()
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output (DEBUG level logging)",
        is_eager=True,
        callback=verbose_callback,
    ),
):
    """
    COVID-19 Detection via Anatomical Landmarks.

    Use --verbose/-v for detailed output.
    """
    pass


def get_device(device: str) -> "torch.device":
    """Obtener dispositivo de PyTorch."""
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


def get_optimal_num_workers() -> int:
    """
    Obtener numero optimo de workers para DataLoader segun el sistema operativo.

    Session 33: Bug B1 fix - num_workers dinamico.
    Sandbox fix: fallback a 0 si el entorno no permite semaforos/colas.

    - Windows: 0 (problemas con multiprocessing en subprocesos)
    - Linux/Mac: min(4, cpu_count) para multiprocessing seguro, salvo restricciones
      de semaforos; en ese caso usar 0.

    Returns:
        int: Numero de workers recomendado
    """
    import os
    import platform
    import multiprocessing as mp

    # Windows tiene problemas con multiprocessing en DataLoader dentro de CLI
    if platform.system() == "Windows" or sys.platform == "win32":
        return 0

    # Permitir override via variable de entorno (debug/sandbox)
    if os.environ.get("FORCE_NUM_WORKERS_ZERO"):
        logger.warning("FORCE_NUM_WORKERS_ZERO=1 -> usando num_workers=0")
        return 0
    if os.environ.get("NUM_WORKERS_OVERRIDE"):
        try:
            return max(0, int(os.environ["NUM_WORKERS_OVERRIDE"]))
        except ValueError:
            logger.warning("NUM_WORKERS_OVERRIDE invalido, ignorando")

    # Linux y macOS pueden usar multiprocessing
    cpu_count = os.cpu_count() or 4
    candidate = min(4, cpu_count)

    # Verificar que se puedan crear semaforos/locks (algunos sandboxes no lo permiten)
    try:
        lock = mp.get_context("fork").Lock()
        lock.acquire()
        lock.release()
        return candidate
    except Exception as exc:
        logger.warning("Multiprocessing no disponible (%s); usando num_workers=0", exc)
        return 0


def detect_architecture_from_checkpoint(state_dict: dict) -> dict:
    """
    Detecta la arquitectura del modelo a partir del state_dict.

    Args:
        state_dict: Diccionario con los pesos del modelo

    Returns:
        dict con los parametros de arquitectura detectados:
        - use_coord_attention: bool
        - deep_head: bool
        - hidden_dim: int
    """
    # Detectar coord_attention
    use_coord_attention = any("coord_attention" in k for k in state_dict.keys())

    # Detectar deep_head (tiene indices 6, 9 en head)
    deep_head = "head.9.weight" in state_dict or "head.9.bias" in state_dict

    # Detectar hidden_dim
    if deep_head:
        # En deep_head, head.5 es Linear(512, hidden_dim)
        # hidden_dim es la primera dimension de head.5.weight
        if "head.5.weight" in state_dict:
            hidden_dim = state_dict["head.5.weight"].shape[0]
        else:
            hidden_dim = 256  # fallback
    else:
        # En head simple, head.2 es Linear(feature_dim, hidden_dim)
        # hidden_dim es la primera dimension de head.2.weight
        if "head.2.weight" in state_dict:
            weight = state_dict["head.2.weight"]
            # Puede ser Linear o BatchNorm, verificar shape
            if len(weight.shape) == 2:
                hidden_dim = weight.shape[0]
            else:
                hidden_dim = 256  # fallback
        else:
            hidden_dim = 256  # fallback

    return {
        "use_coord_attention": use_coord_attention,
        "deep_head": deep_head,
        "hidden_dim": hidden_dim,
    }


@app.command()
def train(
    data_root: Optional[str] = typer.Option(
        None,
        "--data-root",
        "-d",
        help="Directorio raiz de datos"
    ),
    csv_path: Optional[str] = typer.Option(
        None,
        "--csv-path",
        help="Path al CSV de coordenadas"
    ),
    checkpoint_dir: str = typer.Option(
        "checkpoints",
        "--checkpoint-dir",
        help="Directorio para guardar checkpoints"
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo: auto, cuda, cpu, mps"
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Seed para reproducibilidad"
    ),
    phase1_epochs: int = typer.Option(
        15,
        "--phase1-epochs",
        help="Epocas para fase 1 (backbone congelado)"
    ),
    phase2_epochs: int = typer.Option(
        100,
        "--phase2-epochs",
        help="Epocas para fase 2 (fine-tuning)"
    ),
    batch_size: int = typer.Option(
        16,
        "--batch-size",
        "-b",
        help="Tamano de batch"
    ),
    coord_attention: bool = typer.Option(
        True,
        "--coord-attention/--no-coord-attention",
        help="Usar Coordinate Attention module"
    ),
    deep_head: bool = typer.Option(
        True,
        "--deep-head/--no-deep-head",
        help="Usar cabeza profunda con GroupNorm"
    ),
    hidden_dim: int = typer.Option(
        768,
        "--hidden-dim",
        help="Dimension de capa oculta"
    ),
    dropout: float = typer.Option(
        0.3,
        "--dropout",
        help="Tasa de dropout"
    ),
    # Loss function
    loss_type: str = typer.Option(
        "wing",
        "--loss",
        help="Tipo de loss: wing, weighted_wing, combined"
    ),
    # CLAHE preprocessing
    use_clahe: bool = typer.Option(
        True,
        "--clahe/--no-clahe",
        help="Usar CLAHE para mejora de contraste"
    ),
    clahe_clip: float = typer.Option(
        DEFAULT_CLAHE_CLIP_LIMIT,
        "--clahe-clip",
        help="CLAHE clip limit"
    ),
    clahe_tile: int = typer.Option(
        DEFAULT_CLAHE_TILE_SIZE,
        "--clahe-tile",
        help="CLAHE tile size"
    ),
    # Learning rates
    phase1_lr: float = typer.Option(
        1e-3,
        "--phase1-lr",
        help="Learning rate para fase 1"
    ),
    phase2_backbone_lr: float = typer.Option(
        2e-5,
        "--phase2-backbone-lr",
        help="Learning rate del backbone en fase 2"
    ),
    phase2_head_lr: float = typer.Option(
        2e-4,
        "--phase2-head-lr",
        help="Learning rate de la cabeza en fase 2"
    ),
    # Early stopping
    phase1_patience: int = typer.Option(
        5,
        "--phase1-patience",
        help="Paciencia early stopping fase 1"
    ),
    phase2_patience: int = typer.Option(
        10,
        "--phase2-patience",
        help="Paciencia early stopping fase 2"
    ),
):
    """
    Entrenar modelo de prediccion de landmarks.

    Entrenamiento en dos fases:
    - Fase 1: Backbone congelado, entrenar solo cabeza
    - Fase 2: Fine-tuning completo con LR diferenciado

    Valores por defecto reproducen el mejor modelo (3.71 px ensemble):
    - Loss: WingLoss
    - CLAHE: enabled (clip=2.0, tile=4)
    - Arquitectura: CoordAttention + DeepHead (hidden_dim=768)
    """
    import torch
    from torch.utils.data import DataLoader, random_split

    from src_v2.data import LandmarkDataset, get_train_transforms, get_val_transforms
    from src_v2.data.utils import load_coordinates_csv
    from src_v2.models import create_model, CombinedLandmarkLoss
    from src_v2.training.trainer import LandmarkTrainer
    from sklearn.model_selection import train_test_split

    def safe_train_test_split(dataframe, test_size, seed, desc, stratify_col=None):
        """Stratified split with graceful fallback when class counts are too small."""
        stratify_vals = dataframe[stratify_col] if stratify_col else None
        try:
            return train_test_split(
                dataframe,
                test_size=test_size,
                random_state=seed,
                stratify=stratify_vals,
            )
        except ValueError as exc:
            logger.warning("%s: fallback to non-stratified split (%s)", desc, exc)
            return train_test_split(
                dataframe,
                test_size=test_size,
                random_state=seed,
                stratify=None,
            )

    logger.info("=" * 60)
    logger.info("COVID-19 Landmark Detection - Training")
    logger.info("=" * 60)

    # Configurar seed completo para reproducibilidad
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Dispositivo
    torch_device = get_device(device)
    logger.info("Usando dispositivo: %s", torch_device)

    # Valores por defecto si no se especificaron
    if data_root is None:
        data_root = "data/"
    if csv_path is None:
        csv_path = "data/coordenadas/coordenadas_maestro.csv"

    # Verificar paths
    if not Path(data_root).exists():
        logger.error("Directorio de datos no existe: %s", data_root)
        raise typer.Exit(code=1)

    if not Path(csv_path).exists():
        logger.error("CSV de coordenadas no existe: %s", csv_path)
        raise typer.Exit(code=1)

    # Cargar CSV de coordenadas
    logger.info("Cargando dataset desde %s", data_root)
    df = load_coordinates_csv(csv_path)
    logger.info("CSV cargado: %d muestras", len(df))

    # Split train/val/test (75/15/10)
    # IMPORTANTE: Siempre usar random_state=42 para el split de datos
    # El seed del modelo solo afecta inicialización, no el split
    SPLIT_SEED = 42  # Fijo para reproducibilidad

    train_df, temp_df = safe_train_test_split(
        df,
        test_size=0.25,
        seed=SPLIT_SEED,
        desc="Train/temp split",
        stratify_col='category',
    )
    val_df, test_df = safe_train_test_split(
        temp_df,
        test_size=0.4,
        seed=SPLIT_SEED,
        desc="Val/test split",
        stratify_col='category',
    )

    logger.info("Dataset splits: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))

    # Crear datasets con transforms (con CLAHE si está habilitado)
    logger.info("Transforms: CLAHE=%s (clip=%.1f, tile=%d)", use_clahe, clahe_clip, clahe_tile)
    train_transform = get_train_transforms(
        output_size=DEFAULT_IMAGE_SIZE,
        use_clahe=use_clahe,
        clahe_clip_limit=clahe_clip,
        clahe_tile_size=clahe_tile
    )
    val_transform = get_val_transforms(
        output_size=DEFAULT_IMAGE_SIZE,
        use_clahe=use_clahe,
        clahe_clip_limit=clahe_clip,
        clahe_tile_size=clahe_tile
    )

    train_set = LandmarkDataset(train_df, data_root, transform=train_transform)
    val_set = LandmarkDataset(val_df, data_root, transform=val_transform)

    # DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=get_optimal_num_workers(),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=get_optimal_num_workers(),
        pin_memory=True
    )

    # Crear modelo
    logger.info("Creando modelo ResNet18Landmarks")
    logger.info("  coord_attention=%s, deep_head=%s, hidden_dim=%d, dropout=%.2f",
                coord_attention, deep_head, hidden_dim, dropout)
    model = create_model(
        pretrained=True,
        freeze_backbone=True,
        dropout_rate=dropout,
        hidden_dim=hidden_dim,
        use_coord_attention=coord_attention,
        deep_head=deep_head,
    )
    model = model.to(torch_device)

    # Loss function
    from src_v2.models.losses import WingLoss, WeightedWingLoss

    logger.info("Loss function: %s", loss_type)
    if loss_type == "wing":
        criterion = WingLoss(
            omega=DEFAULT_WING_OMEGA,
            epsilon=DEFAULT_WING_EPSILON,
            normalized=True,
            image_size=DEFAULT_IMAGE_SIZE
        )
    elif loss_type == "weighted_wing":
        from src_v2.models.losses import get_landmark_weights
        weights = get_landmark_weights("uniform")
        criterion = WeightedWingLoss(
            omega=DEFAULT_WING_OMEGA,
            epsilon=DEFAULT_WING_EPSILON,
            weights=weights.to(torch_device),
            normalized=True,
            image_size=DEFAULT_IMAGE_SIZE
        )
    elif loss_type == "combined":
        criterion = CombinedLandmarkLoss(
            image_size=DEFAULT_IMAGE_SIZE,
            central_weight=1.0,
            symmetry_weight=0.5,
            symmetry_margin=DEFAULT_SYMMETRY_MARGIN
        )
    else:
        logger.error("Loss desconocida: '%s'", loss_type)
        logger.info("Hint: Opciones válidas: wing, weighted_wing, combined")
        logger.info("      - wing: Wing Loss básica para landmarks")
        logger.info("      - weighted_wing: Wing Loss con pesos por landmark")
        logger.info("      - combined: Wing + simetría + landmarks centrales")
        raise typer.Exit(code=1)
    criterion = criterion.to(torch_device)

    # Trainer
    trainer = LandmarkTrainer(
        model=model,
        device=torch_device,
        save_dir=checkpoint_dir,
        image_size=DEFAULT_IMAGE_SIZE
    )

    # Entrenar
    logger.info("Iniciando entrenamiento...")
    logger.info("  Phase 1: epochs=%d, lr=%.1e, patience=%d", phase1_epochs, phase1_lr, phase1_patience)
    logger.info("  Phase 2: epochs=%d, backbone_lr=%.1e, head_lr=%.1e, patience=%d",
                phase2_epochs, phase2_backbone_lr, phase2_head_lr, phase2_patience)
    history = trainer.train_full(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        phase1_epochs=phase1_epochs,
        phase2_epochs=phase2_epochs,
        phase1_lr=phase1_lr,
        phase2_backbone_lr=phase2_backbone_lr,
        phase2_head_lr=phase2_head_lr,
        phase1_patience=phase1_patience,
        phase2_patience=phase2_patience,
    )

    # Guardar modelo final
    final_path = Path(checkpoint_dir) / "final_model.pt"
    trainer.save_model(str(final_path))
    logger.info("Modelo guardado en: %s", final_path)
    last_path = Path(checkpoint_dir) / "last.pt"
    if last_path != final_path:
        trainer.save_model(str(last_path))
        logger.info("Modelo guardado en: %s", last_path)

    logger.info("=" * 60)
    logger.info("Entrenamiento completado!")
    logger.info("=" * 60)


@app.command()
def evaluate(
    checkpoint: str = typer.Argument(
        ...,
        help="Path al checkpoint del modelo"
    ),
    data_root: str = typer.Option(
        "data/",
        "--data-root",
        "-d",
        help="Directorio raiz de datos"
    ),
    csv_path: str = typer.Option(
        "data/coordenadas/coordenadas_maestro.csv",
        "--csv-path",
        help="Path al CSV de coordenadas"
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo: auto, cuda, cpu, mps"
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Tamano de batch"
    ),
    output_json: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Guardar resultados en JSON"
    ),
    use_tta: bool = typer.Option(
        False,
        "--tta",
        help="Usar Test-Time Augmentation"
    ),
    split: str = typer.Option(
        "test",
        "--split",
        "-s",
        help="Split a evaluar: test, val, train, all"
    ),
    # CLAHE preprocessing (debe coincidir con entrenamiento)
    use_clahe: bool = typer.Option(
        True,
        "--clahe/--no-clahe",
        help="Usar CLAHE para mejora de contraste (debe coincidir con entrenamiento)"
    ),
    clahe_clip: float = typer.Option(
        DEFAULT_CLAHE_CLIP_LIMIT,
        "--clahe-clip",
        help="CLAHE clip limit"
    ),
    clahe_tile: int = typer.Option(
        DEFAULT_CLAHE_TILE_SIZE,
        "--clahe-tile",
        help="CLAHE tile size"
    ),
):
    """
    Evaluar modelo en dataset de test.

    Calcula metricas de error por landmark y por categoria.
    Por defecto evalua solo el split de test (10% de datos, seed=42).
    """
    import json

    import torch
    from torch.utils.data import DataLoader

    from src_v2.data import LandmarkDataset, get_val_transforms
    from src_v2.data.utils import load_coordinates_csv
    from src_v2.models import create_model
    from src_v2.evaluation.metrics import evaluate_model, evaluate_model_with_tta

    logger.info("=" * 60)
    logger.info("COVID-19 Landmark Detection - Evaluation")
    logger.info("=" * 60)

    # Verificar checkpoint
    if not Path(checkpoint).exists():
        logger.error("Checkpoint no existe: %s", checkpoint)
        raise typer.Exit(code=1)

    # Dispositivo
    torch_device = get_device(device)
    logger.info("Usando dispositivo: %s", torch_device)

    # Cargar checkpoint y detectar arquitectura
    logger.info("Cargando modelo desde: %s", checkpoint)
    checkpoint_data = torch.load(checkpoint, map_location=torch_device, weights_only=False)

    if "model_state_dict" in checkpoint_data:
        state_dict = checkpoint_data["model_state_dict"]
    else:
        state_dict = checkpoint_data

    # Detectar arquitectura automaticamente del checkpoint
    arch_params = detect_architecture_from_checkpoint(state_dict)
    logger.info("Arquitectura detectada: coord_attention=%s, deep_head=%s, hidden_dim=%d",
                arch_params["use_coord_attention"], arch_params["deep_head"], arch_params["hidden_dim"])

    model = create_model(
        pretrained=False,
        use_coord_attention=arch_params["use_coord_attention"],
        deep_head=arch_params["deep_head"],
        hidden_dim=arch_params["hidden_dim"],
    )
    model.load_state_dict(state_dict)
    model = model.to(torch_device)
    model.eval()

    # Crear dataset
    logger.info("Cargando dataset desde: %s", data_root)
    df = load_coordinates_csv(csv_path)
    logger.info("CSV cargado: %d muestras totales", len(df))

    # Filtrar por split si es necesario
    if split != "all":
        from sklearn.model_selection import train_test_split
        SPLIT_SEED = 42  # Mismo seed que en training

        def safe_split(dataframe, test_size, desc, stratify_col='category'):
            if len(dataframe) < 2 or int(len(dataframe) * (1 - test_size)) < 1:
                logger.warning("%s: dataset demasiado pequeño (%d); usando todo en el primer split", desc, len(dataframe))
                empty = dataframe.iloc[0:0]
                return dataframe, empty
            try:
                return train_test_split(
                    dataframe,
                    test_size=test_size,
                    random_state=SPLIT_SEED,
                    stratify=dataframe[stratify_col],
                )
            except ValueError as exc:
                logger.warning("%s: fallback a split sin estratificar (%s)", desc, exc)
                return train_test_split(
                    dataframe,
                    test_size=test_size,
                    random_state=SPLIT_SEED,
                    stratify=None,
                )

        train_df, temp_df = safe_split(df, 0.25, "Split train/temp")
        val_df, test_df = safe_split(temp_df, 0.4, "Split val/test")

        if split == "test":
            df = test_df
        elif split == "val":
            df = val_df
        elif split == "train":
            df = train_df
        else:
            logger.error("Split invalido: %s (usar: test, val, train, all)", split)
            raise typer.Exit(code=1)

    if len(df) == 0:
        logger.error("Split '%s' sin muestras; no se puede evaluar", split)
        raise typer.Exit(code=1)

    logger.info("Evaluando split '%s': %d muestras", split, len(df))
    logger.info("Transforms: CLAHE=%s (clip=%.1f, tile=%d)", use_clahe, clahe_clip, clahe_tile)

    test_transform = get_val_transforms(
        output_size=DEFAULT_IMAGE_SIZE,
        use_clahe=use_clahe,
        clahe_clip_limit=clahe_clip,
        clahe_tile_size=clahe_tile
    )
    test_dataset = LandmarkDataset(df, data_root, transform=test_transform)

    # Custom collate para incluir metadata
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        landmarks = torch.stack([item[1] for item in batch])
        metas = [item[2] for item in batch]
        return images, landmarks, metas

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=get_optimal_num_workers(),
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Evaluar
    logger.info("Evaluando modelo...")
    if use_tta:
        logger.info("Usando Test-Time Augmentation")
        results = evaluate_model_with_tta(
            model, test_loader, torch_device, image_size=ORIGINAL_IMAGE_SIZE
        )
    else:
        results = evaluate_model(
            model, test_loader, torch_device, image_size=ORIGINAL_IMAGE_SIZE
        )

    # Mostrar resultados
    logger.info("-" * 40)
    logger.info("RESULTADOS")
    logger.info("-" * 40)
    overall = results["overall"]
    logger.info("Error promedio: %.2f px", overall["mean"])
    logger.info("Error mediana: %.2f px", overall["median"])
    logger.info("Error std: %.2f px", overall["std"])

    if "per_landmark" in results:
        logger.info("\nError por landmark:")
        for name, stats in results["per_landmark"].items():
            logger.info("  %s: %.2f px (std=%.2f)", name, stats["mean"], stats["std"])

    if "per_category" in results:
        logger.info("\nError por categoria:")
        for cat, stats in results["per_category"].items():
            logger.info("  %s: %.2f px (n=%d)", cat, stats["mean"], stats["count"])

    # Guardar JSON si se especifica
    if output_json:
        # Convertir numpy/torch types a Python types para JSON (maneja dicts anidados)
        def to_serializable(obj):
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [to_serializable(v) for v in obj]
            # torch.Tensor / numpy.ndarray (may have more than 1 element)
            if hasattr(obj, "tolist"):
                try:
                    # Scalar tensor/array
                    if getattr(obj, "numel", lambda: None)() == 1 or getattr(obj, "size", lambda: None) == 1:
                        return float(obj.item())
                except Exception:
                    pass
                try:
                    return obj.tolist()
                except Exception:
                    return str(obj)
            if hasattr(obj, "item"):
                try:
                    return obj.item()
                except Exception:
                    pass
            try:
                return float(obj)
            except Exception:
                return str(obj)

        results_json = to_serializable(results)
        with open(output_json, "w") as f:
            json.dump(results_json, f, indent=2)
        logger.info("Resultados guardados en: %s", output_json)

    logger.info("=" * 60)
    logger.info("Evaluacion completada!")
    logger.info("=" * 60)


@app.command()
def predict(
    image: str = typer.Argument(
        ...,
        help="Path a la imagen de rayos X"
    ),
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint",
        "-c",
        help="Path al checkpoint del modelo"
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo: auto, cuda, cpu, mps"
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Guardar imagen con landmarks visualizados"
    ),
    output_json: Optional[str] = typer.Option(
        None,
        "--json",
        "-j",
        "--output-json",
        help="Guardar coordenadas en JSON"
    ),
    # CLAHE preprocessing (debe coincidir con entrenamiento)
    use_clahe: bool = typer.Option(
        True,
        "--clahe/--no-clahe",
        help="Usar CLAHE para mejora de contraste (debe coincidir con entrenamiento)"
    ),
    clahe_clip: float = typer.Option(
        DEFAULT_CLAHE_CLIP_LIMIT,
        "--clahe-clip",
        help="CLAHE clip limit"
    ),
    clahe_tile: int = typer.Option(
        DEFAULT_CLAHE_TILE_SIZE,
        "--clahe-tile",
        help="CLAHE tile size"
    ),
):
    """
    Predecir landmarks en una imagen de rayos X.

    Muestra coordenadas predichas y opcionalmente guarda visualizacion.
    IMPORTANTE: Usar --clahe si el modelo fue entrenado con CLAHE (default).
    """
    import json

    import cv2
    import numpy as np
    import torch
    from PIL import Image

    from src_v2.constants import LANDMARK_NAMES, IMAGENET_MEAN, IMAGENET_STD
    from src_v2.models import create_model

    logger.info("=" * 60)
    logger.info("COVID-19 Landmark Detection - Predict")
    logger.info("=" * 60)

    # Verificar archivos
    if not Path(image).exists():
        logger.error("Imagen no existe: %s", image)
        logger.info("Hint: Verifica que la ruta sea correcta y la imagen exista")
        raise typer.Exit(code=1)

    if not Path(checkpoint).exists():
        logger.error("Checkpoint no existe: %s", checkpoint)
        logger.info("Hint: Usa 'python -m src_v2 train' para entrenar un modelo")
        raise typer.Exit(code=1)

    # Dispositivo
    torch_device = get_device(device)
    logger.info("Usando dispositivo: %s", torch_device)

    # Cargar checkpoint y detectar arquitectura
    logger.info("Cargando modelo desde: %s", checkpoint)
    checkpoint_data = torch.load(checkpoint, map_location=torch_device, weights_only=False)

    if "model_state_dict" in checkpoint_data:
        state_dict = checkpoint_data["model_state_dict"]
    else:
        state_dict = checkpoint_data

    # Detectar arquitectura automaticamente del checkpoint
    arch_params = detect_architecture_from_checkpoint(state_dict)
    logger.info("Arquitectura detectada: coord_attention=%s, deep_head=%s, hidden_dim=%d",
                arch_params["use_coord_attention"], arch_params["deep_head"], arch_params["hidden_dim"])

    model = create_model(
        pretrained=False,
        use_coord_attention=arch_params["use_coord_attention"],
        deep_head=arch_params["deep_head"],
        hidden_dim=arch_params["hidden_dim"],
    )
    model.load_state_dict(state_dict)
    model = model.to(torch_device)
    model.eval()

    # Cargar y preprocesar imagen
    logger.info("Procesando imagen: %s", image)
    logger.info("Preprocessing: CLAHE=%s (clip=%.1f, tile=%d)", use_clahe, clahe_clip, clahe_tile)

    img = Image.open(image).convert("RGB")
    original_size = img.size

    # Resize a 224x224
    img_resized = img.resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), Image.BILINEAR)
    img_array = np.array(img_resized, dtype=np.uint8)

    # Aplicar CLAHE si está habilitado (antes de normalizar)
    if use_clahe:
        # Convertir a grayscale para CLAHE
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
        img_clahe = clahe.apply(img_gray)
        # Convertir de vuelta a RGB
        img_array = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)

    # Normalizar
    img_float = img_array.astype(np.float32) / 255.0

    # Normalizar con ImageNet stats
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img_normalized = (img_float - mean) / std

    # Convertir a tensor
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
    img_tensor = img_tensor.to(torch_device)

    # Predecir
    with torch.no_grad():
        predictions = model(img_tensor)

    # Convertir a coordenadas en pixeles
    landmarks_norm = predictions.squeeze().cpu().numpy().reshape(NUM_LANDMARKS, 2)

    size_array = np.array(original_size, dtype=np.float32)
    landmarks_original = landmarks_norm * size_array
    landmarks_original[:, 0] = np.clip(landmarks_original[:, 0], 0, size_array[0] - 1)
    landmarks_original[:, 1] = np.clip(landmarks_original[:, 1], 0, size_array[1] - 1)

    landmarks_resized = landmarks_norm * DEFAULT_IMAGE_SIZE
    landmarks_resized = np.clip(landmarks_resized, 0, DEFAULT_IMAGE_SIZE - 1)

    # Mostrar resultados
    logger.info("-" * 40)
    logger.info(
        "LANDMARKS PREDICHOS (pixeles en imagen original %dx%d):",
        original_size[0],
        original_size[1],
    )
    logger.info("-" * 40)

    landmark_dict = {}
    for i, name in enumerate(LANDMARK_NAMES):
        x, y = landmarks_original[i]
        logger.info("  %s: (%.1f, %.1f)", name, x, y)
        landmark_dict[name] = {"x": float(x), "y": float(y)}

    landmark_model_space = {
        name: {"x": float(x), "y": float(y)}
        for name, (x, y) in zip(LANDMARK_NAMES, landmarks_resized)
    }

    # Guardar JSON si se especifica
    if output_json:
        result = {
            "image": str(image),
            "model_input_size": DEFAULT_IMAGE_SIZE,
            "original_size": {"width": original_size[0], "height": original_size[1]},
            "landmarks": landmark_dict,  # pixeles en tamaño original
            "landmarks_model_space": landmark_model_space,  # pixeles en imagen 224x224
            "landmarks_normalized": [
                {"x": float(x), "y": float(y)} for x, y in landmarks_norm
            ],
        }
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Coordenadas guardadas en: %s", output_json)

    # Guardar visualizacion si se especifica
    if output:
        img_vis = np.array(img_resized)

        # Dibujar landmarks
        for i, (x, y) in enumerate(landmarks_resized):
            x, y = int(x), int(y)
            cv2.circle(img_vis, (x, y), 3, (255, 0, 0), -1)
            cv2.putText(
                img_vis,
                f"L{i+1}",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 0, 0),
                1
            )

        cv2.imwrite(output, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
        logger.info("Visualizacion guardada en: %s", output)

    logger.info("=" * 60)
    logger.info("Prediccion completada!")
    logger.info("=" * 60)


@app.command()
def warp(
    input_dir: str = typer.Argument(
        ...,
        help="Directorio con imagenes de entrada"
    ),
    output_dir: str = typer.Argument(
        ...,
        help="Directorio de salida para imagenes warpeadas"
    ),
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint",
        "-c",
        help="Path al checkpoint del modelo para prediccion de landmarks"
    ),
    canonical_shape: str = typer.Option(
        "outputs/shape_analysis/canonical_shape_gpa.json",
        "--canonical",
        help="Path a la forma canonica (.json o .npy)"
    ),
    triangles: str = typer.Option(
        "outputs/shape_analysis/canonical_delaunay_triangles.json",
        "--triangles",
        help="Path a los triangulos de Delaunay (.json o .npy)"
    ),
    margin_scale: float = typer.Option(
        1.05,
        "--margin-scale",
        "-m",
        help="Factor de escala para margenes"
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo: auto, cuda, cpu, mps"
    ),
    pattern: str = typer.Option(
        "**/*.png",
        "--pattern",
        "-p",
        help="Patron glob para buscar imagenes"
    ),
    # CLAHE preprocessing (debe coincidir con entrenamiento)
    use_clahe: bool = typer.Option(
        True,
        "--clahe/--no-clahe",
        help="Usar CLAHE para mejora de contraste (debe coincidir con entrenamiento)"
    ),
    clahe_clip: float = typer.Option(
        DEFAULT_CLAHE_CLIP_LIMIT,
        "--clahe-clip",
        help="CLAHE clip limit"
    ),
    clahe_tile: int = typer.Option(
        DEFAULT_CLAHE_TILE_SIZE,
        "--clahe-tile",
        help="CLAHE tile size"
    ),
):
    """
    Aplicar warping geometrico a un dataset de imagenes.

    Usa el modelo de landmarks para predecir puntos anatomicos
    y aplica transformacion piecewise affine a forma canonica.
    IMPORTANTE: Usar --clahe si el modelo fue entrenado con CLAHE (default).
    """
    import sys

    import cv2
    import numpy as np
    import torch
    from tqdm import tqdm

    from src_v2.constants import IMAGENET_MEAN, IMAGENET_STD
    from src_v2.models import create_model

    logger.info("=" * 60)
    logger.info("COVID-19 Landmark Detection - Warp Dataset")
    logger.info("=" * 60)

    # Verificar paths
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error("Directorio de entrada no existe: %s", input_dir)
        logger.info("Hint: El directorio debe contener subdirectorios por categoría (COVID, Normal, etc.)")
        raise typer.Exit(code=1)

    if not Path(checkpoint).exists():
        logger.error("Checkpoint no existe: %s", checkpoint)
        logger.info("Hint: Usa 'python -m src_v2 train' para entrenar un modelo de landmarks")
        raise typer.Exit(code=1)

    # Cargar forma canonica y triangulos
    if not Path(canonical_shape).exists():
        logger.error("Forma canonica no existe: %s", canonical_shape)
        logger.info("Hint: Genera la forma canónica con 'python -m src_v2 gpa'")
        raise typer.Exit(code=1)

    if not Path(triangles).exists():
        logger.error("Triangulos de Delaunay no existen: %s", triangles)
        logger.info("Hint: Los triángulos se generan junto con la forma canónica")
        raise typer.Exit(code=1)

    # Soportar formato JSON y NPY
    if canonical_shape.endswith('.json'):
        import json
        with open(canonical_shape, 'r') as f:
            data = json.load(f)
        canonical = np.array(data['canonical_shape_pixels'])
    else:
        canonical = np.load(canonical_shape)

    if triangles.endswith('.json'):
        import json
        with open(triangles, 'r') as f:
            data = json.load(f)
        tri = np.array(data['triangles'])
    else:
        tri = np.load(triangles)

    logger.info("Forma canonica cargada: %s", canonical.shape)
    logger.info("Triangulos Delaunay cargados: %s", tri.shape)

    # Crear directorio de salida
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Dispositivo
    torch_device = get_device(device)
    logger.info("Usando dispositivo: %s", torch_device)

    # Cargar checkpoint y detectar arquitectura
    logger.info("Cargando modelo desde: %s", checkpoint)
    checkpoint_data = torch.load(checkpoint, map_location=torch_device, weights_only=False)

    if "model_state_dict" in checkpoint_data:
        state_dict = checkpoint_data["model_state_dict"]
    else:
        state_dict = checkpoint_data

    # Detectar arquitectura automaticamente del checkpoint
    arch_params = detect_architecture_from_checkpoint(state_dict)
    logger.info("Arquitectura detectada: coord_attention=%s, deep_head=%s, hidden_dim=%d",
                arch_params["use_coord_attention"], arch_params["deep_head"], arch_params["hidden_dim"])

    model = create_model(
        pretrained=False,
        use_coord_attention=arch_params["use_coord_attention"],
        deep_head=arch_params["deep_head"],
        hidden_dim=arch_params["hidden_dim"],
    )
    model.load_state_dict(state_dict)
    model = model.to(torch_device)
    model.eval()

    # Importar funcion de warp desde el modulo correcto
    from src_v2.processing.warp import piecewise_affine_warp

    # Buscar imagenes
    images = list(input_path.glob(pattern))
    logger.info("Encontradas %d imagenes con patron '%s'", len(images), pattern)

    if len(images) == 0:
        logger.warning("No se encontraron imagenes")
        raise typer.Exit(code=0)

    # Procesar imagenes
    success = 0
    failed = 0

    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    # Crear CLAHE una vez si está habilitado
    clahe_processor = None
    if use_clahe:
        clahe_processor = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
        logger.info("Preprocessing: CLAHE enabled (clip=%.1f, tile=%d)", clahe_clip, clahe_tile)

    for img_path in tqdm(images, desc="Warping"):
        try:
            # Cargar imagen
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                failed += 1
                continue

            # Resize a 224x224
            if img.shape[0] != DEFAULT_IMAGE_SIZE or img.shape[1] != DEFAULT_IMAGE_SIZE:
                img = cv2.resize(img, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))

            # Aplicar CLAHE si está habilitado
            img_for_model = img
            if clahe_processor is not None:
                img_for_model = clahe_processor.apply(img)

            # Preprocesar para modelo (convertir a RGB y normalizar)
            img_rgb = cv2.cvtColor(img_for_model, cv2.COLOR_GRAY2RGB)
            img_float = img_rgb.astype(np.float32) / 255.0
            img_normalized = (img_float - mean) / std

            # Convertir a tensor
            img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
            img_tensor = img_tensor.to(torch_device)

            # Predecir landmarks
            with torch.no_grad():
                predictions = model(img_tensor)

            landmarks = predictions.squeeze().cpu().numpy()
            landmarks = landmarks.reshape(NUM_LANDMARKS, 2) * DEFAULT_IMAGE_SIZE

            # Escalar landmarks desde centroide
            centroid = landmarks.mean(axis=0)
            landmarks_scaled = centroid + (landmarks - centroid) * margin_scale

            # Clip a limites de imagen
            landmarks_clipped = np.clip(landmarks_scaled, 2, DEFAULT_IMAGE_SIZE - 3)

            # Escalar canonical shape
            canonical_scaled = centroid + (canonical - canonical.mean(axis=0)) * margin_scale
            canonical_clipped = np.clip(canonical_scaled, 2, DEFAULT_IMAGE_SIZE - 3)

            # Aplicar warp
            warped = piecewise_affine_warp(
                img,
                landmarks_clipped.astype(np.float32),
                canonical_clipped.astype(np.float32),
                tri
            )

            # Guardar
            relative_path = img_path.relative_to(input_path)
            out_file = output_path / relative_path
            out_file.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(out_file), warped)
            success += 1

        except Exception as e:
            logger.warning("Error procesando %s: %s", img_path, e)
            failed += 1

    logger.info("-" * 40)
    logger.info("RESUMEN")
    logger.info("-" * 40)
    logger.info("Imagenes procesadas: %d", success)
    logger.info("Imagenes fallidas: %d", failed)
    logger.info("Directorio de salida: %s", output_dir)

    logger.info("=" * 60)
    logger.info("Warping completado!")
    logger.info("=" * 60)


@app.command()
def version():
    """Mostrar version del paquete."""
    from src_v2 import __version__
    typer.echo(f"COVID-19 Landmark Detection v{__version__}")


@app.command("evaluate-ensemble")
def evaluate_ensemble(
    checkpoints: list[str] = typer.Argument(
        ...,
        help="Paths a los checkpoints de modelos (minimo 2)"
    ),
    data_root: str = typer.Option(
        "data/",
        "--data-root",
        "-d",
        help="Directorio raiz de datos"
    ),
    csv_path: str = typer.Option(
        "data/coordenadas/coordenadas_maestro.csv",
        "--csv-path",
        help="Path al CSV de coordenadas"
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo: auto, cuda, cpu, mps"
    ),
    batch_size: int = typer.Option(
        16,
        "--batch-size",
        "-b",
        help="Tamano de batch"
    ),
    output_json: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Guardar resultados en JSON"
    ),
    use_tta: bool = typer.Option(
        True,
        "--tta/--no-tta",
        help="Usar Test-Time Augmentation (habilitado por defecto)"
    ),
    split: str = typer.Option(
        "test",
        "--split",
        "-s",
        help="Split a evaluar: test, val, train, all"
    ),
    # CLAHE preprocessing
    use_clahe: bool = typer.Option(
        True,
        "--clahe/--no-clahe",
        help="Usar CLAHE para mejora de contraste"
    ),
    clahe_clip: float = typer.Option(
        DEFAULT_CLAHE_CLIP_LIMIT,
        "--clahe-clip",
        help="CLAHE clip limit"
    ),
    clahe_tile: int = typer.Option(
        DEFAULT_CLAHE_TILE_SIZE,
        "--clahe-tile",
        help="CLAHE tile size"
    ),
):
    """
    Evaluar ensemble de multiples modelos.

    Promedia las predicciones de multiples modelos para reducir error.
    Por defecto usa TTA (Test-Time Augmentation) en cada modelo.

    Ejemplo para reproducir 3.71 px con 4 modelos:

        python -m src_v2 evaluate-ensemble \\
            checkpoints/session10/ensemble/seed123/final_model.pt \\
            checkpoints/session10/ensemble/seed456/final_model.pt \\
            checkpoints/session13/seed321/final_model.pt \\
            checkpoints/session13/seed789/final_model.pt \\
            --tta --clahe
    """
    import json

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from src_v2.data import LandmarkDataset, get_val_transforms
    from src_v2.data.utils import load_coordinates_csv
    from src_v2.models import create_model
    from src_v2.constants import SYMMETRIC_PAIRS

    logger.info("=" * 60)
    logger.info("COVID-19 Landmark Detection - Ensemble Evaluation")
    logger.info("=" * 60)

    # Validar numero de checkpoints
    if len(checkpoints) < 2:
        logger.error("Se requieren al menos 2 checkpoints para ensemble")
        raise typer.Exit(code=1)

    # Verificar que todos los checkpoints existen
    for ckpt in checkpoints:
        if not Path(ckpt).exists():
            logger.error("Checkpoint no existe: %s", ckpt)
            logger.info("Hint: Verifica las rutas de los checkpoints")
            raise typer.Exit(code=1)

    logger.info("Ensemble con %d modelos:", len(checkpoints))
    for i, ckpt in enumerate(checkpoints, 1):
        logger.info("  [%d] %s", i, ckpt)

    # Dispositivo
    torch_device = get_device(device)
    logger.info("Usando dispositivo: %s", torch_device)

    # Cargar modelos
    models = []
    for ckpt in checkpoints:
        checkpoint_data = torch.load(ckpt, map_location=torch_device, weights_only=False)

        if "model_state_dict" in checkpoint_data:
            state_dict = checkpoint_data["model_state_dict"]
        else:
            state_dict = checkpoint_data

        # Detectar arquitectura
        arch_params = detect_architecture_from_checkpoint(state_dict)

        model = create_model(
            pretrained=False,
            use_coord_attention=arch_params["use_coord_attention"],
            deep_head=arch_params["deep_head"],
            hidden_dim=arch_params["hidden_dim"],
        )
        model.load_state_dict(state_dict)
        model = model.to(torch_device)
        model.eval()
        models.append(model)

    logger.info("Cargados %d modelos", len(models))

    # Cargar dataset
    logger.info("Cargando dataset desde: %s", data_root)
    df = load_coordinates_csv(csv_path)

    # Filtrar por split
    if split != "all":
        from sklearn.model_selection import train_test_split
        SPLIT_SEED = 42
        train_df, temp_df = train_test_split(
            df, test_size=0.25, random_state=SPLIT_SEED, stratify=df['category']
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.4, random_state=SPLIT_SEED, stratify=temp_df['category']
        )

        if split == "test":
            df = test_df
        elif split == "val":
            df = val_df
        elif split == "train":
            df = train_df
        else:
            logger.error("Split invalido: %s", split)
            raise typer.Exit(code=1)

    logger.info("Evaluando split '%s': %d muestras", split, len(df))
    logger.info("Transforms: CLAHE=%s (clip=%.1f, tile=%d)", use_clahe, clahe_clip, clahe_tile)
    logger.info("TTA: %s", "habilitado" if use_tta else "deshabilitado")

    # Crear dataloader
    test_transform = get_val_transforms(
        output_size=DEFAULT_IMAGE_SIZE,
        use_clahe=use_clahe,
        clahe_clip_limit=clahe_clip,
        clahe_tile_size=clahe_tile
    )
    test_dataset = LandmarkDataset(df, data_root, transform=test_transform)

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        landmarks = torch.stack([item[1] for item in batch])
        metas = [item[2] for item in batch]
        return images, landmarks, metas

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=get_optimal_num_workers(),
        pin_memory=True,
        collate_fn=collate_fn
    )

    def predict_with_tta(model, images):
        """Prediccion con Test-Time Augmentation."""
        with torch.no_grad():
            # Original
            pred1 = model(images)

            # Flip horizontal
            images_flip = torch.flip(images, dims=[3])
            pred2 = model(images_flip)

            # Invertir flip en predicciones
            pred2 = pred2.view(-1, NUM_LANDMARKS, 2)
            pred2[:, :, 0] = 1 - pred2[:, :, 0]  # Invertir X

            # Intercambiar pares simetricos
            for left, right in SYMMETRIC_PAIRS:
                pred2[:, [left, right]] = pred2[:, [right, left]]

            pred2 = pred2.view(-1, NUM_LANDMARKS * 2)

            # Promediar
            return (pred1 + pred2) / 2

    # Evaluar ensemble
    all_preds = []
    all_targets = []
    all_categories = []

    logger.info("Evaluando ensemble...")
    for batch in tqdm(test_loader, desc="Evaluating"):
        images = batch[0].to(torch_device)
        targets = batch[1].to(torch_device)
        metadata = batch[2]

        # Extraer categorias
        if isinstance(metadata, list) and len(metadata) > 0:
            if isinstance(metadata[0], dict):
                categories = [m.get('category', 'Unknown') for m in metadata]
            else:
                categories = ['Unknown'] * len(images)
        else:
            categories = ['Unknown'] * len(images)

        # Predicciones de cada modelo
        preds = []
        for model in models:
            if use_tta:
                pred = predict_with_tta(model, images)
            else:
                with torch.no_grad():
                    pred = model(images)
            preds.append(pred)

        # Promediar predicciones (pesos iguales)
        preds_stack = torch.stack(preds)  # (n_models, batch, 30)
        ensemble_pred = preds_stack.mean(dim=0)

        all_preds.append(ensemble_pred.cpu())
        all_targets.append(targets.cpu())
        all_categories.extend(categories)

    # Concatenar resultados
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calcular metricas
    all_preds = all_preds.view(-1, NUM_LANDMARKS, 2)
    all_targets = all_targets.view(-1, NUM_LANDMARKS, 2)

    # Error en pixeles
    errors_px = torch.norm((all_preds - all_targets) * DEFAULT_IMAGE_SIZE, dim=-1)

    # Metricas generales
    results = {
        "overall": {
            "mean": float(errors_px.mean()),
            "std": float(errors_px.std()),
            "median": float(errors_px.median()),
        },
        "percentiles": {
            "p50": float(torch.quantile(errors_px.float(), 0.50)),
            "p75": float(torch.quantile(errors_px.float(), 0.75)),
            "p90": float(torch.quantile(errors_px.float(), 0.90)),
            "p95": float(torch.quantile(errors_px.float(), 0.95)),
        },
        "per_landmark": {},
        "per_category": {},
        "ensemble_info": {
            "n_models": len(checkpoints),
            "checkpoints": checkpoints,
            "tta_enabled": use_tta,
            "clahe_enabled": use_clahe,
        }
    }

    # Error por landmark
    for i in range(NUM_LANDMARKS):
        landmark_name = f"L{i+1}"
        landmark_errors = errors_px[:, i]
        results["per_landmark"][landmark_name] = {
            "mean": float(landmark_errors.mean()),
            "std": float(landmark_errors.std()),
            "median": float(landmark_errors.median()),
            "max": float(landmark_errors.max()),
        }

    # Error por categoria
    errors_flat = errors_px.mean(dim=1)
    for cat in set(all_categories):
        cat_mask = torch.tensor([c == cat for c in all_categories])
        cat_errors = errors_flat[cat_mask]
        results["per_category"][cat] = {
            "mean": float(cat_errors.mean()),
            "std": float(cat_errors.std()),
            "count": int(cat_mask.sum()),
        }

    # Mostrar resultados
    logger.info("-" * 60)
    logger.info("RESULTADOS - ENSEMBLE (%d modelos)", len(checkpoints))
    logger.info("-" * 60)

    overall = results["overall"]
    logger.info("Error promedio: %.2f px", overall["mean"])
    logger.info("Error mediana:  %.2f px", overall["median"])
    logger.info("Error std:      %.2f px", overall["std"])

    logger.info("\nPercentiles:")
    for k, v in results["percentiles"].items():
        logger.info("  %s: %.2f px", k, v)

    logger.info("\nError por landmark:")
    sorted_landmarks = sorted(
        results["per_landmark"].items(),
        key=lambda x: x[1]["mean"]
    )
    for name, data in sorted_landmarks:
        logger.info("  %s: %.2f px (std=%.2f)", name, data["mean"], data["std"])

    logger.info("\nError por categoria:")
    for cat, data in sorted(results["per_category"].items(), key=lambda x: x[1]["mean"]):
        logger.info("  %s: %.2f +/- %.2f px (n=%d)", cat, data["mean"], data["std"], data["count"])

    # Guardar JSON si se especifica
    if output_json:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("\nResultados guardados en: %s", output_json)

    logger.info("=" * 60)
    logger.info("Evaluacion de ensemble completada!")
    logger.info("=" * 60)


@app.command()
def classify(
    image: str = typer.Argument(
        ...,
        help="Path a imagen individual o directorio con imagenes"
    ),
    classifier_checkpoint: str = typer.Option(
        ...,
        "--classifier",
        "-clf",
        help="Path al checkpoint del clasificador"
    ),
    # Warping options
    use_warp: bool = typer.Option(
        False,
        "--warp/--no-warp",
        help="Aplicar normalizacion geometrica (warp) antes de clasificar"
    ),
    landmark_model: Optional[str] = typer.Option(
        None,
        "--landmark-model",
        "-lm",
        help="Path al modelo de landmarks (requerido si --warp)"
    ),
    landmark_ensemble: Optional[list[str]] = typer.Option(
        None,
        "--landmark-ensemble",
        "-le",
        help="Paths a modelos de ensemble para landmarks (alternativa a --landmark-model)"
    ),
    use_tta: bool = typer.Option(
        False,
        "--tta/--no-tta",
        help="Usar Test-Time Augmentation para landmarks"
    ),
    classifier_tta: bool = typer.Option(
        False,
        "--classifier-tta/--no-classifier-tta",
        help="Usar Test-Time Augmentation (flip horizontal) para clasificador"
    ),
    canonical_shape: str = typer.Option(
        "outputs/shape_analysis/canonical_shape_gpa.json",
        "--canonical",
        help="Path a forma canonica (.json)"
    ),
    triangles: str = typer.Option(
        "outputs/shape_analysis/canonical_delaunay_triangles.json",
        "--triangles",
        help="Path a triangulos Delaunay (.json)"
    ),
    margin_scale: float = typer.Option(
        1.05,
        "--margin-scale",
        help="Factor de escala para margenes en warping"
    ),
    # CLAHE
    use_clahe: bool = typer.Option(
        True,
        "--clahe/--no-clahe",
        help="Usar CLAHE para mejora de contraste"
    ),
    clahe_clip: float = typer.Option(
        DEFAULT_CLAHE_CLIP_LIMIT,
        "--clahe-clip",
        help="CLAHE clip limit"
    ),
    clahe_tile: int = typer.Option(
        DEFAULT_CLAHE_TILE_SIZE,
        "--clahe-tile",
        help="CLAHE tile size"
    ),
    # Output
    output_json: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Guardar resultados en JSON"
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo: auto, cuda, cpu, mps"
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Tamano de batch para procesamiento"
    ),
):
    """
    Clasificar imagenes COVID-19 (COVID, Normal, Viral_Pneumonia).

    Modo basico (sin warping):
        python -m src_v2 classify imagen.png --classifier model.pt

    Modo con warping (normalizacion geometrica):
        python -m src_v2 classify imagen.png --classifier clf.pt \\
            --warp --landmark-model lm.pt

    Modo con ensemble de landmarks:
        python -m src_v2 classify directorio/ --classifier clf.pt \\
            --warp --landmark-ensemble m1.pt m2.pt m3.pt m4.pt --tta

    TTA para clasificador (sin warping):
        python -m src_v2 classify imagen.png --classifier clf.pt --classifier-tta
    """
    import json
    import sys

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm

    from src_v2.constants import IMAGENET_MEAN, IMAGENET_STD, SYMMETRIC_PAIRS, CLASSIFIER_CLASSES
    from src_v2.models import create_classifier, get_classifier_transforms, create_model

    logger.info("=" * 60)
    logger.info("COVID-19 Classification")
    logger.info("=" * 60)

    # Verificar parametros de warping
    if use_warp:
        if landmark_model is None and landmark_ensemble is None:
            logger.error("--warp requiere --landmark-model o --landmark-ensemble")
            raise typer.Exit(code=1)

        if not Path(canonical_shape).exists():
            logger.error("Forma canonica no existe: %s", canonical_shape)
            raise typer.Exit(code=1)

        if not Path(triangles).exists():
            logger.error("Triangulos no existen: %s", triangles)
            raise typer.Exit(code=1)

    # Verificar checkpoint del clasificador
    if not Path(classifier_checkpoint).exists():
        logger.error("Checkpoint del clasificador no existe: %s", classifier_checkpoint)
        raise typer.Exit(code=1)

    # Dispositivo
    torch_device = get_device(device)
    logger.info("Dispositivo: %s", torch_device)

    # Cargar clasificador
    logger.info("Cargando clasificador desde: %s", classifier_checkpoint)
    classifier = create_classifier(checkpoint=classifier_checkpoint, device=torch_device)
    classifier.eval()

    # Obtener nombres de clases del checkpoint
    ckpt_data = torch.load(classifier_checkpoint, map_location="cpu", weights_only=False)
    class_names = ckpt_data.get("class_names", CLASSIFIER_CLASSES)
    logger.info("Clases: %s", class_names)

    # Cargar modelos de landmarks si se usa warping
    landmark_models = []
    if use_warp:
        # Cargar forma canonica y triangulos
        if canonical_shape.endswith('.json'):
            with open(canonical_shape, 'r') as f:
                data = json.load(f)
            canonical = np.array(data['canonical_shape_pixels'])
        else:
            canonical = np.load(canonical_shape)

        if triangles.endswith('.json'):
            with open(triangles, 'r') as f:
                data = json.load(f)
            tri = np.array(data['triangles'])
        else:
            tri = np.load(triangles)

        logger.info("Forma canonica: %s, Triangulos: %s", canonical.shape, tri.shape)

        # Cargar modelo(s) de landmarks
        if landmark_ensemble:
            checkpoints_lm = landmark_ensemble
        else:
            checkpoints_lm = [landmark_model]

        for ckpt in checkpoints_lm:
            if not Path(ckpt).exists():
                logger.error("Checkpoint de landmarks no existe: %s", ckpt)
                raise typer.Exit(code=1)

            checkpoint_data = torch.load(ckpt, map_location=torch_device, weights_only=False)
            if "model_state_dict" in checkpoint_data:
                state_dict = checkpoint_data["model_state_dict"]
            else:
                state_dict = checkpoint_data

            arch_params = detect_architecture_from_checkpoint(state_dict)
            model = create_model(
                pretrained=False,
                use_coord_attention=arch_params["use_coord_attention"],
                deep_head=arch_params["deep_head"],
                hidden_dim=arch_params["hidden_dim"],
            )
            model.load_state_dict(state_dict)
            model = model.to(torch_device)
            model.eval()
            landmark_models.append(model)

        logger.info("Modelos de landmarks cargados: %d", len(landmark_models))
        logger.info("TTA: %s", "habilitado" if use_tta else "deshabilitado")

        # Importar funcion de warp desde el modulo correcto
        from src_v2.processing.warp import piecewise_affine_warp

    # Encontrar imagenes
    input_path = Path(image)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = list(input_path.glob("**/*.png")) + list(input_path.glob("**/*.jpg"))
        image_paths = sorted(image_paths)
    else:
        logger.error("Path no existe: %s", image)
        raise typer.Exit(code=1)

    logger.info("Imagenes encontradas: %d", len(image_paths))

    if len(image_paths) == 0:
        logger.warning("No se encontraron imagenes")
        raise typer.Exit(code=0)

    # Transforms para clasificador
    clf_transform = get_classifier_transforms(train=False, img_size=DEFAULT_IMAGE_SIZE)

    # CLAHE processor
    clahe_processor = None
    if use_clahe:
        clahe_processor = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))

    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    def predict_landmarks_ensemble(img_tensor):
        """Predecir landmarks con ensemble + TTA opcional."""
        preds = []
        for model in landmark_models:
            if use_tta:
                with torch.no_grad():
                    pred1 = model(img_tensor)
                    images_flip = torch.flip(img_tensor, dims=[3])
                    pred2 = model(images_flip)
                    pred2 = pred2.view(-1, NUM_LANDMARKS, 2)
                    pred2[:, :, 0] = 1 - pred2[:, :, 0]
                    for left, right in SYMMETRIC_PAIRS:
                        pred2[:, [left, right]] = pred2[:, [right, left]]
                    pred2 = pred2.view(-1, NUM_LANDMARKS * 2)
                    pred = (pred1 + pred2) / 2
            else:
                with torch.no_grad():
                    pred = model(img_tensor)
            preds.append(pred)

        preds_stack = torch.stack(preds)
        return preds_stack.mean(dim=0)

    # Procesar imagenes
    results_list = []

    for img_path in tqdm(image_paths, desc="Clasificando"):
        try:
            # Cargar imagen
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning("No se pudo cargar: %s", img_path)
                continue

            # Resize
            if img.shape[0] != DEFAULT_IMAGE_SIZE or img.shape[1] != DEFAULT_IMAGE_SIZE:
                img = cv2.resize(img, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))

            # Preparar imagen para clasificador
            img_for_clf = img.copy()

            # Warping si esta habilitado
            if use_warp:
                # Aplicar CLAHE para modelo de landmarks
                img_for_lm = img.copy()
                if clahe_processor is not None:
                    img_for_lm = clahe_processor.apply(img_for_lm)

                # Preprocesar para modelo de landmarks
                img_rgb = cv2.cvtColor(img_for_lm, cv2.COLOR_GRAY2RGB)
                img_float = img_rgb.astype(np.float32) / 255.0
                img_normalized = (img_float - mean) / std
                img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
                img_tensor = img_tensor.to(torch_device)

                # Predecir landmarks
                landmarks_pred = predict_landmarks_ensemble(img_tensor)
                landmarks = landmarks_pred.squeeze().cpu().numpy()
                landmarks = landmarks.reshape(NUM_LANDMARKS, 2) * DEFAULT_IMAGE_SIZE

                # Escalar landmarks
                centroid = landmarks.mean(axis=0)
                landmarks_scaled = centroid + (landmarks - centroid) * margin_scale
                landmarks_clipped = np.clip(landmarks_scaled, 2, DEFAULT_IMAGE_SIZE - 3)

                # Escalar canonical
                canonical_scaled = centroid + (canonical - canonical.mean(axis=0)) * margin_scale
                canonical_clipped = np.clip(canonical_scaled, 2, DEFAULT_IMAGE_SIZE - 3)

                # Aplicar warp
                img_for_clf = piecewise_affine_warp(
                    img,
                    landmarks_clipped.astype(np.float32),
                    canonical_clipped.astype(np.float32),
                    tri
                )

            # Convertir a PIL para transforms
            img_pil = Image.fromarray(img_for_clf)
            img_tensor = clf_transform(img_pil).unsqueeze(0).to(torch_device)

            # Clasificar
            with torch.no_grad():
                logits = classifier(img_tensor)
                if classifier_tta:
                    flipped = torch.flip(img_tensor, dims=[3])
                    logits_flipped = classifier(flipped)
                    logits = (logits + logits_flipped) / 2.0
                probs = torch.softmax(logits, dim=1)
                pred_class = probs.argmax(dim=1).item()

            # Resultado
            result = {
                "image": str(img_path),
                "prediction": class_names[pred_class],
                "confidence": float(probs[0, pred_class]),
                "probabilities": {
                    name: float(probs[0, i]) for i, name in enumerate(class_names)
                }
            }
            results_list.append(result)

            # Log individual
            logger.debug("%s -> %s (%.2f%%)",
                        img_path.name, result["prediction"], result["confidence"] * 100)

        except Exception as e:
            logger.warning("Error procesando %s: %s", img_path, e)
            continue

    # Mostrar resumen
    logger.info("-" * 40)
    logger.info("RESUMEN")
    logger.info("-" * 40)
    logger.info("Imagenes procesadas: %d", len(results_list))

    if len(results_list) > 0:
        # Contar predicciones por clase
        from collections import Counter
        pred_counts = Counter(r["prediction"] for r in results_list)
        logger.info("\nDistribucion de predicciones:")
        for cls, count in sorted(pred_counts.items()):
            pct = count / len(results_list) * 100
            logger.info("  %s: %d (%.1f%%)", cls, count, pct)

        # Si es una sola imagen, mostrar detalle
        if len(results_list) == 1:
            r = results_list[0]
            logger.info("\nResultado:")
            logger.info("  Prediccion: %s", r["prediction"])
            logger.info("  Confianza: %.2f%%", r["confidence"] * 100)
            logger.info("  Probabilidades:")
            for cls, prob in sorted(r["probabilities"].items(), key=lambda x: -x[1]):
                logger.info("    %s: %.2f%%", cls, prob * 100)

    # Guardar JSON
    if output_json:
        output_data = {
            "config": {
                "classifier": classifier_checkpoint,
                "warping_enabled": use_warp,
                "tta_enabled": use_tta if use_warp else False,
                "clahe_enabled": use_clahe,
            },
            "results": results_list,
        }
        with open(output_json, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info("\nResultados guardados en: %s", output_json)

    logger.info("=" * 60)
    logger.info("Clasificacion completada!")
    logger.info("=" * 60)


@app.command("train-classifier")
def train_classifier(
    ctx: typer.Context,
    data_dir: Optional[str] = typer.Argument(
        None,
        help="Directorio del dataset (debe contener train/, val/, test/)"
    ),
    output_dir: str = typer.Option(
        "outputs/classifier",
        "--output-dir",
        "-o",
        help="Directorio de salida para modelo y resultados"
    ),
    backbone: str = typer.Option(
        "resnet18",
        "--backbone",
        "-b",
        help="Arquitectura: resnet18, efficientnet_b0 o densenet121"
    ),
    epochs: int = typer.Option(
        50,
        "--epochs",
        "-e",
        help="Numero de epocas"
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        help="Tamano de batch"
    ),
    lr: float = typer.Option(
        1e-4,
        "--lr",
        help="Learning rate"
    ),
    use_class_weights: bool = typer.Option(
        True,
        "--class-weights/--no-class-weights",
        help="Usar pesos de clase para balanceo"
    ),
    patience: int = typer.Option(
        10,
        "--patience",
        help="Paciencia para early stopping"
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo: auto, cuda, cpu, mps"
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Semilla aleatoria"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="JSON con defaults para reproducir entrenamiento"
    ),
):
    """
    Entrenar clasificador CNN para COVID-19.

    El dataset debe tener estructura:
        data_dir/
        ├── train/
        │   ├── COVID/
        │   ├── Normal/
        │   └── Viral_Pneumonia/
        ├── val/
        └── test/

    Ejemplo:
        python -m src_v2 train-classifier outputs/warped_lung_best/session_warping \\
            --backbone resnet18 --epochs 50 --batch-size 32
        python -m src_v2 train-classifier --config configs/classifier_warped_base.json
    """
    import json
    from collections import Counter
    from datetime import datetime

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

    from src_v2.models import ImageClassifier, get_classifier_transforms, get_class_weights

    logger.info("=" * 60)
    logger.info("COVID-19 Classifier Training")
    logger.info("=" * 60)

    if config:
        config_path = Path(config)
        if not config_path.is_file():
            logger.error("Config no existe: %s", config)
            raise typer.Exit(code=1)
        with config_path.open("r") as handle:
            config_data = json.load(handle)
        if not isinstance(config_data, dict):
            logger.error("Config debe ser un JSON con pares clave/valor.")
            raise typer.Exit(code=1)

        alias_map = {"model": "backbone"}
        normalized = {}
        for key, value in config_data.items():
            normalized[alias_map.get(key, key)] = value

        valid_keys = {
            "data_dir",
            "output_dir",
            "backbone",
            "epochs",
            "batch_size",
            "lr",
            "use_class_weights",
            "patience",
            "device",
            "seed",
        }
        unknown_keys = sorted(set(normalized) - valid_keys)
        if unknown_keys:
            logger.error("Config con claves desconocidas: %s", ", ".join(unknown_keys))
            raise typer.Exit(code=1)

        param_values = {
            "data_dir": data_dir,
            "output_dir": output_dir,
            "backbone": backbone,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "use_class_weights": use_class_weights,
            "patience": patience,
            "device": device,
            "seed": seed,
        }
        def is_default_source(source: object) -> bool:
            if source is None:
                return True
            source_name = getattr(source, "name", None)
            if source_name:
                return source_name == "DEFAULT"
            return str(source).lower() == "default"

        for key, value in normalized.items():
            if is_default_source(ctx.get_parameter_source(key)):
                param_values[key] = value

        data_dir = param_values["data_dir"]
        output_dir = param_values["output_dir"]
        backbone = param_values["backbone"]
        epochs = param_values["epochs"]
        batch_size = param_values["batch_size"]
        lr = param_values["lr"]
        use_class_weights = param_values["use_class_weights"]
        patience = param_values["patience"]
        device = param_values["device"]
        seed = param_values["seed"]

        logger.info("Config cargada: %s", config_path)

    # Configurar semilla completa para reproducibilidad
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Verificar directorios
    if not data_dir:
        logger.error("data_dir es requerido. Usa argumento o --config.")
        raise typer.Exit(code=1)

    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    test_dir = data_path / "test"

    for d in [train_dir, val_dir, test_dir]:
        if not d.exists():
            logger.error("Directorio no existe: %s", d)
            raise typer.Exit(code=1)

    # Crear directorio de salida
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Dispositivo
    torch_device = get_device(device)
    logger.info("Dispositivo: %s", torch_device)

    # Transforms
    train_transform = get_classifier_transforms(train=True, img_size=DEFAULT_IMAGE_SIZE)
    eval_transform = get_classifier_transforms(train=False, img_size=DEFAULT_IMAGE_SIZE)

    # Datasets
    logger.info("Cargando datasets...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    class_names = train_dataset.classes
    logger.info("Clases: %s", class_names)
    logger.info("Train: %d, Val: %d, Test: %d",
                len(train_dataset), len(val_dataset), len(test_dataset))

    # Distribucion de clases
    train_labels = [train_dataset.targets[i] for i in range(len(train_dataset))]
    train_counts = Counter(train_labels)
    logger.info("Distribucion train:")
    for idx, name in enumerate(class_names):
        logger.info("  %s: %d", name, train_counts[idx])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=get_optimal_num_workers())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=get_optimal_num_workers())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=get_optimal_num_workers())

    # Modelo
    logger.info("Creando modelo %s...", backbone)
    model = ImageClassifier(
        backbone=backbone,
        num_classes=len(class_names),
        pretrained=True,
        dropout=0.3,
    )
    model = model.to(torch_device)

    # Class weights
    class_weights_tensor = None
    if use_class_weights:
        class_weights_tensor = get_class_weights(train_labels, len(class_names)).to(torch_device)
        logger.info("Class weights: %s", class_weights_tensor.cpu().numpy())

    # Loss y optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # Training loop
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_f1_macro': [], 'val_f1_weighted': []
    }

    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None

    logger.info("Iniciando entrenamiento: %d epocas", epochs)

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(torch_device), labels.to(torch_device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(torch_device), labels.to(torch_device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(all_labels)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1_macro = f1_score(all_labels, all_preds, average='macro')
        val_f1_weighted = f1_score(all_labels, all_preds, average='weighted')

        # Scheduler
        scheduler.step(val_f1_macro)

        # History
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1_macro'].append(val_f1_macro)
        history['val_f1_weighted'].append(val_f1_weighted)

        logger.info("Epoch %3d/%d: Train Loss=%.4f Acc=%.4f | Val Loss=%.4f Acc=%.4f F1=%.4f",
                   epoch + 1, epochs, train_loss, train_acc, val_loss, val_acc, val_f1_macro)

        # Early stopping
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            logger.info("  -> Nuevo mejor modelo: F1 = %.4f", best_val_f1)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping en epoch %d", epoch + 1)
                break

    # Cargar mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Guardar modelo
    model_path = output_path / "best_classifier.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'model_name': backbone,
        'best_val_f1': best_val_f1,
    }, model_path)
    logger.info("Modelo guardado en: %s", model_path)

    # Evaluacion final en test
    logger.info("-" * 40)
    logger.info("EVALUACION FINAL EN TEST")
    logger.info("-" * 40)

    model.eval()
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(torch_device), labels.to(torch_device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1_macro = f1_score(test_labels, test_preds, average='macro')
    test_f1_weighted = f1_score(test_labels, test_preds, average='weighted')

    logger.info("Test Accuracy: %.4f", test_acc)
    logger.info("Test F1 Macro: %.4f", test_f1_macro)
    logger.info("Test F1 Weighted: %.4f", test_f1_weighted)

    logger.info("\nClassification Report:")
    report = classification_report(test_labels, test_preds, target_names=class_names)
    logger.info("\n%s", report)

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    logger.info("Confusion Matrix:")
    logger.info("\n%s", cm)

    # Guardar resultados
    results = {
        'model': backbone,
        'epochs_trained': len(history['train_loss']),
        'best_val_f1': float(best_val_f1),
        'test_metrics': {
            'accuracy': float(test_acc),
            'f1_macro': float(test_f1_macro),
            'f1_weighted': float(test_f1_weighted),
        },
        'per_class_metrics': classification_report(
            test_labels, test_preds, target_names=class_names, output_dict=True
        ),
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'timestamp': datetime.now().isoformat(),
    }

    results_path = output_path / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("Resultados guardados en: %s", results_path)

    logger.info("=" * 60)
    logger.info("Entrenamiento completado!")
    logger.info("=" * 60)


@app.command("evaluate-classifier")
def evaluate_classifier(
    checkpoint: str = typer.Argument(
        ...,
        help="Path al checkpoint del clasificador"
    ),
    data_dir: str = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Directorio del dataset"
    ),
    split: str = typer.Option(
        "test",
        "--split",
        "-s",
        help="Split a evaluar: test, val, all"
    ),
    output_json: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Guardar resultados en JSON"
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo: auto, cuda, cpu, mps"
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        help="Tamano de batch"
    ),
    tta: bool = typer.Option(
        False,
        "--tta/--no-tta",
        help="Usar Test-Time Augmentation (flip horizontal) en evaluacion"
    ),
):
    """
    Evaluar clasificador en dataset.

    Ejemplo:
        python -m src_v2 evaluate-classifier outputs/classifier/best_classifier.pt \\
            --data-dir outputs/warped_lung_best/session_warping --split test
        python -m src_v2 evaluate-classifier outputs/classifier/best_classifier.pt \\
            --data-dir outputs/warped_lung_best/session_warping --split test --tta
    """
    import json

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

    from src_v2.models import create_classifier, get_classifier_transforms

    logger.info("=" * 60)
    logger.info("COVID-19 Classifier Evaluation")
    logger.info("=" * 60)

    # Verificar paths
    if not Path(checkpoint).exists():
        logger.error("Checkpoint no existe: %s", checkpoint)
        raise typer.Exit(code=1)

    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error("Directorio de datos no existe: %s", data_dir)
        raise typer.Exit(code=1)

    # Dispositivo
    torch_device = get_device(device)
    logger.info("Dispositivo: %s", torch_device)

    # Cargar modelo
    logger.info("Cargando modelo desde: %s", checkpoint)
    model = create_classifier(checkpoint=checkpoint, device=torch_device)
    model.eval()

    # Obtener class names del checkpoint
    ckpt_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    class_names = ckpt_data.get("class_names", CLASSIFIER_CLASSES)

    # Transforms
    eval_transform = get_classifier_transforms(train=False, img_size=DEFAULT_IMAGE_SIZE)

    # Cargar datasets segun split
    if split == "all":
        # Combinar todos los splits
        all_datasets = []
        for s in ["train", "val", "test"]:
            s_dir = data_path / s
            if s_dir.exists():
                ds = datasets.ImageFolder(s_dir, transform=eval_transform)
                all_datasets.append(ds)
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset(all_datasets)
        logger.info("Evaluando todos los splits: %d muestras", len(dataset))
    else:
        split_dir = data_path / split
        if not split_dir.exists():
            logger.error("Split '%s' no existe en %s", split, data_dir)
            raise typer.Exit(code=1)
        dataset = datasets.ImageFolder(split_dir, transform=eval_transform)
        logger.info("Evaluando split '%s': %d muestras", split, len(dataset))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=get_optimal_num_workers())

    # Evaluar
    all_preds = []
    all_labels = []

    logger.info("Evaluando...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(torch_device)
            outputs = model(inputs)
            if tta:
                flipped = torch.flip(inputs, dims=[3])
                outputs_flipped = model(flipped)
                outputs = (outputs + outputs_flipped) / 2.0
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Metricas
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    # Mostrar resultados
    logger.info("-" * 40)
    logger.info("RESULTADOS")
    logger.info("-" * 40)
    logger.info("Accuracy: %.4f (%.2f%%)", acc, acc * 100)
    logger.info("F1 Macro: %.4f", f1_macro)
    logger.info("F1 Weighted: %.4f", f1_weighted)

    logger.info("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    logger.info("\n%s", report)

    logger.info("Confusion Matrix:")
    logger.info("\n%s", cm)

    # Guardar JSON
    if output_json:
        results = {
            "checkpoint": checkpoint,
            "data_dir": data_dir,
            "split": split,
            "n_samples": len(all_labels),
            "metrics": {
                "accuracy": float(acc),
                "f1_macro": float(f1_macro),
                "f1_weighted": float(f1_weighted),
            },
            "per_class": classification_report(
                all_labels, all_preds, target_names=class_names, output_dict=True
            ),
            "confusion_matrix": cm.tolist(),
            "class_names": class_names,
        }
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("\nResultados guardados en: %s", output_json)

    logger.info("=" * 60)
    logger.info("Evaluacion completada!")
    logger.info("=" * 60)


@app.command("evaluate-classifier-ensemble")
def evaluate_classifier_ensemble(
    config: str = typer.Option(
        ...,
        "--config", "-c",
        help="Path to ensemble configuration JSON"
    ),
    output: str = typer.Option(
        "outputs/classifier_cv/ensemble_test_results.json",
        "--output", "-o",
        help="Output JSON path"
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device: auto, cuda, cpu, mps"
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        help="Batch size for inference"
    ),
    predictions_csv: Optional[str] = typer.Option(
        None,
        "--predictions-csv",
        help="Save per-sample predictions to CSV (debug output)"
    ),
    tta: Optional[bool] = typer.Option(
        None,
        "--tta/--no-tta",
        help="Enable/disable TTA (overrides config use_tta if provided)"
    ),
):
    """
    Evaluate 5-fold cross-validation ensemble on test set.

    Computes both soft voting (weighted probability averaging) and
    hard voting (majority vote) for comparison. This establishes
    baseline ensemble metrics; final evaluation with validation
    checks is performed in Phase 5.

    Example:
        python -m src_v2 evaluate-classifier-ensemble \\
            --config configs/ensemble_classifier.json \\
            --output outputs/classifier_cv/ensemble_test_results.json
    """
    import json
    from datetime import datetime

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        accuracy_score,
    )

    from src_v2.models import get_classifier_transforms
    from src_v2.evaluation.ensemble import (
        load_ensemble_models,
        weighted_soft_voting,
        hard_voting,
        ensemble_inference,
        ensemble_inference_with_tta,
        validate_ensemble_setup,
        categorize_tta_impact,
        compute_tta_delta_metrics,
    )

    logger.info("=" * 60)
    logger.info("COVID-19 Classifier Ensemble Evaluation")
    logger.info("=" * 60)

    # Load config
    config_path = Path(config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config)
        raise typer.Exit(code=1)

    with open(config_path) as f:
        config_data = json.load(f)

    checkpoint_paths = config_data.get("checkpoint_paths")
    data_dir = config_data.get("data_dir")
    baseline_accuracy = config_data.get("baseline_accuracy", 0.9768)

    # Resolve TTA setting: CLI overrides config
    use_tta = tta if tta is not None else config_data.get("use_tta", True)
    logger.info("TTA enabled: %s", use_tta)

    if not checkpoint_paths:
        logger.error("Config missing 'checkpoint_paths' field")
        raise typer.Exit(code=1)
    if not data_dir:
        logger.error("Config missing 'data_dir' field")
        raise typer.Exit(code=1)

    # Verify checkpoint paths exist
    for i, ckpt_path in enumerate(checkpoint_paths):
        if not Path(ckpt_path).exists():
            logger.error("Checkpoint not found: %s", ckpt_path)
            raise typer.Exit(code=1)

    # Verify data directory
    data_path = Path(data_dir)
    test_dir = data_path / "test"
    if not test_dir.exists():
        logger.error("Test directory not found: %s", test_dir)
        raise typer.Exit(code=1)

    # Setup device
    torch_device = get_device(device)
    logger.info("Device: %s", torch_device)
    logger.info("Number of models: %d", len(checkpoint_paths))

    # Load ensemble models
    logger.info("Loading ensemble models...")
    models, weights = load_ensemble_models(checkpoint_paths, torch_device)
    logger.info("Validation weights (F1-macro): %s", [f"{w:.4f}" for w in weights])

    # Get class names from first checkpoint
    ckpt_data = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
    class_names = ckpt_data.get("class_names", CLASSIFIER_CLASSES)
    logger.info("Classes: %s", class_names)

    # Prepare test dataset
    eval_transform = get_classifier_transforms(train=False, img_size=DEFAULT_IMAGE_SIZE)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=get_optimal_num_workers()
    )
    logger.info("Test samples: %d", len(test_dataset))

    # Validate ensemble setup
    logger.info("Running sanity checks...")
    validate_ensemble_setup(models, test_loader, expected_samples=len(test_dataset))

    # Run ensemble inference
    logger.info("Running ensemble inference...")
    model_preds, model_probs, labels, tta_details = ensemble_inference_with_tta(
        models, test_loader, torch_device, use_tta=use_tta
    )

    # Compute baseline predictions for TTA comparison (if TTA enabled)
    if use_tta:
        logger.info("Computing baseline (no-TTA) predictions for case-level comparison...")
        logger.info("NOTE: This runs a second inference pass without TTA (doubles inference time)")
        baseline_model_preds, baseline_model_probs, _, _ = ensemble_inference_with_tta(
            models, test_loader, torch_device, use_tta=False
        )
        baseline_soft_preds, _ = weighted_soft_voting(baseline_model_probs, weights)
        baseline_soft_preds_np = baseline_soft_preds.numpy()

    # Compute per-fold metrics
    logger.info("Computing per-fold metrics...")
    per_fold_metrics = []
    for i in range(len(models)):
        fold_preds = model_preds[i].numpy()
        fold_acc = accuracy_score(labels, fold_preds)
        fold_f1_macro = f1_score(labels, fold_preds, average='macro')
        fold_f1_weighted = f1_score(labels, fold_preds, average='weighted')
        per_fold_metrics.append({
            "fold": i + 1,
            "checkpoint": str(checkpoint_paths[i]),
            "validation_f1_macro": weights[i],
            "test_metrics": {
                "accuracy": float(fold_acc),
                "f1_macro": float(fold_f1_macro),
                "f1_weighted": float(fold_f1_weighted),
            }
        })

    # Compute soft voting ensemble
    logger.info("Computing soft voting ensemble...")
    soft_preds, soft_probs = weighted_soft_voting(model_probs, weights)
    soft_preds_np = soft_preds.numpy()

    soft_acc = accuracy_score(labels, soft_preds_np)
    soft_f1_macro = f1_score(labels, soft_preds_np, average='macro')
    soft_f1_weighted = f1_score(labels, soft_preds_np, average='weighted')
    soft_cm = confusion_matrix(labels, soft_preds_np)
    soft_report = classification_report(
        labels, soft_preds_np, target_names=class_names, output_dict=True
    )

    ensemble_soft_metrics = {
        "method": "weighted_probability_averaging",
        "weights_source": "validation_f1_macro",
        "weights": weights,
        "metrics": {
            "accuracy": float(soft_acc),
            "f1_macro": float(soft_f1_macro),
            "f1_weighted": float(soft_f1_weighted),
        },
        "per_class": soft_report,
        "confusion_matrix": soft_cm.tolist(),
    }

    # Compute TTA case-level impact and delta metrics
    if use_tta:
        # Case-level analysis: helped/hurt/neutral
        case_level = categorize_tta_impact(
            pred_baseline=baseline_soft_preds_np,
            pred_tta=soft_preds_np,
            ground_truth=labels
        )
        logger.info("TTA Case-Level Impact: helped=%d, hurt=%d, neutral=%d",
            case_level["summary"]["helped"],
            case_level["summary"]["hurt"],
            case_level["summary"]["neutral"])

        # Delta metrics: accuracy and F1 deltas
        baseline_acc = accuracy_score(labels, baseline_soft_preds_np)
        baseline_f1_macro = f1_score(labels, baseline_soft_preds_np, average='macro')
        baseline_report = classification_report(
            labels, baseline_soft_preds_np, target_names=class_names, output_dict=True
        )
        baseline_metrics_dict = {
            "accuracy": float(baseline_acc),
            "f1_macro": float(baseline_f1_macro),
            "per_class": baseline_report,
        }
        tta_metrics_dict = {
            "accuracy": float(soft_acc),
            "f1_macro": float(soft_f1_macro),
            "per_class": soft_report,
        }
        delta_metrics = compute_tta_delta_metrics(
            baseline_metrics=baseline_metrics_dict,
            tta_metrics=tta_metrics_dict,
            class_names=class_names,
        )
        logger.info("TTA Delta: accuracy=%+.4f, f1_macro=%+.4f",
            delta_metrics["accuracy_delta"],
            delta_metrics["f1_macro_delta"])

    # Compute hard voting ensemble
    logger.info("Computing hard voting ensemble...")
    hard_preds = hard_voting(model_preds)
    hard_preds_np = hard_preds.numpy()

    hard_acc = accuracy_score(labels, hard_preds_np)
    hard_f1_macro = f1_score(labels, hard_preds_np, average='macro')
    hard_f1_weighted = f1_score(labels, hard_preds_np, average='weighted')

    ensemble_hard_metrics = {
        "method": "majority_vote",
        "metrics": {
            "accuracy": float(hard_acc),
            "f1_macro": float(hard_f1_macro),
            "f1_weighted": float(hard_f1_weighted),
        },
    }

    # Compute comparison with baseline
    baseline_mean = baseline_accuracy
    baseline_std = config_data.get("baseline_std", 0.0016)
    comparison = {
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "ensemble_soft_delta": float(soft_acc - baseline_mean),
        "ensemble_hard_delta": float(hard_acc - baseline_mean),
    }

    # Create output JSON
    output_data = {
        "description": "5-fold ensemble evaluation on test set",
        "timestamp": datetime.now().isoformat(),
        "tta_enabled": use_tta,
        "n_folds": len(checkpoint_paths),
        "test_set_size": len(labels),
        "per_fold_metrics": per_fold_metrics,
        "ensemble_soft_voting": ensemble_soft_metrics,
        "ensemble_hard_voting": ensemble_hard_metrics,
        "comparison": comparison,
        "class_names": class_names,
    }

    # Add TTA-specific fields to output
    if use_tta:
        # Only include summary in JSON (per_sample list is large)
        output_data["case_level_analysis"] = {
            "summary": case_level["summary"],
            "net_improvement": case_level["summary"]["helped"] - case_level["summary"]["hurt"],
        }
        output_data["tta_delta_metrics"] = delta_metrics

    # Save output JSON
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info("Results saved to: %s", output_path)

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("ENSEMBLE EVALUATION RESULTS")
    logger.info("=" * 60)

    # Per-fold metrics
    logger.info("\n--- Per-Fold Metrics (Test Set) ---")
    logger.info(f"{'Fold':<6} {'Val F1':>8} {'Acc':>8} {'F1-Macro':>10} {'F1-Weighted':>12}")
    logger.info("-" * 50)
    for fold_result in per_fold_metrics:
        fold_num = fold_result["fold"]
        val_f1 = fold_result["validation_f1_macro"]
        test_acc = fold_result["test_metrics"]["accuracy"]
        test_f1_macro = fold_result["test_metrics"]["f1_macro"]
        test_f1_weighted = fold_result["test_metrics"]["f1_weighted"]
        logger.info(
            f"{fold_num:<6} {val_f1:>8.4f} {test_acc:>8.4f} {test_f1_macro:>10.4f} {test_f1_weighted:>12.4f}"
        )

    # Ensemble soft voting
    logger.info("\n--- Ensemble Soft Voting (Weighted Probability Averaging) ---")
    logger.info(f"Accuracy:    {soft_acc:.4f} ({soft_acc*100:.2f}%)")
    logger.info(f"F1-Macro:    {soft_f1_macro:.4f}")
    logger.info(f"F1-Weighted: {soft_f1_weighted:.4f}")
    logger.info(f"Delta vs Baseline: {comparison['ensemble_soft_delta']:+.4f}")

    # Ensemble hard voting
    logger.info("\n--- Ensemble Hard Voting (Majority Vote) ---")
    logger.info(f"Accuracy:    {hard_acc:.4f} ({hard_acc*100:.2f}%)")
    logger.info(f"F1-Macro:    {hard_f1_macro:.4f}")
    logger.info(f"F1-Weighted: {hard_f1_weighted:.4f}")
    logger.info(f"Delta vs Baseline: {comparison['ensemble_hard_delta']:+.4f}")

    # Comparison
    logger.info("\n--- Comparison with Baseline ---")
    logger.info(f"Baseline (mean ± std): {baseline_mean:.4f} ± {baseline_std:.4f}")
    logger.info(f"Soft Voting:           {soft_acc:.4f} ({comparison['ensemble_soft_delta']:+.4f})")
    logger.info(f"Hard Voting:           {hard_acc:.4f} ({comparison['ensemble_hard_delta']:+.4f})")

    # Confusion matrix
    logger.info("\nConfusion Matrix (Soft Voting):")
    logger.info(f"\n{soft_cm}")

    # TTA case-level impact and delta metrics (if TTA enabled)
    if use_tta:
        logger.info("\n--- TTA Case-Level Impact ---")
        logger.info(f"Helped (baseline wrong, TTA correct): {case_level['summary']['helped']}")
        logger.info(f"Hurt (baseline correct, TTA wrong):   {case_level['summary']['hurt']}")
        logger.info(f"Neutral (same outcome):               {case_level['summary']['neutral']}")
        logger.info(f"Net improvement: {case_level['summary']['helped'] - case_level['summary']['hurt']}")
        logger.info("\n--- TTA Delta Metrics ---")
        logger.info(f"Accuracy delta: {delta_metrics['accuracy_delta']:+.4f}")
        logger.info(f"F1-Macro delta: {delta_metrics['f1_macro_delta']:+.4f}")
        for cls, delta in delta_metrics['per_class_f1_delta'].items():
            logger.info(f"  {cls} F1 delta: {delta:+.4f}")

    # Save predictions CSV if requested
    if predictions_csv:
        import pandas as pd
        pred_df = pd.DataFrame({
            "sample_idx": range(len(labels)),
            "true_label": labels,
            "soft_prediction": soft_preds_np,
            "hard_prediction": hard_preds_np,
            "soft_correct": (labels == soft_preds_np).astype(int),
            "hard_correct": (labels == hard_preds_np).astype(int),
        })
        pred_df.to_csv(predictions_csv, index=False)
        logger.info(f"\nPredictions saved to: {predictions_csv}")

    logger.info("\n" + "=" * 60)
    logger.info("Ensemble evaluation completed!")
    logger.info("=" * 60)


@app.command("cross-validate-classifier")
def cross_validate_classifier(
    ctx: typer.Context,
    data_dir: Optional[str] = typer.Argument(
        None,
        help="Directorio del dataset (debe contener train/, val/, test/)"
    ),
    output_dir: str = typer.Option(
        "outputs/classifier_cross_validation",
        "--output-dir",
        "-o",
        help="Directorio de salida para resultados por fold"
    ),
    backbone: str = typer.Option(
        "resnet18",
        "--backbone",
        "-b",
        help="Arquitectura: resnet18, efficientnet_b0 o densenet121"
    ),
    epochs: int = typer.Option(
        50,
        "--epochs",
        "-e",
        help="Numero de epocas"
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        help="Tamano de batch"
    ),
    lr: float = typer.Option(
        1e-4,
        "--lr",
        help="Learning rate"
    ),
    use_class_weights: bool = typer.Option(
        True,
        "--class-weights/--no-class-weights",
        help="Usar pesos de clase para balanceo"
    ),
    patience: int = typer.Option(
        10,
        "--patience",
        help="Paciencia para early stopping"
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo: auto, cuda, cpu, mps"
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Semilla aleatoria"
    ),
    folds: int = typer.Option(
        5,
        "--folds",
        "-k",
        help="Numero de folds para validacion cruzada"
    ),
    eval_test: bool = typer.Option(
        False,
        "--eval-test/--no-eval-test",
        help="Evaluar cada fold en el split test fijo"
    ),
    save_checkpoints: bool = typer.Option(
        True,
        "--save-checkpoints/--no-save-checkpoints",
        help="Guardar el mejor modelo por fold"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="JSON con defaults para reproducir entrenamiento"
    ),
):
    """
    Validacion cruzada (k-fold) para el clasificador CNN.

    Usa train+val como pool para folds estratificados y deja test fijo.

    Ejemplo:
        python -m src_v2 cross-validate-classifier \\
            outputs/warped_lung_best/session_warping \\
            --folds 5 --device cuda --eval-test
        python -m src_v2 cross-validate-classifier --config configs/classifier_warped_base.json \\
            --folds 5 --output-dir outputs/classifier_cv
    """
    import copy
    import json
    import random
    from collections import Counter
    from datetime import datetime

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from PIL import Image
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader, Dataset
    from torchvision import datasets

    from src_v2.models import ImageClassifier, get_classifier_transforms, get_class_weights

    logger.info("=" * 60)
    logger.info("COVID-19 Classifier Cross-Validation")
    logger.info("=" * 60)

    if config:
        config_path = Path(config)
        if not config_path.is_file():
            logger.error("Config no existe: %s", config)
            raise typer.Exit(code=1)
        with config_path.open("r") as handle:
            config_data = json.load(handle)
        if not isinstance(config_data, dict):
            logger.error("Config debe ser un JSON con pares clave/valor.")
            raise typer.Exit(code=1)

        alias_map = {"model": "backbone"}
        normalized = {}
        for key, value in config_data.items():
            normalized[alias_map.get(key, key)] = value

        valid_keys = {
            "data_dir",
            "output_dir",
            "backbone",
            "epochs",
            "batch_size",
            "lr",
            "use_class_weights",
            "patience",
            "device",
            "seed",
            "folds",
            "eval_test",
            "save_checkpoints",
        }
        unknown_keys = sorted(set(normalized) - valid_keys)
        if unknown_keys:
            logger.error("Config con claves desconocidas: %s", ", ".join(unknown_keys))
            raise typer.Exit(code=1)

        param_values = {
            "data_dir": data_dir,
            "output_dir": output_dir,
            "backbone": backbone,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "use_class_weights": use_class_weights,
            "patience": patience,
            "device": device,
            "seed": seed,
            "folds": folds,
            "eval_test": eval_test,
            "save_checkpoints": save_checkpoints,
        }

        def is_default_source(source: object) -> bool:
            if source is None:
                return True
            source_name = getattr(source, "name", None)
            if source_name:
                return source_name == "DEFAULT"
            return str(source).lower() == "default"

        for key, value in normalized.items():
            if is_default_source(ctx.get_parameter_source(key)):
                param_values[key] = value

        data_dir = param_values["data_dir"]
        output_dir = param_values["output_dir"]
        backbone = param_values["backbone"]
        epochs = param_values["epochs"]
        batch_size = param_values["batch_size"]
        lr = param_values["lr"]
        use_class_weights = param_values["use_class_weights"]
        patience = param_values["patience"]
        device = param_values["device"]
        seed = param_values["seed"]
        folds = param_values["folds"]
        eval_test = param_values["eval_test"]
        save_checkpoints = param_values["save_checkpoints"]

        logger.info("Config cargada: %s", config_path)

    if not data_dir:
        logger.error("data_dir es requerido. Usa argumento o --config.")
        raise typer.Exit(code=1)

    if folds < 2:
        logger.error("folds debe ser >= 2 (recibido=%d)", folds)
        raise typer.Exit(code=1)

    # Configurar semilla completa para reproducibilidad
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    test_dir = data_path / "test"

    for split_dir in [train_dir, val_dir, test_dir]:
        if not split_dir.exists():
            logger.error("Directorio no existe: %s", split_dir)
            raise typer.Exit(code=1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch_device = get_device(device)
    logger.info("Dispositivo: %s", torch_device)

    base_train = datasets.ImageFolder(train_dir)
    base_val = datasets.ImageFolder(val_dir)

    if base_train.classes != base_val.classes:
        logger.error("Las clases de train y val no coinciden.")
        raise typer.Exit(code=1)

    class_names = base_train.classes
    full_samples = base_train.samples + base_val.samples
    full_labels = base_train.targets + base_val.targets

    class_counts = Counter(full_labels)
    min_count = min(class_counts.values()) if class_counts else 0
    if min_count < folds:
        logger.error(
            "Clases insuficientes para %d folds (min_count=%d).",
            folds,
            min_count,
        )
        raise typer.Exit(code=1)

    logger.info("Clases: %s", class_names)
    logger.info("Train+Val pool: %d muestras", len(full_samples))
    for idx, name in enumerate(class_names):
        logger.info("  %s: %d", name, class_counts.get(idx, 0))

    eval_transform = get_classifier_transforms(train=False, img_size=DEFAULT_IMAGE_SIZE)
    train_transform = get_classifier_transforms(train=True, img_size=DEFAULT_IMAGE_SIZE)

    test_dataset = None
    if eval_test:
        test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)
        logger.info("Test fijo: %d muestras", len(test_dataset))

    labels_order = list(range(len(class_names)))

    class ImagePathDataset(Dataset):
        """Dataset basado en una lista de (path, label)."""

        def __init__(self, samples: list[tuple[str, int]], transform):
            self.samples = samples
            self.transform = transform

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int):
            path, label = self.samples[idx]
            with Image.open(path) as img:
                image = img.convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label

    def evaluate_basic(model, loader, criterion):
        model.eval()
        total_loss = 0.0
        preds = []
        labels = []
        with torch.no_grad():
            for inputs, labels_batch in loader:
                inputs = inputs.to(torch_device)
                labels_batch = labels_batch.to(torch_device)
                outputs = model(inputs)
                loss = criterion(outputs, labels_batch)
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                preds.extend(predicted.cpu().numpy())
                labels.extend(labels_batch.cpu().numpy())
        if not labels:
            return 0.0, 0.0, 0.0, 0.0
        val_loss = total_loss / len(labels)
        val_acc = accuracy_score(labels, preds)
        val_f1_macro = f1_score(labels, preds, average="macro")
        val_f1_weighted = f1_score(labels, preds, average="weighted")
        return val_loss, val_acc, val_f1_macro, val_f1_weighted

    def evaluate_full(model, loader):
        if loader is None:
            return None
        model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for inputs, labels_batch in loader:
                inputs = inputs.to(torch_device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                preds.extend(predicted.cpu().numpy())
                labels.extend(labels_batch.numpy())

        if not labels:
            empty_cm = [[0 for _ in labels_order] for _ in labels_order]
            return {
                "accuracy": 0.0,
                "f1_macro": 0.0,
                "f1_weighted": 0.0,
                "per_class_metrics": {},
                "confusion_matrix": empty_cm,
                "n_samples": 0,
            }

        acc = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average="macro")
        f1_weighted = f1_score(labels, preds, average="weighted")
        report = classification_report(
            labels,
            preds,
            labels=labels_order,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(labels, preds, labels=labels_order)
        return {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "per_class_metrics": report,
            "confusion_matrix": cm.tolist(),
            "n_samples": len(labels),
        }

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_results = []

    logger.info("Iniciando %d-fold CV...", folds)

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(full_samples, full_labels),
        start=1,
    ):
        logger.info("-" * 40)
        logger.info("Fold %d/%d", fold_idx, folds)

        fold_dir = output_path / f"fold_{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_samples = [full_samples[i] for i in train_idx]
        val_samples = [full_samples[i] for i in val_idx]

        train_labels = [full_labels[i] for i in train_idx]
        val_labels = [full_labels[i] for i in val_idx]

        train_counts = Counter(train_labels)
        val_counts = Counter(val_labels)

        logger.info("Train: %d muestras", len(train_samples))
        for idx, name in enumerate(class_names):
            logger.info("  %s: %d", name, train_counts.get(idx, 0))

        logger.info("Val: %d muestras", len(val_samples))
        for idx, name in enumerate(class_names):
            logger.info("  %s: %d", name, val_counts.get(idx, 0))

        train_dataset = ImagePathDataset(train_samples, train_transform)
        val_dataset = ImagePathDataset(val_samples, eval_transform)

        use_pin_memory = torch_device.type == "cuda"
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=get_optimal_num_workers(),
            pin_memory=use_pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=get_optimal_num_workers(),
            pin_memory=use_pin_memory,
        )

        test_loader = None
        if eval_test and test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=get_optimal_num_workers(),
                pin_memory=use_pin_memory,
            )

        model = ImageClassifier(
            backbone=backbone,
            num_classes=len(class_names),
            pretrained=True,
            dropout=0.3,
        )
        model = model.to(torch_device)

        class_weights_tensor = None
        if use_class_weights:
            class_weights_tensor = get_class_weights(train_labels, len(class_names)).to(torch_device)
            logger.info("Class weights: %s", class_weights_tensor.cpu().numpy())

        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1_macro": [],
            "val_f1_weighted": [],
        }

        best_val_f1 = 0.0
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, labels_batch in train_loader:
                inputs = inputs.to(torch_device)
                labels_batch = labels_batch.to(torch_device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                train_total += labels_batch.size(0)
                train_correct += predicted.eq(labels_batch).sum().item()

            train_loss /= max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)

            val_loss, val_acc, val_f1_macro, val_f1_weighted = evaluate_basic(
                model,
                val_loader,
                criterion,
            )

            scheduler.step(val_f1_macro)

            history["train_loss"].append(float(train_loss))
            history["train_acc"].append(float(train_acc))
            history["val_loss"].append(float(val_loss))
            history["val_acc"].append(float(val_acc))
            history["val_f1_macro"].append(float(val_f1_macro))
            history["val_f1_weighted"].append(float(val_f1_weighted))

            logger.info(
                "Epoch %3d/%d: Train Loss=%.4f Acc=%.4f | Val Loss=%.4f Acc=%.4f F1=%.4f",
                epoch + 1,
                epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                val_f1_macro,
            )

            if val_f1_macro > best_val_f1:
                best_val_f1 = val_f1_macro
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
                logger.info("  -> Nuevo mejor modelo: F1 = %.4f", best_val_f1)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping en epoch %d", epoch + 1)
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        val_metrics = evaluate_full(model, val_loader)
        test_metrics = evaluate_full(model, test_loader) if test_loader else None

        if save_checkpoints:
            model_path = fold_dir / "best_classifier.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "model_name": backbone,
                    "best_val_f1": float(best_val_f1),
                    "fold": fold_idx,
                    "k_folds": folds,
                },
                model_path,
            )
            logger.info("Modelo guardado en: %s", model_path)

        history_path = fold_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        fold_result = {
            "fold": fold_idx,
            "epochs_trained": len(history["train_loss"]),
            "best_val_f1": float(best_val_f1),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "class_distribution": {
                "train": {class_names[i]: int(train_counts.get(i, 0)) for i in labels_order},
                "val": {class_names[i]: int(val_counts.get(i, 0)) for i in labels_order},
            },
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

        results_path = fold_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(fold_result, f, indent=2)
        logger.info("Resultados guardados en: %s", results_path)

        fold_results.append(fold_result)

    val_accs = [r["val_metrics"]["accuracy"] for r in fold_results]
    val_f1s = [r["val_metrics"]["f1_macro"] for r in fold_results]
    val_f1w = [r["val_metrics"]["f1_weighted"] for r in fold_results]

    summary = {
        "val_accuracy_mean": float(np.mean(val_accs)),
        "val_accuracy_std": float(np.std(val_accs)),
        "val_f1_macro_mean": float(np.mean(val_f1s)),
        "val_f1_macro_std": float(np.std(val_f1s)),
        "val_f1_weighted_mean": float(np.mean(val_f1w)),
        "val_f1_weighted_std": float(np.std(val_f1w)),
    }

    if eval_test:
        test_accs = [r["test_metrics"]["accuracy"] for r in fold_results if r["test_metrics"]]
        test_f1s = [r["test_metrics"]["f1_macro"] for r in fold_results if r["test_metrics"]]
        test_f1w = [r["test_metrics"]["f1_weighted"] for r in fold_results if r["test_metrics"]]
        summary.update({
            "test_accuracy_mean": float(np.mean(test_accs)) if test_accs else 0.0,
            "test_accuracy_std": float(np.std(test_accs)) if test_accs else 0.0,
            "test_f1_macro_mean": float(np.mean(test_f1s)) if test_f1s else 0.0,
            "test_f1_macro_std": float(np.std(test_f1s)) if test_f1s else 0.0,
            "test_f1_weighted_mean": float(np.mean(test_f1w)) if test_f1w else 0.0,
            "test_f1_weighted_std": float(np.std(test_f1w)) if test_f1w else 0.0,
        })

    summary_payload = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "backbone": backbone,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "use_class_weights": use_class_weights,
        "patience": patience,
        "device": str(torch_device),
        "seed": seed,
        "folds": folds,
        "eval_test": eval_test,
        "class_names": class_names,
        "train_val_samples": len(full_samples),
        "test_samples": len(test_dataset) if test_dataset else None,
        "fold_results": fold_results,
        "summary": summary,
    }

    summary_path = output_path / "cross_validation_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2)

    logger.info("-" * 40)
    logger.info("RESUMEN CV")
    logger.info("-" * 40)
    logger.info(
        "Val Accuracy: %.4f +/- %.4f",
        summary["val_accuracy_mean"],
        summary["val_accuracy_std"],
    )
    logger.info(
        "Val F1 Macro: %.4f +/- %.4f",
        summary["val_f1_macro_mean"],
        summary["val_f1_macro_std"],
    )
    logger.info(
        "Val F1 Weighted: %.4f +/- %.4f",
        summary["val_f1_weighted_mean"],
        summary["val_f1_weighted_std"],
    )
    if eval_test:
        logger.info(
            "Test Accuracy: %.4f +/- %.4f",
            summary["test_accuracy_mean"],
            summary["test_accuracy_std"],
        )
        logger.info(
            "Test F1 Macro: %.4f +/- %.4f",
            summary["test_f1_macro_mean"],
            summary["test_f1_macro_std"],
        )

    logger.info("Resumen guardado en: %s", summary_path)
    logger.info("=" * 60)
    logger.info("Cross-validation completada!")
    logger.info("=" * 60)


@app.command("cross-evaluate")
def cross_evaluate(
    model_a: str = typer.Argument(
        ...,
        help="Path al checkpoint del primer modelo (ej: modelo original)"
    ),
    model_b: str = typer.Argument(
        ...,
        help="Path al checkpoint del segundo modelo (ej: modelo warped)"
    ),
    data_a: str = typer.Option(
        ...,
        "--data-a",
        "-a",
        help="Directorio del primer dataset (ej: dataset original)"
    ),
    data_b: str = typer.Option(
        ...,
        "--data-b",
        "-B",
        help="Directorio del segundo dataset (ej: dataset warped)"
    ),
    split: str = typer.Option(
        "test",
        "--split",
        "-s",
        help="Split a evaluar: test, val, all"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directorio de salida para resultados JSON y analisis"
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        help="Tamano de batch"
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo: auto, cuda, cpu, mps"
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Semilla para reproducibilidad del split"
    ),
):
    """
    Evaluacion cruzada de dos modelos en dos datasets.

    Ejecuta matriz completa de evaluacion:
    - Modelo A en Dataset A (baseline A)
    - Modelo A en Dataset B (cross A→B)
    - Modelo B en Dataset B (baseline B)
    - Modelo B en Dataset A (cross B→A)

    Calcula gaps de generalizacion para medir cual modelo generaliza mejor.

    Ejemplo para reproducir Session 30 (Original vs Warped):

        python -m src_v2 cross-evaluate \\
            outputs/classifier_original/best.pt \\
            outputs/classifier_warped/best.pt \\
            --data-a data/dataset/COVID-19_Radiography_Dataset \\
            --data-b outputs/full_warped_dataset \\
            --output-dir outputs/cross_evaluation
    """
    import json
    from collections import defaultdict
    from datetime import datetime

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import datasets
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from tqdm import tqdm

    from src_v2.models import create_classifier, get_classifier_transforms

    logger.info("=" * 60)
    logger.info("Cross-Evaluation: Modelo A vs Modelo B")
    logger.info("=" * 60)

    # Configurar semilla completa para reproducibilidad
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Verificar paths
    for path, name in [(model_a, "Modelo A"), (model_b, "Modelo B")]:
        if not Path(path).exists():
            logger.error("%s no existe: %s", name, path)
            raise typer.Exit(code=1)

    for path, name in [(data_a, "Dataset A"), (data_b, "Dataset B")]:
        if not Path(path).exists():
            logger.error("%s no existe: %s", name, path)
            raise typer.Exit(code=1)

    # Dispositivo
    torch_device = get_device(device)
    logger.info("Dispositivo: %s", torch_device)

    # Cargar modelos
    logger.info("Cargando Modelo A: %s", model_a)
    classifier_a = create_classifier(checkpoint=model_a, device=torch_device)
    classifier_a.eval()

    logger.info("Cargando Modelo B: %s", model_b)
    classifier_b = create_classifier(checkpoint=model_b, device=torch_device)
    classifier_b.eval()

    # Obtener class names del primer modelo
    ckpt_a = torch.load(model_a, map_location="cpu", weights_only=False)
    class_names = ckpt_a.get("class_names", ["COVID", "Normal", "Viral_Pneumonia"])
    logger.info("Clases: %s", class_names)

    # Transforms
    eval_transform = get_classifier_transforms(train=False, img_size=DEFAULT_IMAGE_SIZE)

    # Funcion helper para cargar dataset
    def load_dataset(data_dir: str, split_name: str):
        """Carga dataset de un directorio."""
        data_path = Path(data_dir)

        # Detectar estructura: con splits (train/val/test) o plana
        if (data_path / split_name).exists():
            # Estructura con splits
            split_dir = data_path / split_name
            return datasets.ImageFolder(split_dir, transform=eval_transform)
        elif (data_path / "test").exists():
            # Usar test por defecto si existe
            return datasets.ImageFolder(data_path / "test", transform=eval_transform)
        else:
            # Estructura plana: buscar subdirectorios de clases
            # Verificar si hay carpetas de clases directamente
            class_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name in class_names]
            if class_dirs:
                return datasets.ImageFolder(data_path, transform=eval_transform)
            else:
                # Buscar en subcarpeta 'images' (estructura COVID-19 Radiography)
                # Crear dataset manualmente
                image_paths = []
                for class_name in class_names:
                    class_dir = data_path / class_name / "images"
                    if not class_dir.exists():
                        class_dir = data_path / class_name
                    if class_dir.exists():
                        for img_path in class_dir.glob("*.png"):
                            image_paths.append((img_path, class_names.index(class_name)))

                class OriginalDataset(Dataset):
                    def __init__(self, paths, transform):
                        self.paths = paths
                        self.transform = transform
                        self.classes = class_names

                    def __len__(self):
                        return len(self.paths)

                    def __getitem__(self, idx):
                        path, label = self.paths[idx]
                        from PIL import Image
                        img = Image.open(path)
                        if self.transform:
                            img = self.transform(img)
                        return img, label

                return OriginalDataset(image_paths, eval_transform)

    # Cargar datasets
    logger.info("Cargando Dataset A: %s (split=%s)", data_a, split)
    dataset_a = load_dataset(data_a, split)
    logger.info("  Muestras: %d", len(dataset_a))

    logger.info("Cargando Dataset B: %s (split=%s)", data_b, split)
    dataset_b = load_dataset(data_b, split)
    logger.info("  Muestras: %d", len(dataset_b))

    loader_a = DataLoader(dataset_a, batch_size=batch_size, shuffle=False, num_workers=get_optimal_num_workers())
    loader_b = DataLoader(dataset_b, batch_size=batch_size, shuffle=False, num_workers=get_optimal_num_workers())

    def evaluate_model_on_loader(model, dataloader, desc="Evaluando"):
        """Evalua modelo y retorna metricas."""
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc=desc, leave=False):
                inputs = inputs.to(torch_device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        accuracy = accuracy_score(all_labels, all_preds) * 100
        f1_macro = f1_score(all_labels, all_preds, average='macro') * 100
        cm = confusion_matrix(all_labels, all_preds)

        # Accuracy por clase
        class_accuracies = {}
        for idx, name in enumerate(class_names):
            if idx < len(cm) and cm[idx].sum() > 0:
                class_acc = cm[idx, idx] / cm[idx].sum() * 100
                class_accuracies[name] = float(class_acc)

        return {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'confusion_matrix': cm.tolist(),
            'class_accuracies': class_accuracies,
            'n_samples': len(all_labels),
            'n_errors': int(sum(1 for p, l in zip(all_preds, all_labels) if p != l))
        }

    # Ejecutar matriz de evaluacion
    logger.info("-" * 60)
    logger.info("Ejecutando matriz de cross-evaluation...")
    logger.info("-" * 60)

    results = {}

    logger.info("[1/4] Modelo A en Dataset A (baseline)")
    results['a_on_a'] = evaluate_model_on_loader(classifier_a, loader_a, "A→A")
    logger.info("  Accuracy: %.2f%%", results['a_on_a']['accuracy'])

    logger.info("[2/4] Modelo A en Dataset B (cross)")
    results['a_on_b'] = evaluate_model_on_loader(classifier_a, loader_b, "A→B")
    logger.info("  Accuracy: %.2f%%", results['a_on_b']['accuracy'])

    logger.info("[3/4] Modelo B en Dataset B (baseline)")
    results['b_on_b'] = evaluate_model_on_loader(classifier_b, loader_b, "B→B")
    logger.info("  Accuracy: %.2f%%", results['b_on_b']['accuracy'])

    logger.info("[4/4] Modelo B en Dataset A (cross)")
    results['b_on_a'] = evaluate_model_on_loader(classifier_b, loader_a, "B→A")
    logger.info("  Accuracy: %.2f%%", results['b_on_a']['accuracy'])

    # Calcular gaps de generalizacion
    gap_a = results['a_on_a']['accuracy'] - results['a_on_b']['accuracy']
    gap_b = results['b_on_b']['accuracy'] - results['b_on_a']['accuracy']
    ratio = abs(gap_a / gap_b) if gap_b != 0 else float('inf')

    # Mostrar resumen
    logger.info("=" * 60)
    logger.info("RESULTADOS CROSS-EVALUATION")
    logger.info("=" * 60)

    logger.info("\nMatriz de Evaluacion:")
    logger.info("                    Dataset A      Dataset B")
    logger.info("  Modelo A          %.2f%%         %.2f%%", results['a_on_a']['accuracy'], results['a_on_b']['accuracy'])
    logger.info("  Modelo B          %.2f%%         %.2f%%", results['b_on_a']['accuracy'], results['b_on_b']['accuracy'])

    logger.info("\nGaps de Generalizacion:")
    logger.info("  Gap Modelo A (A→A minus A→B): %.2f%%", gap_a)
    logger.info("  Gap Modelo B (B→B minus B→A): %.2f%%", gap_b)

    if gap_b > 0 and gap_a > 0:
        logger.info("\n  RATIO: Modelo A gap es %.1fx mayor que Modelo B", ratio)
        if gap_a > gap_b:
            logger.info("  CONCLUSION: Modelo B GENERALIZA MEJOR")
        else:
            logger.info("  CONCLUSION: Modelo A GENERALIZA MEJOR")

    # Guardar resultados
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        full_results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'model_a': model_a,
                'model_b': model_b,
                'data_a': data_a,
                'data_b': data_b,
                'split': split,
                'seed': seed,
            },
            'results': results,
            'gaps': {
                'gap_a': float(gap_a),
                'gap_b': float(gap_b),
                'ratio': float(ratio) if ratio != float('inf') else 'inf',
                'better_generalizer': 'B' if gap_a > gap_b else 'A',
            },
            'class_names': class_names,
        }

        results_file = output_path / "cross_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(full_results, f, indent=2)
        logger.info("\nResultados guardados en: %s", results_file)

    logger.info("=" * 60)
    logger.info("Cross-evaluation completada!")
    logger.info("=" * 60)


@app.command("evaluate-external")
def evaluate_external(
    checkpoint: str = typer.Argument(
        ...,
        help="Path al checkpoint del clasificador (entrenado con 3 clases)"
    ),
    external_data: str = typer.Option(
        ...,
        "--external-data",
        "-e",
        help="Directorio del dataset externo (estructura: test/positive, test/negative)"
    ),
    output_json: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Guardar resultados en JSON"
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        help="Tamano de batch"
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo: auto, cuda, cpu, mps"
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        "-t",
        help="Umbral de decision para clase positiva (COVID)"
    ),
    # CLAHE preprocessing (debe coincidir con entrenamiento)
    use_clahe: bool = typer.Option(
        True,
        "--clahe/--no-clahe",
        help="Aplicar CLAHE (debe coincidir con preprocesamiento del modelo)"
    ),
    clahe_clip: float = typer.Option(
        DEFAULT_CLAHE_CLIP_LIMIT,
        "--clahe-clip",
        help="CLAHE clip limit"
    ),
    clahe_tile: int = typer.Option(
        DEFAULT_CLAHE_TILE_SIZE,
        "--clahe-tile",
        help="CLAHE tile size"
    ),
):
    """
    Evaluar clasificador en dataset externo binario (FedCOVIDx/Dataset3).

    El modelo de 3 clases (COVID, Normal, Viral_Pneumonia) se evalua
    en un dataset binario (positive/negative) usando mapeo:
    - P(positive) = P(COVID)
    - P(negative) = P(Normal) + P(Viral_Pneumonia)

    Estructura esperada del dataset externo:
        external_data/
        └── test/
            ├── positive/   # COVID
            └── negative/   # No-COVID

    Metricas calculadas:
    - Accuracy, Sensitivity (recall COVID), Specificity
    - Precision, F1-score, AUC-ROC

    Ejemplo:

        python -m src_v2 evaluate-external \\
            outputs/classifier/best.pt \\
            --external-data outputs/external_validation/dataset3 \\
            --output results.json
    """
    import json
    from datetime import datetime

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )
    from tqdm import tqdm

    import cv2
    from src_v2.models import create_classifier, get_classifier_transforms

    logger.info("=" * 60)
    logger.info("Evaluacion Externa - Dataset Binario")
    logger.info("=" * 60)
    logger.info("Preprocesamiento: CLAHE=%s (clip=%.1f, tile=%d)", use_clahe, clahe_clip, clahe_tile)

    # Verificar paths
    if not Path(checkpoint).exists():
        logger.error("Checkpoint no existe: %s", checkpoint)
        raise typer.Exit(code=1)

    external_path = Path(external_data)
    if not external_path.exists():
        logger.error("Dataset externo no existe: %s", external_data)
        raise typer.Exit(code=1)

    # Buscar directorio test
    test_dir = external_path / "test"
    if not test_dir.exists():
        test_dir = external_path  # Usar raiz si no hay subcarpeta test

    positive_dir = test_dir / "positive"
    negative_dir = test_dir / "negative"

    if not positive_dir.exists() or not negative_dir.exists():
        logger.error("Estructura invalida. Se esperan carpetas 'positive' y 'negative' en: %s", test_dir)
        raise typer.Exit(code=1)

    # Dispositivo
    torch_device = get_device(device)
    logger.info("Dispositivo: %s", torch_device)

    # Cargar modelo
    logger.info("Cargando modelo: %s", checkpoint)
    model = create_classifier(checkpoint=checkpoint, device=torch_device)
    model.eval()

    # Obtener class names del modelo
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    class_names_3 = ckpt.get("class_names", ["COVID", "Normal", "Viral_Pneumonia"])
    logger.info("Clases del modelo: %s", class_names_3)

    # Verificar que COVID sea la primera clase (indice 0)
    covid_idx = 0
    if "COVID" in class_names_3:
        covid_idx = class_names_3.index("COVID")
    logger.info("Indice de COVID: %d", covid_idx)

    # Transforms
    eval_transform = get_classifier_transforms(train=False, img_size=DEFAULT_IMAGE_SIZE)

    # Crear CLAHE processor si está habilitado
    clahe_processor = None
    if use_clahe:
        clahe_processor = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))

    # Dataset externo binario con soporte CLAHE
    class ExternalBinaryDataset(Dataset):
        def __init__(self, pos_dir, neg_dir, transform, clahe_proc=None):
            self.samples = []
            self.transform = transform
            self.clahe_proc = clahe_proc

            # Cargar positivos (COVID)
            for img_path in pos_dir.glob("*.png"):
                self.samples.append((img_path, 1))
            for img_path in pos_dir.glob("*.jpg"):
                self.samples.append((img_path, 1))

            # Cargar negativos (No-COVID)
            for img_path in neg_dir.glob("*.png"):
                self.samples.append((img_path, 0))
            for img_path in neg_dir.glob("*.jpg"):
                self.samples.append((img_path, 0))

            self.n_positive = sum(1 for _, l in self.samples if l == 1)
            self.n_negative = len(self.samples) - self.n_positive

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]

            # Cargar imagen en grayscale para CLAHE
            img_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                # Fallback a PIL si cv2 falla
                img = Image.open(path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
            else:
                # Aplicar CLAHE si está habilitado
                if self.clahe_proc is not None:
                    img_gray = self.clahe_proc.apply(img_gray)
                # Convertir a RGB para el modelo
                img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
                img = Image.fromarray(img_rgb)

            if self.transform:
                img = self.transform(img)
            return img, label

    # Crear dataset
    dataset = ExternalBinaryDataset(positive_dir, negative_dir, eval_transform, clahe_processor)
    logger.info("Dataset externo: %d muestras (positive=%d, negative=%d)",
                len(dataset), dataset.n_positive, dataset.n_negative)

    if len(dataset) == 0:
        logger.error("Dataset vacio")
        raise typer.Exit(code=1)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=get_optimal_num_workers())

    # Evaluar con mapeo 3→2 clases
    all_probs_positive = []
    all_labels = []

    logger.info("Evaluando con mapeo 3→2 clases...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluando"):
            inputs = inputs.to(torch_device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)

            # Mapeo: P(positive) = P(COVID)
            prob_positive = probs[:, covid_idx]

            all_probs_positive.extend(prob_positive.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs_positive = np.array(all_probs_positive)
    all_labels = np.array(all_labels)

    # Predicciones binarias
    predictions = (all_probs_positive >= threshold).astype(int)

    # Metricas
    accuracy = accuracy_score(all_labels, predictions)
    sensitivity = recall_score(all_labels, predictions, pos_label=1)  # Recall de COVID
    specificity = recall_score(all_labels, predictions, pos_label=0)  # Recall de No-COVID
    precision = precision_score(all_labels, predictions, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels, predictions, pos_label=1)

    # AUC-ROC
    try:
        auc_roc = roc_auc_score(all_labels, all_probs_positive)
    except ValueError:
        auc_roc = 0.0

    cm = confusion_matrix(all_labels, predictions)

    # Mostrar resultados
    logger.info("-" * 60)
    logger.info("RESULTADOS - Evaluacion Externa Binaria")
    logger.info("-" * 60)
    logger.info("Umbral de decision: %.2f", threshold)
    logger.info("")
    logger.info("Accuracy:     %.2f%% (%.4f)", accuracy * 100, accuracy)
    logger.info("Sensitivity:  %.2f%% (Recall COVID)", sensitivity * 100)
    logger.info("Specificity:  %.2f%% (Recall No-COVID)", specificity * 100)
    logger.info("Precision:    %.2f%%", precision * 100)
    logger.info("F1-Score:     %.2f%%", f1 * 100)
    logger.info("AUC-ROC:      %.4f", auc_roc)
    logger.info("")
    logger.info("Confusion Matrix:")
    logger.info("                 Pred Neg    Pred Pos")
    logger.info("  Actual Neg     %5d       %5d", cm[0, 0], cm[0, 1])
    logger.info("  Actual Pos     %5d       %5d", cm[1, 0], cm[1, 1])

    # Guardar JSON
    if output_json:
        results = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint': checkpoint,
            'external_data': external_data,
            'preprocessing': {
                'clahe_enabled': use_clahe,
                'clahe_clip_limit': clahe_clip if use_clahe else None,
                'clahe_tile_size': clahe_tile if use_clahe else None,
            },
            'class_mapping': {
                'model_classes': class_names_3,
                'positive_mapped_to': 'COVID',
                'negative_mapped_to': 'Normal + Viral_Pneumonia',
            },
            'threshold': threshold,
            'n_samples': len(all_labels),
            'n_positive': int(dataset.n_positive),
            'n_negative': int(dataset.n_negative),
            'metrics': {
                'accuracy': float(accuracy),
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'precision': float(precision),
                'f1_score': float(f1),
                'auc_roc': float(auc_roc),
            },
            'confusion_matrix': cm.tolist(),
        }

        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info("\nResultados guardados en: %s", output_json)

    logger.info("=" * 60)
    logger.info("Evaluacion externa completada!")
    logger.info("=" * 60)


@app.command("test-robustness")
def test_robustness(
    checkpoint: str = typer.Argument(
        ...,
        help="Path al checkpoint del clasificador"
    ),
    data_dir: str = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Directorio del dataset de prueba"
    ),
    split: str = typer.Option(
        "test",
        "--split",
        "-s",
        help="Split a evaluar: test, val"
    ),
    output_json: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Guardar resultados en JSON"
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        help="Tamano de batch"
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo: auto, cuda, cpu, mps"
    ),
):
    """
    Evaluar robustez del clasificador ante perturbaciones.

    Aplica diferentes perturbaciones a las imagenes y mide la degradacion:
    - JPEG compression (Q=50, Q=30)
    - Gaussian blur (sigma=1.0, sigma=2.0)
    - Gaussian noise (sigma=0.05, sigma=0.10)

    Util para comparar modelos entrenados en datos originales vs warped.

    Ejemplo:

        python -m src_v2 test-robustness \\
            outputs/classifier/best.pt \\
            --data-dir outputs/full_warped_dataset \\
            --output robustness_results.json
    """
    import io
    import json
    from datetime import datetime

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from sklearn.metrics import accuracy_score, f1_score
    from tqdm import tqdm

    from src_v2.models import create_classifier, get_classifier_transforms
    from src_v2.models.classifier import GrayscaleToRGB

    logger.info("=" * 60)
    logger.info("Test de Robustez - Perturbaciones")
    logger.info("=" * 60)

    # Verificar paths
    if not Path(checkpoint).exists():
        logger.error("Checkpoint no existe: %s", checkpoint)
        raise typer.Exit(code=1)

    data_path = Path(data_dir) / split
    if not data_path.exists():
        logger.error("Directorio de datos no existe: %s", data_path)
        raise typer.Exit(code=1)

    # Dispositivo
    torch_device = get_device(device)
    logger.info("Dispositivo: %s", torch_device)

    # Cargar modelo
    logger.info("Cargando modelo: %s", checkpoint)
    model = create_classifier(checkpoint=checkpoint, device=torch_device)
    model.eval()

    # Class names
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    class_names = ckpt.get("class_names", ["COVID", "Normal", "Viral_Pneumonia"])

    # Definir perturbaciones
    perturbations = {
        'original': None,
        'jpeg_q50': lambda img: apply_jpeg_compression(img, quality=50),
        'jpeg_q30': lambda img: apply_jpeg_compression(img, quality=30),
        'blur_sigma1': lambda img: apply_gaussian_blur(img, sigma=1.0),
        'blur_sigma2': lambda img: apply_gaussian_blur(img, sigma=2.0),
        'noise_005': lambda img: apply_gaussian_noise(img, sigma=0.05),
        'noise_010': lambda img: apply_gaussian_noise(img, sigma=0.10),
    }

    def apply_jpeg_compression(img, quality):
        """Aplica compresion JPEG."""
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')

    def apply_gaussian_blur(img, sigma):
        """Aplica blur gaussiano."""
        img_array = np.array(img)
        blurred = cv2.GaussianBlur(img_array, (0, 0), sigma)
        return Image.fromarray(blurred)

    def apply_gaussian_noise(img, sigma):
        """Agrega ruido gaussiano."""
        img_array = np.array(img).astype(np.float32) / 255.0
        noise = np.random.randn(*img_array.shape) * sigma
        noisy = np.clip(img_array + noise, 0, 1)
        return Image.fromarray((noisy * 255).astype(np.uint8))

    # Base transform (sin perturbacion)
    from torchvision import transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    base_transform = transforms.Compose([
        GrayscaleToRGB(),
        transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    # Cargar dataset base (sin transforms especiales)
    base_dataset = datasets.ImageFolder(data_path)
    logger.info("Dataset: %d muestras", len(base_dataset))

    # Evaluar cada perturbacion
    results = {}

    for pert_name, pert_fn in perturbations.items():
        logger.info("Evaluando: %s", pert_name)

        all_preds = []
        all_labels = []

        # Evaluar batch por batch
        for idx in tqdm(range(0, len(base_dataset), batch_size), desc=pert_name, leave=False):
            batch_images = []
            batch_labels = []

            for i in range(idx, min(idx + batch_size, len(base_dataset))):
                img_path, label = base_dataset.samples[i]
                img = Image.open(img_path)

                # Aplicar perturbacion si existe
                if pert_fn is not None:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = pert_fn(img)

                # Aplicar transform base
                img_tensor = base_transform(img)
                batch_images.append(img_tensor)
                batch_labels.append(label)

            # Stack y evaluar
            batch_tensor = torch.stack(batch_images).to(torch_device)

            with torch.no_grad():
                outputs = model(batch_tensor)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels)

        # Calcular metricas
        accuracy = accuracy_score(all_labels, all_preds) * 100
        f1_macro = f1_score(all_labels, all_preds, average='macro') * 100
        error_rate = 100 - accuracy

        results[pert_name] = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'error_rate': float(error_rate),
            'n_samples': len(all_labels),
        }

        logger.info("  Accuracy: %.2f%%, Error: %.2f%%", accuracy, error_rate)

    # Calcular degradaciones relativas al original
    baseline_acc = results['original']['accuracy']

    logger.info("-" * 60)
    logger.info("RESUMEN DE ROBUSTEZ")
    logger.info("-" * 60)
    logger.info("%-15s %10s %10s %12s", "Perturbacion", "Accuracy", "Error", "Degradacion")
    logger.info("-" * 60)

    for name, metrics in results.items():
        degradation = baseline_acc - metrics['accuracy']
        results[name]['degradation'] = float(degradation)
        logger.info("%-15s %9.2f%% %9.2f%% %+11.2f%%",
                   name, metrics['accuracy'], metrics['error_rate'], degradation)

    # Guardar JSON
    if output_json:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint': checkpoint,
            'data_dir': data_dir,
            'split': split,
            'baseline_accuracy': float(baseline_acc),
            'perturbations': results,
            'class_names': class_names,
        }

        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info("\nResultados guardados en: %s", output_json)

    logger.info("=" * 60)
    logger.info("Test de robustez completado!")
    logger.info("=" * 60)


@app.command("compute-canonical")
def compute_canonical(
    landmarks_csv: str = typer.Argument(
        ...,
        help="Path al CSV con coordenadas de landmarks (formato: idx,x1,y1,...,x15,y15,image_name)"
    ),
    output_dir: str = typer.Option(
        "outputs/shape_analysis",
        "--output-dir",
        "-o",
        help="Directorio de salida para archivos JSON"
    ),
    visualize: bool = typer.Option(
        False,
        "--visualize/--no-visualize",
        help="Generar visualizaciones de forma canonica"
    ),
    max_iterations: int = typer.Option(
        100,
        "--max-iterations",
        help="Maximo de iteraciones para GPA"
    ),
    tolerance: float = typer.Option(
        1e-8,
        "--tolerance",
        help="Tolerancia para convergencia de GPA"
    ),
    image_size: int = typer.Option(
        224,
        "--image-size",
        help="Tamano de imagen para escalar forma canonica"
    ),
    padding: float = typer.Option(
        0.1,
        "--padding",
        help="Padding relativo al escalar a imagen (0.1 = 10%%)"
    ),
):
    """
    Calcular forma canonica de landmarks usando Generalized Procrustes Analysis (GPA).

    Este comando:
    1. Carga coordenadas de landmarks desde un CSV
    2. Ejecuta GPA iterativo para encontrar forma consenso
    3. Escala la forma canonica a coordenadas de imagen
    4. Calcula triangulacion de Delaunay
    5. Guarda canonical_shape_gpa.json y canonical_delaunay_triangles.json

    El formato del CSV debe tener columnas:
    - Primera columna: indice
    - Columnas 1-30: coordenadas x1,y1,x2,y2,...,x15,y15
    - Ultima columna: nombre de imagen (opcional)

    Ejemplo:
        python -m src_v2 compute-canonical data/coordenadas/coordenadas_maestro.csv -o outputs/shape_analysis --visualize
    """
    import json
    import numpy as np
    import pandas as pd

    from src_v2.processing.gpa import (
        gpa_iterative,
        scale_canonical_to_image,
        compute_delaunay_triangulation,
    )

    logger.info("=" * 60)
    logger.info("COVID-19 Landmark Detection - Compute Canonical Shape")
    logger.info("=" * 60)

    # Verificar CSV
    csv_path = Path(landmarks_csv)
    if not csv_path.exists():
        logger.error("CSV no existe: %s", landmarks_csv)
        raise typer.Exit(code=1)

    # Crear directorio de salida
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Cargar CSV
    logger.info("Cargando coordenadas desde: %s", landmarks_csv)
    df = pd.read_csv(csv_path, header=None)

    # Determinar formato del CSV
    # El CSV tiene formato: idx, x1, y1, x2, y2, ..., x15, y15, image_name
    # Total: 1 + 30 + 1 = 32 columnas
    n_cols = len(df.columns)
    logger.info("CSV tiene %d columnas", n_cols)

    # Extraer coordenadas (columnas 1-30 son las coordenadas)
    coord_cols = list(range(1, 31))  # Columnas 1 a 30
    coords = df.iloc[:, coord_cols].values

    # Reshape a (n_samples, 15, 2) - IMPORTANTE: convertir a float64 para operaciones GPA
    n_samples = coords.shape[0]
    landmarks = coords.reshape(n_samples, 15, 2).astype(np.float64)

    logger.info("Formas cargadas: %d muestras, %d landmarks", n_samples, 15)

    # Ejecutar GPA
    logger.info("Ejecutando GPA iterativo...")
    canonical_shape, aligned_shapes, convergence_info = gpa_iterative(
        landmarks,
        max_iterations=max_iterations,
        tolerance=tolerance,
        verbose=True
    )

    logger.info("Convergencia en %d iteraciones", convergence_info['n_iterations'])
    logger.info("Convergido: %s", convergence_info['converged'])

    # Escalar a coordenadas de imagen
    logger.info("Escalando a imagen %dx%d (padding=%.1f%%)", image_size, image_size, padding * 100)
    canonical_shape_pixels = scale_canonical_to_image(
        canonical_shape,
        image_size=image_size,
        padding=padding
    )

    # Calcular triangulacion Delaunay
    logger.info("Calculando triangulacion de Delaunay...")
    triangles = compute_delaunay_triangulation(canonical_shape_pixels)
    logger.info("Triangulos generados: %d", len(triangles))

    # Nombres de landmarks
    landmark_names = [
        "L1 (Superior)", "L2 (Inferior)", "L3 (Apex Izq)", "L4 (Apex Der)",
        "L5 (Hilio Izq)", "L6 (Hilio Der)", "L7 (Base Izq)", "L8 (Base Der)",
        "L9 (Centro Sup)", "L10 (Centro Med)", "L11 (Centro Inf)",
        "L12 (Borde Sup Izq)", "L13 (Borde Sup Der)",
        "L14 (Costofrenico Izq)", "L15 (Costofrenico Der)"
    ]

    # Guardar forma canonica
    canonical_data = {
        'canonical_shape_normalized': canonical_shape.tolist(),
        'canonical_shape_pixels': canonical_shape_pixels.tolist(),
        'image_size': image_size,
        'n_landmarks': 15,
        'landmark_names': landmark_names,
        'convergence': {
            'n_iterations': int(convergence_info['n_iterations']),
            'converged': bool(convergence_info['converged']),
            'final_change': float(convergence_info['final_change']),
            'n_shapes_used': int(convergence_info['n_shapes'])
        },
        'method': 'Generalized Procrustes Analysis (GPA)',
    }

    canonical_path = output_path / "canonical_shape_gpa.json"
    with open(canonical_path, 'w') as f:
        json.dump(canonical_data, f, indent=2)
    logger.info("Forma canonica guardada en: %s", canonical_path)

    # Guardar triangulacion
    triangles_data = {
        'num_triangles': len(triangles),
        'triangles': triangles.tolist(),
        'canonical_landmarks': canonical_shape_pixels.tolist(),
        'method': 'GPA + Delaunay',
        'description': 'Triangulacion Delaunay sobre forma canonica GPA'
    }

    triangles_path = output_path / "canonical_delaunay_triangles.json"
    with open(triangles_path, 'w') as f:
        json.dump(triangles_data, f, indent=2)
    logger.info("Triangulacion guardada en: %s", triangles_path)

    # Guardar formas alineadas (opcional, para analisis posterior)
    aligned_path = output_path / "aligned_shapes.npz"
    np.savez(
        aligned_path,
        aligned_shapes=aligned_shapes,
        canonical_shape=canonical_shape,
        canonical_shape_pixels=canonical_shape_pixels,
    )
    logger.info("Formas alineadas guardadas en: %s", aligned_path)

    # Visualizacion opcional
    if visualize:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            figures_dir = output_path / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)

            # Visualizar forma canonica
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(canonical_shape_pixels[:, 0], canonical_shape_pixels[:, 1],
                       c='blue', s=100, zorder=5)

            for i, (x, y) in enumerate(canonical_shape_pixels):
                ax.annotate(f'L{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold')

            # Dibujar triangulacion
            for tri in triangles:
                pts = canonical_shape_pixels[tri]
                pts = np.vstack([pts, pts[0]])  # Cerrar triangulo
                ax.plot(pts[:, 0], pts[:, 1], 'g-', alpha=0.3, linewidth=0.5)

            ax.set_xlim(0, image_size)
            ax.set_ylim(image_size, 0)  # Invertir Y
            ax.set_aspect('equal')
            ax.set_title('Forma Canonica (GPA) + Delaunay', fontsize=14)
            ax.grid(True, alpha=0.3)

            fig_path = figures_dir / "canonical_shape.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("Visualizacion guardada en: %s", fig_path)

            # Grafico de convergencia
            if convergence_info['distances_history']:
                fig, ax = plt.subplots(figsize=(10, 6))
                distances = convergence_info['distances_history']
                ax.plot(distances, 'b-', linewidth=2, marker='o', markersize=4)
                ax.set_xlabel('Iteracion', fontsize=12)
                ax.set_ylabel('Distancia promedio al consenso', fontsize=12)
                ax.set_title('Convergencia del GPA', fontsize=14)
                ax.grid(True, alpha=0.3)

                conv_path = figures_dir / "gpa_convergence.png"
                plt.savefig(conv_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info("Grafico de convergencia guardado en: %s", conv_path)

        except ImportError:
            logger.warning("matplotlib no disponible, saltando visualizaciones")

    # Resumen
    logger.info("-" * 40)
    logger.info("RESUMEN")
    logger.info("-" * 40)
    logger.info("Formas procesadas: %d", n_samples)
    logger.info("Iteraciones GPA: %d", convergence_info['n_iterations'])
    logger.info("Triangulos Delaunay: %d", len(triangles))
    logger.info("Archivos generados:")
    logger.info("  - %s", canonical_path)
    logger.info("  - %s", triangles_path)
    logger.info("  - %s", aligned_path)
    if visualize:
        logger.info("  - %s/figures/", output_path)

    logger.info("=" * 60)
    logger.info("Calculo de forma canonica completado!")
    logger.info("=" * 60)


@app.command("generate-dataset")
def generate_dataset(
    input_dir: Optional[str] = typer.Argument(
        None,
        help="Directorio del dataset original (estructura COVID/Normal/Viral_Pneumonia/images/)"
    ),
    output_dir: Optional[str] = typer.Argument(
        None,
        help="Directorio de salida para dataset warped"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path a un JSON con parametros de warping"
    ),
    checkpoint: Optional[str] = typer.Option(
        None,
        "--checkpoint",
        "-c",
        help="Path al checkpoint del modelo de landmarks (requerido si no hay --predictions/--ensemble-config)"
    ),
    predictions: Optional[str] = typer.Option(
        None,
        "--predictions",
        help="Archivo JSON/NPZ con landmarks predichos (usa cache, evita inferencia)"
    ),
    ensemble_config: Optional[str] = typer.Option(
        None,
        "--ensemble-config",
        help="Config JSON con lista de checkpoints para ensemble de landmarks"
    ),
    canonical_shape: str = typer.Option(
        "outputs/shape_analysis/canonical_shape_gpa.json",
        "--canonical",
        help="Path a la forma canonica (.json)"
    ),
    triangles: str = typer.Option(
        "outputs/shape_analysis/canonical_delaunay_triangles.json",
        "--triangles",
        help="Path a los triangulos de Delaunay (.json)"
    ),
    margin: float = typer.Option(
        1.05,
        "--margin",
        "-m",
        help="Factor de escala para margenes (1.05 = 5%% de expansion)"
    ),
    splits: str = typer.Option(
        "0.75,0.125,0.125",
        "--splits",
        help="Ratios train,val,test separados por coma"
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Semilla para reproducibilidad de splits"
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo: auto, cuda, cpu, mps"
    ),
    use_tta: bool = typer.Option(
        False,
        "--tta/--no-tta",
        help="Usar Test-Time Augmentation para prediccion de landmarks"
    ),
    use_clahe: bool = typer.Option(
        True,
        "--clahe/--no-clahe",
        help="Usar CLAHE para prediccion de landmarks"
    ),
    clahe_clip: float = typer.Option(
        DEFAULT_CLAHE_CLIP_LIMIT,
        "--clahe-clip",
        help="CLAHE clip limit"
    ),
    clahe_tile: int = typer.Option(
        DEFAULT_CLAHE_TILE_SIZE,
        "--clahe-tile",
        help="CLAHE tile size"
    ),
    classes: str = typer.Option(
        "COVID,Normal,Viral Pneumonia",
        "--classes",
        help="Clases a procesar separadas por coma"
    ),
    use_full_coverage: bool = typer.Option(
        True,
        "--use-full-coverage/--no-full-coverage",
        help="Agregar puntos de borde para cobertura completa (fill_rate ~99%)"
    ),
):
    """
    Generar dataset warped completo con splits train/val/test.

    Este comando:
    1. Carga imagenes del dataset original
    2. Predice landmarks con modelo/ensemble o usa cache precomputado
    3. Aplica warping con margen configurable
    4. Crea splits train/val/test estratificados
    5. Guarda metadata (landmarks.json, images.csv, dataset_summary.json)

    Estructura de salida:
        output_dir/
        ├── train/
        │   ├── COVID/
        │   ├── Normal/
        │   └── Viral_Pneumonia/
        ├── val/
        ├── test/
        ├── dataset_summary.json
        └── {split}/landmarks.json, images.csv

    Ejemplo:
        python -m src_v2 generate-dataset data/dataset/COVID-19_Radiography_Dataset outputs/warped_lung_best/session_warping --checkpoint checkpoints/model.pt --margin 1.05
        python -m src_v2 generate-dataset --config configs/warping_best.json
    """
    import json
    import time
    from collections import defaultdict

    import cv2
    import numpy as np
    import torch
    from tqdm import tqdm

    from src_v2.constants import IMAGENET_MEAN, IMAGENET_STD, SYMMETRIC_PAIRS
    from src_v2.models import create_model
    from src_v2.processing.warp import (
        piecewise_affine_warp,
        scale_landmarks_from_centroid,
        clip_landmarks_to_image,
        compute_fill_rate,
    )

    def normalize_pred_path(path_value: str, root: Path) -> str:
        path = Path(path_value)
        if path.is_absolute():
            try:
                return path.relative_to(root).as_posix()
            except ValueError:
                return path.as_posix()
        return path.as_posix()

    def load_predictions_cache(pred_path: Path, root: Path):
        metadata = {}
        by_path = {}
        by_name_category = {}

        def normalize_category(value: Optional[str]) -> Optional[str]:
            if isinstance(value, str):
                return value.replace(" ", "_")
            return value

        if pred_path.suffix.lower() == ".json":
            with open(pred_path, "r") as f:
                payload = json.load(f)
            metadata = payload.get("metadata", {})
            entries = payload.get("predictions", payload.get("items", []))
            for entry in entries:
                image_path = entry.get("image_path") or entry.get("image") or ""
                image_name = entry.get("image_name") or Path(image_path).stem
                category = normalize_category(entry.get("category") or entry.get("class"))
                landmarks = np.array(entry.get("landmarks", []), dtype=np.float32)
                if landmarks.shape == (NUM_LANDMARKS * 2,):
                    landmarks = landmarks.reshape(NUM_LANDMARKS, 2)
                if landmarks.shape != (NUM_LANDMARKS, 2):
                    continue
                if image_path:
                    rel_path = normalize_pred_path(image_path, root)
                    by_path[rel_path] = landmarks
                if image_name and category:
                    by_name_category[(image_name, category)] = landmarks
        elif pred_path.suffix.lower() == ".npz":
            data = np.load(pred_path, allow_pickle=True)
            if "metadata_json" in data:
                try:
                    metadata = json.loads(str(data["metadata_json"].item()))
                except json.JSONDecodeError:
                    metadata = {}
            elif "metadata" in data:
                try:
                    metadata = data["metadata"].item()
                except ValueError:
                    metadata = {}

            if "landmarks" not in data:
                raise ValueError("NPZ no contiene clave 'landmarks'")
            landmarks = data["landmarks"]
            if landmarks.shape[-1] == NUM_LANDMARKS * 2:
                landmarks = landmarks.reshape(-1, NUM_LANDMARKS, 2)
            if landmarks.shape[1:] != (NUM_LANDMARKS, 2):
                raise ValueError("Landmarks en NPZ con shape invalido")

            image_paths = data["image_paths"] if "image_paths" in data else None
            image_names = data["image_names"] if "image_names" in data else None
            categories = data["categories"] if "categories" in data else None

            if image_paths is None and image_names is None:
                raise ValueError("NPZ debe incluir image_paths o image_names")

            total = landmarks.shape[0]
            for idx in range(total):
                image_path = str(image_paths[idx]) if image_paths is not None else ""
                image_name = str(image_names[idx]) if image_names is not None else Path(image_path).stem
                category = normalize_category(str(categories[idx])) if categories is not None else None
                lm = landmarks[idx].astype(np.float32)
                if image_path:
                    rel_path = normalize_pred_path(image_path, root)
                    by_path[rel_path] = lm
                if image_name and category:
                    by_name_category[(image_name, category)] = lm
        else:
            raise ValueError("Formato de predicciones no soportado (usa .json o .npz)")

        return by_path, by_name_category, metadata

    logger.info("=" * 60)
    logger.info("COVID-19 Landmark Detection - Generate Warped Dataset")
    logger.info("=" * 60)

    argv = sys.argv[1:]

    def option_present(*names: str) -> bool:
        for name in names:
            if name in argv:
                return True
            prefix = f"{name}="
            if any(arg.startswith(prefix) for arg in argv):
                return True
        return False

    config_data = {}
    if config:
        config_path = Path(config)
        if not config_path.exists():
            logger.error("Config no existe: %s", config)
            raise typer.Exit(code=1)
        with open(config_path, "r") as f:
            config_data = json.load(f)

        def override_param(
            param_name: str,
            current_value,
            config_key: Optional[str] = None,
            option_names: Optional[tuple[str, ...]] = None,
        ):
            key = config_key or param_name
            if key not in config_data:
                return current_value
            if option_names and option_present(*option_names):
                return current_value
            if option_names is None and current_value is not None:
                return current_value
            return config_data[key]

        input_dir = override_param("input_dir", input_dir, "input_dir")
        output_dir = override_param("output_dir", output_dir, "output_dir")
        checkpoint = override_param("checkpoint", checkpoint, "checkpoint", ("--checkpoint", "-c"))
        predictions = override_param("predictions", predictions, "predictions", ("--predictions",))
        ensemble_config = override_param("ensemble_config", ensemble_config, "ensemble_config", ("--ensemble-config",))
        canonical_shape = override_param("canonical_shape", canonical_shape, "canonical", ("--canonical",))
        triangles = override_param("triangles", triangles, "triangles", ("--triangles",))
        margin = override_param("margin", margin, "margin", ("--margin", "-m"))
        splits = override_param("splits", splits, "splits", ("--splits",))
        seed = override_param("seed", seed, "seed", ("--seed", "-s"))
        device = override_param("device", device, "device", ("--device",))
        classes = override_param("classes", classes, "classes", ("--classes",))
        use_full_coverage = override_param(
            "use_full_coverage",
            use_full_coverage,
            "use_full_coverage",
            ("--use-full-coverage", "--no-full-coverage"),
        )
        use_tta = override_param("use_tta", use_tta, "tta", ("--tta", "--no-tta"))
        use_clahe = override_param("use_clahe", use_clahe, "clahe", ("--clahe", "--no-clahe"))
        clahe_clip = override_param("clahe_clip", clahe_clip, "clahe_clip", ("--clahe-clip",))
        clahe_tile = override_param("clahe_tile", clahe_tile, "clahe_tile", ("--clahe-tile",))

    if isinstance(splits, (list, tuple)):
        splits = ",".join(str(x) for x in splits)
    if isinstance(classes, (list, tuple)):
        classes = ",".join(str(x) for x in classes)

    if input_dir is None or output_dir is None:
        logger.error("input_dir y output_dir son requeridos (usa argumentos o --config)")
        raise typer.Exit(code=1)

    # Verificar paths
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error("Directorio de entrada no existe: %s", input_dir)
        raise typer.Exit(code=1)

    if not Path(canonical_shape).exists():
        logger.error("Forma canonica no existe: %s", canonical_shape)
        logger.info("Genera la forma canonica primero con: python -m src_v2 compute-canonical")
        raise typer.Exit(code=1)

    if not Path(triangles).exists():
        logger.error("Triangulos no existen: %s", triangles)
        raise typer.Exit(code=1)

    # Parsear splits
    try:
        split_ratios = [float(x) for x in splits.split(",")]
        if len(split_ratios) != 3:
            raise ValueError("Se requieren exactamente 3 valores")
        if abs(sum(split_ratios) - 1.0) > 0.01:
            raise ValueError(f"Los ratios deben sumar 1.0, suman {sum(split_ratios)}")
        train_ratio, val_ratio, test_ratio = split_ratios
    except Exception as e:
        logger.error("Error parseando splits '%s': %s", splits, e)
        raise typer.Exit(code=1)

    # Parsear clases
    class_list = [c.strip() for c in classes.split(",")]
    class_mapping = {c: c.replace(" ", "_") for c in class_list}
    logger.info("Clases a procesar: %s", class_list)

    predictions_path = Path(predictions) if predictions else None
    predictions_by_path = {}
    predictions_by_name_category = {}
    predictions_meta = {}
    missing_predictions = []
    missing_predictions_total = 0
    ensemble_meta = None
    checkpoint_paths = []

    if predictions_path:
        if not predictions_path.exists():
            logger.error("Predicciones no existen: %s", predictions_path)
            raise typer.Exit(code=1)
        try:
            predictions_by_path, predictions_by_name_category, predictions_meta = load_predictions_cache(
                predictions_path, input_path
            )
        except ValueError as exc:
            logger.error("Error cargando predicciones: %s", exc)
            raise typer.Exit(code=1)
        predictions_count = max(len(predictions_by_path), len(predictions_by_name_category))
        logger.info("Predicciones cargadas: %d", predictions_count)
        cached_input = predictions_meta.get("input_dir") if isinstance(predictions_meta, dict) else None
        if cached_input:
            try:
                if Path(cached_input).resolve() != input_path.resolve():
                    logger.warning("Predicciones generadas con input_dir=%s (actual=%s)",
                                   cached_input, input_path)
            except Exception:
                logger.warning("No se pudo validar input_dir en metadata de predicciones")
        if checkpoint or ensemble_config:
            logger.info("Ignorando --checkpoint/--ensemble-config porque se usa --predictions")
    else:
        if ensemble_config:
            ensemble_path = Path(ensemble_config)
            if not ensemble_path.exists():
                logger.error("Ensemble config no existe: %s", ensemble_path)
                raise typer.Exit(code=1)
            with open(ensemble_path, "r") as f:
                ensemble_meta = json.load(f)
            checkpoint_paths = ensemble_meta.get("models", [])
            if not isinstance(checkpoint_paths, list) or not checkpoint_paths:
                logger.error("Ensemble config debe incluir una lista 'models'")
                raise typer.Exit(code=1)
            if not option_present("--tta", "--no-tta") and "tta" in ensemble_meta:
                use_tta = bool(ensemble_meta["tta"])
            if not option_present("--clahe", "--no-clahe") and "clahe" in ensemble_meta:
                use_clahe = bool(ensemble_meta["clahe"])
            if not option_present("--clahe-clip") and "clahe_clip" in ensemble_meta:
                clahe_clip = float(ensemble_meta["clahe_clip"])
            if not option_present("--clahe-tile") and "clahe_tile" in ensemble_meta:
                clahe_tile = int(ensemble_meta["clahe_tile"])
        elif checkpoint:
            checkpoint_paths = [checkpoint]
        else:
            logger.error("Se requiere --checkpoint o --ensemble-config si no se usa --predictions")
            raise typer.Exit(code=1)

        for ckpt in checkpoint_paths:
            if not Path(ckpt).exists():
                logger.error("Checkpoint no existe: %s", ckpt)
                raise typer.Exit(code=1)

    # Cargar forma canonica y triangulos
    logger.info("Cargando forma canonica: %s", canonical_shape)
    with open(canonical_shape, 'r') as f:
        canonical_data = json.load(f)
    canonical = np.array(canonical_data['canonical_shape_pixels'])

    logger.info("Cargando triangulos: %s", triangles)
    with open(triangles, 'r') as f:
        tri_data = json.load(f)
    tri = np.array(tri_data['triangles'])

    logger.info("Forma canonica: %s", canonical.shape)
    logger.info("Triangulos: %s", tri.shape)

    # Crear directorio de salida
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch_device = None
    landmark_models = []
    if not predictions_path:
        torch_device = get_device(device)
        logger.info("Usando dispositivo: %s", torch_device)

        logger.info("Cargando %d modelo(s) de landmarks", len(checkpoint_paths))
        for ckpt in checkpoint_paths:
            logger.info("  - %s", ckpt)
            checkpoint_data = torch.load(ckpt, map_location=torch_device, weights_only=False)

            if "model_state_dict" in checkpoint_data:
                state_dict = checkpoint_data["model_state_dict"]
            else:
                state_dict = checkpoint_data

            arch_params = detect_architecture_from_checkpoint(state_dict)
            logger.info("Arquitectura: coord_attention=%s, deep_head=%s, hidden_dim=%d",
                        arch_params["use_coord_attention"], arch_params["deep_head"], arch_params["hidden_dim"])

            model = create_model(
                pretrained=False,
                use_coord_attention=arch_params["use_coord_attention"],
                deep_head=arch_params["deep_head"],
                hidden_dim=arch_params["hidden_dim"],
            )
            model.load_state_dict(state_dict)
            model = model.to(torch_device)
            model.eval()
            landmark_models.append(model)

        logger.info("TTA: %s", "habilitado" if use_tta else "deshabilitado")

    # Recolectar imagenes
    logger.info("Recolectando imagenes del dataset...")
    all_images = []

    for class_name in class_list:
        # Buscar en estructura típica: class_name/images/
        class_dir = input_path / class_name / "images"
        if not class_dir.exists():
            # Intentar sin subcarpeta images
            class_dir = input_path / class_name
        if not class_dir.exists():
            logger.warning("Directorio de clase no existe: %s", class_dir)
            continue

        images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        mapped_class = class_mapping[class_name]
        all_images.extend([(img, mapped_class) for img in images])
        logger.info("  %s: %d imagenes", class_name, len(images))

    if not all_images:
        logger.error("No se encontraron imagenes")
        raise typer.Exit(code=1)

    logger.info("Total: %d imagenes", len(all_images))

    # Crear splits estratificados
    logger.info("Creando splits (train=%.0f%%, val=%.0f%%, test=%.0f%%)...",
                train_ratio * 100, val_ratio * 100, test_ratio * 100)

    # Configurar semilla para reproducibilidad
    random.seed(seed)
    np.random.seed(seed)

    # Agrupar por clase
    by_class = defaultdict(list)
    for path, class_name in all_images:
        by_class[class_name].append(path)

    split_data = {'train': [], 'val': [], 'test': []}

    for class_name, paths in by_class.items():
        np.random.shuffle(paths)
        n = len(paths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        split_data['train'].extend([(p, class_name) for p in paths[:n_train]])
        split_data['val'].extend([(p, class_name) for p in paths[n_train:n_train + n_val]])
        split_data['test'].extend([(p, class_name) for p in paths[n_train + n_val:]])

    for split_name, split_images in split_data.items():
        np.random.shuffle(split_images)
        by_class_count = defaultdict(int)
        for _, c in split_images:
            by_class_count[c] += 1
        logger.info("  %s: %d imagenes %s", split_name, len(split_images), dict(by_class_count))

    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    clahe_obj = None
    if not predictions_path and use_clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))

    def predict_landmarks_ensemble(img_tensor: "torch.Tensor") -> "torch.Tensor":
        preds = []
        for lm_model in landmark_models:
            if use_tta:
                with torch.no_grad():
                    pred1 = lm_model(img_tensor)
                    images_flip = torch.flip(img_tensor, dims=[3])
                    pred2 = lm_model(images_flip)
                    pred2 = pred2.view(-1, NUM_LANDMARKS, 2)
                    pred2[:, :, 0] = 1 - pred2[:, :, 0]
                    for left, right in SYMMETRIC_PAIRS:
                        pred2[:, [left, right]] = pred2[:, [right, left]]
                    pred2 = pred2.view(-1, NUM_LANDMARKS * 2)
                    pred = (pred1 + pred2) / 2
            else:
                with torch.no_grad():
                    pred = lm_model(img_tensor)
            preds.append(pred)

        preds_stack = torch.stack(preds)
        return preds_stack.mean(dim=0)

    # Procesar imagenes
    logger.info("Procesando imagenes (margin=%.2f)...", margin)
    if predictions_path:
        logger.info("Usando cache de landmarks: %s", predictions_path)
    else:
        logger.info("Preprocessing: CLAHE=%s (clip=%.1f, tile=%d)", use_clahe, clahe_clip, clahe_tile)
        logger.info("TTA: %s", "habilitado" if use_tta else "deshabilitado")

    all_stats = {}
    all_landmarks = {}
    start_time = time.time()

    for split_name, split_images in split_data.items():
        logger.info("=== %s ===", split_name.upper())

        stats = {
            'processed': 0,
            'failed': 0,
            'fill_rates': [],
            'by_class': defaultdict(lambda: {'count': 0, 'fill_rates': []})
        }
        landmarks_data = []

        pbar = tqdm(split_images, desc=f"  {split_name}", ncols=80)

        for image_path, class_name in pbar:
            # Definir path de salida
            output_filename = f"{image_path.stem}_warped.png"
            image_output_path = output_path / split_name / class_name / output_filename

            try:
                # Cargar imagen
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    stats['failed'] += 1
                    continue

                # Resize a 224x224
                if image.shape[0] != DEFAULT_IMAGE_SIZE or image.shape[1] != DEFAULT_IMAGE_SIZE:
                    image = cv2.resize(image, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))

                if predictions_path:
                    rel_path = image_path.relative_to(input_path).as_posix()
                    landmarks = predictions_by_path.get(rel_path)
                    if landmarks is None:
                        landmarks = predictions_by_name_category.get((image_path.stem, class_name))
                    if landmarks is None:
                        stats['failed'] += 1
                        missing_predictions_total += 1
                        if len(missing_predictions) < 20:
                            missing_predictions.append(rel_path)
                        continue
                else:
                    # Preparar para prediccion
                    img_array = image.copy()

                    if clahe_obj is not None:
                        img_array = clahe_obj.apply(img_array)

                    # Convertir a RGB y normalizar
                    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                    img_float = img_rgb.astype(np.float32) / 255.0
                    img_normalized = (img_float - mean) / std

                    # Convertir a tensor
                    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
                    img_tensor = img_tensor.to(torch_device)

                    # Predecir landmarks
                    predictions = predict_landmarks_ensemble(img_tensor)

                    landmarks = predictions.squeeze().cpu().numpy()
                    landmarks = landmarks.reshape(NUM_LANDMARKS, 2) * DEFAULT_IMAGE_SIZE

                # Aplicar margin_scale
                scaled_landmarks = scale_landmarks_from_centroid(landmarks, margin)
                scaled_landmarks = clip_landmarks_to_image(scaled_landmarks)

                # Aplicar warping
                warped = piecewise_affine_warp(
                    image=image,
                    source_landmarks=scaled_landmarks,
                    target_landmarks=canonical,
                    triangles=tri,
                    use_full_coverage=use_full_coverage
                )

                # Calcular fill rate
                fill_rate = compute_fill_rate(warped)

                # Guardar imagen
                image_output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_output_path), warped)

                # Actualizar estadisticas
                stats['processed'] += 1
                stats['fill_rates'].append(fill_rate)
                stats['by_class'][class_name]['count'] += 1
                stats['by_class'][class_name]['fill_rates'].append(fill_rate)

                landmarks_data.append({
                    'image_name': image_path.stem,
                    'class': class_name,
                    'landmarks': landmarks.tolist()
                })

                # Actualizar barra de progreso cada 100 imagenes
                if stats['processed'] % 100 == 0:
                    avg_fill = np.mean(stats['fill_rates'][-100:]) if stats['fill_rates'] else 0
                    pbar.set_postfix({'fill': f'{avg_fill:.1%}'})

            except Exception as e:
                stats['failed'] += 1
                logger.debug("Error procesando %s: %s", image_path, e)
                continue

        all_stats[split_name] = stats
        all_landmarks[split_name] = landmarks_data

        # Resumen del split
        if stats['fill_rates']:
            fill_rates = np.array(stats['fill_rates'])
            logger.info("  Procesadas: %d/%d, Failed: %d",
                       stats['processed'], len(split_images), stats['failed'])
            logger.info("  Fill rate: %.1f%% +/- %.1f%%", fill_rates.mean() * 100, fill_rates.std() * 100)

    elapsed = time.time() - start_time

    # Guardar metadatos
    logger.info("Guardando metadatos...")

    # Resumen general
    predictions_meta_dict = predictions_meta if isinstance(predictions_meta, dict) else {}
    predictions_meta_serializable = None
    if predictions_meta_dict:
        try:
            json.dumps(predictions_meta_dict)
            predictions_meta_serializable = predictions_meta_dict
        except TypeError:
            predictions_meta_serializable = {"raw": str(predictions_meta_dict)}

    landmarks_info = {
        "source": "predictions" if predictions_path else "inference",
        "predictions_path": str(predictions_path) if predictions_path else None,
        "ensemble_config": ensemble_config if (ensemble_config and not predictions_path) else None,
        "checkpoint": checkpoint if (checkpoint and not predictions_path) else None,
        "tta": predictions_meta_dict.get("tta") if predictions_meta_dict else use_tta,
        "clahe": predictions_meta_dict.get("clahe") if predictions_meta_dict else use_clahe,
        "clahe_clip": predictions_meta_dict.get("clahe_clip") if predictions_meta_dict else clahe_clip,
        "clahe_tile": predictions_meta_dict.get("clahe_tile") if predictions_meta_dict else clahe_tile,
    }
    if predictions_path:
        landmarks_info["predictions_metadata"] = predictions_meta_serializable
        landmarks_info["missing_predictions"] = missing_predictions_total
        if missing_predictions:
            landmarks_info["missing_examples"] = missing_predictions

    summary = {
        'margin_scale': margin,
        'source_dataset': str(input_path),
        'classes': list(class_mapping.values()),
        'splits': {},
        'processing_time_minutes': elapsed / 60,
        'seed': seed,
        'use_full_coverage': use_full_coverage,
        'config': config,
        'landmarks': landmarks_info,
    }

    for split_name, stats in all_stats.items():
        fill_rates = np.array(stats['fill_rates']) if stats['fill_rates'] else np.array([0])
        summary['splits'][split_name] = {
            'total': len(split_data[split_name]),
            'processed': stats['processed'],
            'failed': stats['failed'],
            'fill_rate_mean': float(fill_rates.mean()) if len(fill_rates) > 0 else 0,
            'fill_rate_std': float(fill_rates.std()) if len(fill_rates) > 0 else 0,
            'by_class': {
                class_name: {
                    'count': class_stats['count'],
                    'fill_rate_mean': float(np.mean(class_stats['fill_rates'])) if class_stats['fill_rates'] else 0
                }
                for class_name, class_stats in stats['by_class'].items()
            }
        }

    summary_path = output_path / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info("Resumen guardado en: %s", summary_path)

    # Guardar landmarks predichos por split
    for split_name, landmarks_data in all_landmarks.items():
        if not landmarks_data:  # Skip empty splits
            continue
        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        landmarks_path = split_dir / "landmarks.json"
        with open(landmarks_path, 'w') as f:
            json.dump(landmarks_data, f)

    # Crear archivos CSV de indice
    for split_name in split_data.keys():
        if not split_data[split_name]:  # Skip empty splits
            continue
        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        csv_path = split_dir / "images.csv"

        with open(csv_path, 'w') as f:
            f.write("image_name,category,warped_filename\n")
            for image_path, class_name in split_data[split_name]:
                warped_name = f"{image_path.stem}_warped.png"
                f.write(f"{image_path.stem},{class_name},{warped_name}\n")

    # Resumen final
    total_processed = sum(s['processed'] for s in all_stats.values())
    total_failed = sum(s['failed'] for s in all_stats.values())

    if predictions_path and missing_predictions_total > 0:
        logger.warning("Predicciones faltantes: %d", missing_predictions_total)

    logger.info("-" * 40)
    logger.info("RESUMEN FINAL")
    logger.info("-" * 40)
    logger.info("Total procesadas: %d", total_processed)
    logger.info("Total fallidas: %d", total_failed)
    logger.info("Margin scale: %.2f", margin)
    logger.info("Tiempo: %.1f minutos (%.2fs por imagen)",
                elapsed / 60, elapsed / max(total_processed, 1))
    logger.info("Dataset generado en: %s", output_path)

    logger.info("\nEstructura:")
    logger.info("  %s/", output_path.name)
    for split_name, stats in all_stats.items():
        logger.info("  ├── %s/ (%d imagenes)", split_name, stats['processed'])
        for class_name in sorted(stats['by_class'].keys()):
            count = stats['by_class'][class_name]['count']
            logger.info("  │   ├── %s/ (%d)", class_name, count)

    logger.info("=" * 60)
    logger.info("Generacion de dataset completada!")
    logger.info("=" * 60)


# =============================================================================
# GENERATE CROPPED DATASET
# =============================================================================


@app.command("generate-cropped-dataset")
def generate_cropped_dataset(
    input_dir: Optional[str] = typer.Argument(
        None,
        help="Directorio del dataset original (estructura COVID/Normal/Viral_Pneumonia/images/)"
    ),
    output_dir: Optional[str] = typer.Argument(
        None,
        help="Directorio de salida para dataset recortado"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path a un JSON con parametros de recorte"
    ),
    crop_pct: float = typer.Option(
        0.10,
        "--crop-pct",
        "-p",
        help="Porcentaje a recortar por borde (0.10 = 10%%)"
    ),
    min_fill_rate: float = typer.Option(
        0.47,
        "--min-fill-rate",
        help="Fill rate minimo esperado (referencia: warped ~47%%)"
    ),
    splits: str = typer.Option(
        "0.75,0.125,0.125",
        "--splits",
        help="Ratios train,val,test separados por coma"
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Semilla para reproducibilidad de splits"
    ),
    classes: str = typer.Option(
        "COVID,Normal,Viral Pneumonia",
        "--classes",
        help="Clases a procesar separadas por coma"
    ),
):
    """
    Generar dataset recortado (sin warping) con splits train/val/test.

    Este comando:
    1. Carga imagenes del dataset original
    2. Recorta un porcentaje fijo por borde para eliminar etiquetas
    3. Redimensiona a 224x224
    4. Crea splits train/val/test estratificados
    5. Guarda metadata (images.csv, dataset_summary.json)

    Estructura de salida:
        output_dir/
        ├── train/
        │   ├── COVID/
        │   ├── Normal/
        │   └── Viral_Pneumonia/
        ├── val/
        ├── test/
        └── dataset_summary.json

    Ejemplo:
        python -m src_v2 generate-cropped-dataset data/dataset/COVID-19_Radiography_Dataset \\
            outputs/cropped_lung/session_crop10 --crop-pct 0.10
        python -m src_v2 generate-cropped-dataset --config configs/cropping_10.json
    """
    import json
    import time
    from collections import defaultdict

    import cv2
    import numpy as np
    from tqdm import tqdm

    logger.info("=" * 60)
    logger.info("COVID-19 Landmark Detection - Generate Cropped Dataset")
    logger.info("=" * 60)

    argv = sys.argv[1:]

    def option_present(*names: str) -> bool:
        for name in names:
            if name in argv:
                return True
            prefix = f"{name}="
            if any(arg.startswith(prefix) for arg in argv):
                return True
        return False

    config_data = {}
    if config:
        config_path = Path(config)
        if not config_path.exists():
            logger.error("Config no existe: %s", config)
            raise typer.Exit(code=1)
        with open(config_path, "r") as f:
            config_data = json.load(f)

        def override_param(
            param_name: str,
            current_value,
            config_key: Optional[str] = None,
            option_names: Optional[tuple[str, ...]] = None,
        ):
            key = config_key or param_name
            if key not in config_data:
                return current_value
            if option_names and option_present(*option_names):
                return current_value
            if option_names is None and current_value is not None:
                return current_value
            return config_data[key]

        input_dir = override_param("input_dir", input_dir, "input_dir")
        output_dir = override_param("output_dir", output_dir, "output_dir")
        crop_pct = override_param("crop_pct", crop_pct, "crop_pct", ("--crop-pct", "-p"))
        min_fill_rate = override_param(
            "min_fill_rate",
            min_fill_rate,
            "min_fill_rate",
            ("--min-fill-rate",),
        )
        splits = override_param("splits", splits, "splits", ("--splits",))
        seed = override_param("seed", seed, "seed", ("--seed", "-s"))
        classes = override_param("classes", classes, "classes", ("--classes",))

    if isinstance(splits, (list, tuple)):
        splits = ",".join(str(x) for x in splits)
    if isinstance(classes, (list, tuple)):
        classes = ",".join(str(x) for x in classes)

    if input_dir is None or output_dir is None:
        logger.error("input_dir y output_dir son requeridos (usa argumentos o --config)")
        raise typer.Exit(code=1)

    # Normalizar percentiles si se pasan como 10 en lugar de 0.10
    if crop_pct > 1.0:
        logger.info("crop_pct=%.2f interpretado como porcentaje", crop_pct)
        crop_pct = crop_pct / 100.0
    if min_fill_rate > 1.0:
        logger.info("min_fill_rate=%.2f interpretado como porcentaje", min_fill_rate)
        min_fill_rate = min_fill_rate / 100.0

    if crop_pct < 0 or crop_pct >= 0.5:
        logger.error("crop_pct debe estar en [0, 0.5); recibido: %.3f", crop_pct)
        raise typer.Exit(code=1)

    if min_fill_rate <= 0 or min_fill_rate > 1.0:
        logger.error("min_fill_rate debe estar en (0, 1]; recibido: %.3f", min_fill_rate)
        raise typer.Exit(code=1)

    # Verificar paths
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error("Directorio de entrada no existe: %s", input_dir)
        raise typer.Exit(code=1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parsear splits
    try:
        split_ratios = [float(x) for x in splits.split(",")]
        if len(split_ratios) != 3:
            raise ValueError("Se requieren exactamente 3 valores")
        if abs(sum(split_ratios) - 1.0) > 0.01:
            raise ValueError(f"Los ratios deben sumar 1.0, suman {sum(split_ratios)}")
        train_ratio, val_ratio, test_ratio = split_ratios
    except Exception as e:
        logger.error("Error parseando splits '%s': %s", splits, e)
        raise typer.Exit(code=1)

    # Parsear clases
    class_list = [c.strip() for c in classes.split(",")]
    class_mapping = {c: c.replace(" ", "_") for c in class_list}
    logger.info("Clases a procesar: %s", class_list)

    # Calcular fill rate esperado (area relativa con redondeo de pixeles)
    crop_pixels = int(round(DEFAULT_IMAGE_SIZE * crop_pct))
    remaining = DEFAULT_IMAGE_SIZE - (2 * crop_pixels)
    expected_fill_rate = (remaining / DEFAULT_IMAGE_SIZE) ** 2 if remaining > 0 else 0.0
    logger.info(
        "Recorte: %.1f%% por borde (%d px) | Fill rate esperado: %.1f%%",
        crop_pct * 100,
        crop_pixels,
        expected_fill_rate * 100,
    )
    if expected_fill_rate < min_fill_rate:
        logger.error("Fill rate esperado %.1f%% < minimo %.1f%% (ajusta crop_pct)",
                     expected_fill_rate * 100, min_fill_rate * 100)
        raise typer.Exit(code=1)

    crop_label = int(round(crop_pct * 100))

    # Recolectar imagenes
    logger.info("Recolectando imagenes del dataset...")
    all_images = []

    for class_name in class_list:
        # Buscar en estructura típica: class_name/images/
        class_dir = input_path / class_name / "images"
        if not class_dir.exists():
            # Intentar sin subcarpeta images
            class_dir = input_path / class_name
        if not class_dir.exists():
            logger.warning("Directorio de clase no existe: %s", class_dir)
            continue

        images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        mapped_class = class_mapping[class_name]
        all_images.extend([(img, mapped_class) for img in images])
        logger.info("  %s: %d imagenes", class_name, len(images))

    if not all_images:
        logger.error("No se encontraron imagenes")
        raise typer.Exit(code=1)

    logger.info("Total: %d imagenes", len(all_images))

    # Crear splits estratificados
    logger.info("Creando splits (train=%.0f%%, val=%.0f%%, test=%.0f%%)...",
                train_ratio * 100, val_ratio * 100, test_ratio * 100)

    # Configurar semilla para reproducibilidad
    random.seed(seed)
    np.random.seed(seed)

    # Agrupar por clase
    by_class = defaultdict(list)
    for path, class_name in all_images:
        by_class[class_name].append(path)

    split_data = {'train': [], 'val': [], 'test': []}

    for class_name, paths in by_class.items():
        np.random.shuffle(paths)
        n = len(paths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        split_data['train'].extend([(p, class_name) for p in paths[:n_train]])
        split_data['val'].extend([(p, class_name) for p in paths[n_train:n_train + n_val]])
        split_data['test'].extend([(p, class_name) for p in paths[n_train + n_val:]])

    for split_name, split_images in split_data.items():
        np.random.shuffle(split_images)
        by_class_count = defaultdict(int)
        for _, c in split_images:
            by_class_count[c] += 1
        logger.info("  %s: %d imagenes %s", split_name, len(split_images), dict(by_class_count))

    # Procesar imagenes
    logger.info("Procesando imagenes (crop=%.1f%%)...", crop_pct * 100)
    all_stats = {}
    start_time = time.time()

    for split_name, split_images in split_data.items():
        logger.info("=== %s ===", split_name.upper())

        stats = {
            'processed': 0,
            'failed': 0,
            'fill_rates': [],
            'by_class': defaultdict(lambda: {'count': 0, 'fill_rates': []})
        }

        pbar = tqdm(split_images, desc=f"  {split_name}", ncols=80)

        for image_path, class_name in pbar:
            output_filename = f"{image_path.stem}_crop{crop_label}.png"
            image_output_path = output_path / split_name / class_name / output_filename

            try:
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    stats['failed'] += 1
                    continue

                if image.shape[0] != DEFAULT_IMAGE_SIZE or image.shape[1] != DEFAULT_IMAGE_SIZE:
                    image = cv2.resize(image, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))

                height, width = image.shape[:2]
                crop_y = int(round(height * crop_pct))
                crop_x = int(round(width * crop_pct))

                if crop_y > 0 or crop_x > 0:
                    new_h = height - 2 * crop_y
                    new_w = width - 2 * crop_x
                    if new_h <= 0 or new_w <= 0:
                        stats['failed'] += 1
                        continue
                    cropped = image[crop_y:height - crop_y, crop_x:width - crop_x]
                    fill_rate = (cropped.size / image.size) if image.size else 0.0
                else:
                    cropped = image
                    fill_rate = 1.0

                if cropped.shape[0] != DEFAULT_IMAGE_SIZE or cropped.shape[1] != DEFAULT_IMAGE_SIZE:
                    cropped = cv2.resize(cropped, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))

                image_output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_output_path), cropped)

                stats['processed'] += 1
                stats['fill_rates'].append(fill_rate)
                stats['by_class'][class_name]['count'] += 1
                stats['by_class'][class_name]['fill_rates'].append(fill_rate)

                if stats['processed'] % 100 == 0:
                    avg_fill = np.mean(stats['fill_rates'][-100:]) if stats['fill_rates'] else 0
                    pbar.set_postfix({'fill': f'{avg_fill:.1%}'})

            except Exception as e:
                stats['failed'] += 1
                logger.debug("Error procesando %s: %s", image_path, e)
                continue

        all_stats[split_name] = stats

        if stats['fill_rates']:
            fill_rates = np.array(stats['fill_rates'])
            logger.info("  Procesadas: %d/%d, Failed: %d",
                        stats['processed'], len(split_images), stats['failed'])
            logger.info(
                "  Fill rate: %.1f%% +/- %.1f%%",
                fill_rates.mean() * 100,
                fill_rates.std() * 100,
            )

    elapsed = time.time() - start_time

    # Guardar metadatos
    logger.info("Guardando metadatos...")

    summary = {
        'crop_pct': crop_pct,
        'crop_pixels': crop_pixels,
        'min_fill_rate': min_fill_rate,
        'expected_fill_rate': expected_fill_rate,
        'source_dataset': str(input_path),
        'classes': list(class_mapping.values()),
        'splits': {},
        'processing_time_minutes': elapsed / 60,
        'seed': seed,
        'config': config,
    }

    for split_name, stats in all_stats.items():
        fill_rates = np.array(stats['fill_rates']) if stats['fill_rates'] else np.array([0])
        summary['splits'][split_name] = {
            'total': len(split_data[split_name]),
            'processed': stats['processed'],
            'failed': stats['failed'],
            'fill_rate_mean': float(fill_rates.mean()) if len(fill_rates) > 0 else 0,
            'fill_rate_std': float(fill_rates.std()) if len(fill_rates) > 0 else 0,
            'by_class': {
                class_name: {
                    'count': class_stats['count'],
                    'fill_rate_mean': float(np.mean(class_stats['fill_rates']))
                    if class_stats['fill_rates'] else 0
                }
                for class_name, class_stats in stats['by_class'].items()
            }
        }

    summary_path = output_path / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info("Resumen guardado en: %s", summary_path)

    # Crear archivos CSV de indice
    for split_name in split_data.keys():
        if not split_data[split_name]:
            continue
        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        csv_path = split_dir / "images.csv"

        with open(csv_path, 'w') as f:
            f.write("image_name,category,cropped_filename\n")
            for image_path, class_name in split_data[split_name]:
                cropped_name = f"{image_path.stem}_crop{crop_label}.png"
                f.write(f"{image_path.stem},{class_name},{cropped_name}\n")

    total_processed = sum(s['processed'] for s in all_stats.values())
    total_failed = sum(s['failed'] for s in all_stats.values())

    logger.info("-" * 40)
    logger.info("RESUMEN FINAL")
    logger.info("-" * 40)
    logger.info("Total procesadas: %d", total_processed)
    logger.info("Total fallidas: %d", total_failed)
    logger.info("Crop pct: %.1f%%", crop_pct * 100)
    logger.info("Tiempo: %.1f minutos (%.2fs por imagen)",
                elapsed / 60, elapsed / max(total_processed, 1))
    logger.info("Dataset generado en: %s", output_path)

    logger.info("\nEstructura:")
    logger.info("  %s/", output_path.name)
    for split_name, stats in all_stats.items():
        logger.info("  ├── %s/ (%d imagenes)", split_name, stats['processed'])
        for class_name in sorted(stats['by_class'].keys()):
            count = stats['by_class'][class_name]['count']
            logger.info("  │   ├── %s/ (%d)", class_name, count)

    logger.info("=" * 60)
    logger.info("Generacion de dataset completada!")
    logger.info("=" * 60)


# =============================================================================
# APPLY CONTRAST DATASET (SAHS / CLAHE)
# =============================================================================


def _parse_contrast_exts(exts: str) -> set[str]:
    """
    Parse comma-separated image extensions.

    Args:
        exts: Comma-separated extensions (e.g., ".png,.jpg,jpeg").

    Returns:
        Set of normalized extensions with a leading dot.
    """
    parsed = set()
    for token in exts.split(","):
        value = token.strip().lower()
        if not value:
            continue
        if not value.startswith("."):
            value = f".{value}"
        parsed.add(value)
    return parsed


def _is_subpath(child: Path, parent: Path) -> bool:
    """
    Check whether child path is inside parent.

    Args:
        child: Candidate child path.
        parent: Candidate parent path.

    Returns:
        True if child is inside parent.
    """
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _to_gray(image: "np.ndarray") -> "np.ndarray":
    """
    Convert an image array to grayscale.

    Args:
        image: Input array (grayscale, BGR, or BGRA).

    Returns:
        Grayscale array.
    """
    import cv2

    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _build_foreground_mask(gray: "np.ndarray", threshold: int) -> "np.ndarray":
    """
    Build a foreground mask from a grayscale image.

    Args:
        gray: Grayscale image array.
        threshold: Pixels <= threshold are considered background.

    Returns:
        Boolean mask for foreground pixels.
    """
    return gray > threshold


def _apply_sahs_to_gray(
    gray: "np.ndarray",
    upper_factor: float,
    lower_factor: float,
    mask: Optional["np.ndarray"],
) -> "np.ndarray":
    """
    Apply SAHS contrast enhancement to a grayscale image.

    Args:
        gray: Grayscale image array.
        upper_factor: Upper std multiplier.
        lower_factor: Lower std multiplier.
        mask: Optional mask for foreground pixels.

    Returns:
        Enhanced grayscale image array.
    """
    import numpy as np

    if gray is None:
        raise ValueError("gray is None")

    apply_mask = mask is not None
    mask_values = mask if mask is not None else np.ones_like(gray, dtype=bool)
    if not np.any(mask_values):
        return gray

    gray_float = gray.astype(np.float32)
    masked_pixels = gray_float[mask_values]
    gray_mean = float(np.mean(masked_pixels))

    above_mean = masked_pixels[masked_pixels > gray_mean]
    below_or_equal = masked_pixels[masked_pixels <= gray_mean]

    max_value = gray_mean
    min_value = gray_mean

    if above_mean.size:
        std_above = float(np.sqrt(np.mean((above_mean - gray_mean) ** 2)))
        max_value = gray_mean + upper_factor * std_above

    if below_or_equal.size:
        std_below = float(np.sqrt(np.mean((below_or_equal - gray_mean) ** 2)))
        min_value = gray_mean - lower_factor * std_below

    if np.issubdtype(gray.dtype, np.integer):
        scale_max = float(np.iinfo(gray.dtype).max)
    else:
        scale_max = float(np.max(gray_float)) if gray_float.size else 1.0

    if max_value == min_value:
        enhanced_full = gray_float
    else:
        enhanced_full = (scale_max / (max_value - min_value)) * (gray_float - min_value)
        enhanced_full = np.clip(enhanced_full, 0.0, scale_max)

    if apply_mask:
        output = gray_float.copy()
        output[mask_values] = enhanced_full[mask_values]
        return output.astype(gray.dtype)

    return enhanced_full.astype(gray.dtype)


def _apply_clahe_to_gray(
    gray: "np.ndarray",
    clip_limit: float,
    tile_size: int,
    mask: Optional["np.ndarray"],
) -> "np.ndarray":
    """
    Apply CLAHE to a grayscale image.

    Args:
        gray: Grayscale image array.
        clip_limit: CLAHE clip limit.
        tile_size: CLAHE tile size.
        mask: Optional mask for foreground pixels.

    Returns:
        Enhanced grayscale image array.
    """
    import cv2
    import numpy as np

    if gray is None:
        raise ValueError("gray is None")

    if gray.dtype != np.uint8:
        gray_u8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        gray_u8 = gray

    if mask is not None and not np.any(mask):
        return gray_u8 if gray.dtype == np.uint8 else gray

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced_full = clahe.apply(gray_u8)

    if mask is None:
        return enhanced_full if gray.dtype == np.uint8 else enhanced_full.astype(gray.dtype)

    output = gray_u8.copy()
    output[mask] = enhanced_full[mask]
    return output if gray.dtype == np.uint8 else output.astype(gray.dtype)


def _crop_to_foreground(
    gray: "np.ndarray",
    mask: Optional["np.ndarray"],
    output_size: int,
    pad_to_square: bool,
    pad_value: int,
) -> "np.ndarray":
    """
    Crop to the foreground bounding box and resize.

    Args:
        gray: Grayscale image array.
        mask: Foreground mask for cropping.
        output_size: Output size after resizing.
        pad_to_square: Pad to square before resize.
        pad_value: Padding value for background.

    Returns:
        Cropped and resized image array.
    """
    import cv2
    import numpy as np

    if mask is None or not np.any(mask):
        return gray

    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max() + 1
    x_min, x_max = xs.min(), xs.max() + 1
    cropped = gray[y_min:y_max, x_min:x_max]

    if output_size <= 0:
        return cropped

    if pad_to_square:
        side = max(cropped.shape[0], cropped.shape[1])
        canvas = np.full((side, side), pad_value, dtype=cropped.dtype)
        y_offset = (side - cropped.shape[0]) // 2
        x_offset = (side - cropped.shape[1]) // 2
        canvas[y_offset:y_offset + cropped.shape[0], x_offset:x_offset + cropped.shape[1]] = cropped
        resized = cv2.resize(canvas, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    else:
        resized = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    return resized


def _apply_contrast_dataset(
    method: str,
    input_dir: str,
    output_dir: str,
    exts: str,
    copy_non_images: bool,
    overwrite: bool,
    preserve_background: bool,
    background_threshold: int,
    crop_foreground: bool,
    output_size: int,
    pad_to_square: bool,
    pad_value: int,
    upper_factor: float,
    lower_factor: float,
    clahe_clip: float,
    clahe_tile: int,
) -> None:
    import shutil

    import cv2
    from tqdm import tqdm

    method = method.lower().strip()
    if method not in {"sahs", "clahe"}:
        logger.error("Metodo invalido: %s (usa sahs o clahe)", method)
        raise typer.Exit(code=1)

    if background_threshold < 0:
        logger.error("background_threshold debe ser >= 0")
        raise typer.Exit(code=1)

    if crop_foreground and output_size <= 0:
        logger.error("output_size debe ser > 0 si se usa crop_foreground")
        raise typer.Exit(code=1)

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        logger.error("Directorio de entrada no existe: %s", input_dir)
        raise typer.Exit(code=1)

    if input_path.resolve() == output_path.resolve():
        logger.error("input_dir y output_dir deben ser distintos")
        raise typer.Exit(code=1)

    if _is_subpath(output_path.resolve(), input_path.resolve()):
        logger.error("output_dir no puede estar dentro de input_dir")
        raise typer.Exit(code=1)

    exts_set = _parse_contrast_exts(exts)
    if not exts_set:
        logger.error("Lista de extensiones vacia: %s", exts)
        raise typer.Exit(code=1)

    files = [p for p in input_path.rglob("*") if p.is_file()]
    if not files:
        logger.error("No se encontraron archivos en %s", input_dir)
        raise typer.Exit(code=1)

    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Metodo: %s", method)
    if method == "sahs":
        logger.info("SAHS (upper=%.2f, lower=%.2f)", upper_factor, lower_factor)
    else:
        logger.info("CLAHE (clip=%.2f, tile=%d)", clahe_clip, clahe_tile)
    logger.info("Extensiones: %s", ", ".join(sorted(exts_set)))
    logger.info("Preservar fondo: %s (threshold=%d)", preserve_background, background_threshold)
    logger.info("Crop foreground: %s (output=%d, pad=%s)", crop_foreground, output_size, pad_to_square)
    logger.info("Copiar no imagenes: %s", copy_non_images)
    logger.info("Overwrite: %s", overwrite)

    processed = 0
    copied = 0
    skipped = 0
    failed = 0

    pbar = tqdm(files, desc=method.upper(), ncols=80)
    for path in pbar:
        rel_path = path.relative_to(input_path)
        out_path = output_path / rel_path

        if path.suffix.lower() in exts_set:
            if out_path.exists() and not overwrite:
                skipped += 1
                continue
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                failed += 1
                continue

            try:
                gray = _to_gray(image)
            except Exception:
                failed += 1
                continue

            mask = None
            if preserve_background or crop_foreground:
                mask = _build_foreground_mask(gray, background_threshold)

            if method == "sahs":
                enhanced = _apply_sahs_to_gray(
                    gray,
                    upper_factor=upper_factor,
                    lower_factor=lower_factor,
                    mask=mask if preserve_background else None,
                )
            else:
                enhanced = _apply_clahe_to_gray(
                    gray,
                    clip_limit=clahe_clip,
                    tile_size=clahe_tile,
                    mask=mask if preserve_background else None,
                )

            if crop_foreground:
                enhanced = _crop_to_foreground(
                    enhanced,
                    mask=mask,
                    output_size=output_size,
                    pad_to_square=pad_to_square,
                    pad_value=pad_value,
                )

            out_path.parent.mkdir(parents=True, exist_ok=True)
            if cv2.imwrite(str(out_path), enhanced):
                processed += 1
            else:
                failed += 1
        else:
            if not copy_non_images:
                skipped += 1
                continue
            if out_path.exists() and not overwrite:
                skipped += 1
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out_path)
            copied += 1

    logger.info("-" * 40)
    logger.info("RESUMEN CONTRASTE")
    logger.info("-" * 40)
    logger.info("Procesadas: %d", processed)
    logger.info("Copiadas: %d", copied)
    logger.info("Saltadas: %d", skipped)
    logger.info("Fallidas: %d", failed)
    logger.info("Salida: %s", output_path)


@app.command("apply-contrast-dataset")
def apply_contrast_dataset(
    input_dir: str = typer.Argument(
        ...,
        help="Directorio de entrada con dataset",
    ),
    output_dir: str = typer.Argument(
        ...,
        help="Directorio de salida para el dataset procesado",
    ),
    method: str = typer.Option(
        "sahs",
        "--method",
        help="Metodo de contraste: sahs o clahe",
    ),
    exts: str = typer.Option(
        ".png,.jpg,.jpeg",
        "--exts",
        help="Extensiones de imagen separadas por coma",
    ),
    copy_non_images: bool = typer.Option(
        True,
        "--copy-non-images/--no-copy-non-images",
        help="Copiar archivos no imagen (CSV/JSON/etc.)",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Sobrescribir archivos existentes en output",
    ),
    preserve_background: bool = typer.Option(
        True,
        "--preserve-background/--no-preserve-background",
        help="Preservar pixeles de fondo (<= threshold)",
    ),
    background_threshold: int = typer.Option(
        0,
        "--background-threshold",
        help="Umbral para fondo (pixeles <= threshold)",
    ),
    crop_foreground: bool = typer.Option(
        False,
        "--crop-foreground/--no-crop-foreground",
        help="Recortar al bounding box del foreground y redimensionar",
    ),
    output_size: int = typer.Option(
        DEFAULT_IMAGE_SIZE,
        "--output-size",
        help="Tamano final si se usa crop_foreground",
    ),
    pad_to_square: bool = typer.Option(
        True,
        "--pad-to-square/--stretch",
        help="Pad a cuadrado antes de redimensionar",
    ),
    pad_value: int = typer.Option(
        0,
        "--pad-value",
        help="Valor de padding para el fondo",
    ),
    upper_factor: float = typer.Option(
        2.5,
        "--upper-factor",
        help="Factor superior de desviacion estandar (SAHS)",
    ),
    lower_factor: float = typer.Option(
        2.0,
        "--lower-factor",
        help="Factor inferior de desviacion estandar (SAHS)",
    ),
    clahe_clip: float = typer.Option(
        DEFAULT_CLAHE_CLIP_LIMIT,
        "--clahe-clip",
        help="CLAHE clip limit",
    ),
    clahe_tile: int = typer.Option(
        DEFAULT_CLAHE_TILE_SIZE,
        "--clahe-tile",
        help="CLAHE tile size",
    ),
):
    """
    Apply SAHS or CLAHE to all images in a dataset directory.

    The command preserves directory structure and copies non-image
    files (images.csv, dataset_summary.json, landmarks.json, etc.).

    Examples:
        python -m src_v2 apply-contrast-dataset outputs/warped_lung_best/session_warping \
            outputs/warped_lung_best/session_warping_sahs --method sahs
        python -m src_v2 apply-contrast-dataset outputs/warped_lung_best/session_warping \
            outputs/warped_lung_best/session_warping_clahe --method clahe --clahe-tile 4
    """
    _apply_contrast_dataset(
        method=method,
        input_dir=input_dir,
        output_dir=output_dir,
        exts=exts,
        copy_non_images=copy_non_images,
        overwrite=overwrite,
        preserve_background=preserve_background,
        background_threshold=background_threshold,
        crop_foreground=crop_foreground,
        output_size=output_size,
        pad_to_square=pad_to_square,
        pad_value=pad_value,
        upper_factor=upper_factor,
        lower_factor=lower_factor,
        clahe_clip=clahe_clip,
        clahe_tile=clahe_tile,
    )


# =============================================================================
# CROP DATASET (remove black background from warped images)
# =============================================================================


def _crop_dataset_impl(
    input_dir: str,
    output_dir: str,
    exts: str,
    copy_non_images: bool,
    overwrite: bool,
    background_threshold: int,
    output_size: int,
    pad_to_square: bool,
    pad_value: int,
) -> None:
    """Implementation of crop-dataset command."""
    import shutil

    import cv2
    from tqdm import tqdm

    if background_threshold < 0:
        logger.error("background_threshold debe ser >= 0")
        raise typer.Exit(code=1)

    if output_size <= 0:
        logger.error("output_size debe ser > 0")
        raise typer.Exit(code=1)

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        logger.error("Directorio de entrada no existe: %s", input_dir)
        raise typer.Exit(code=1)

    if input_path.resolve() == output_path.resolve():
        logger.error("input_dir y output_dir deben ser distintos")
        raise typer.Exit(code=1)

    if _is_subpath(output_path.resolve(), input_path.resolve()):
        logger.error("output_dir no puede estar dentro de input_dir")
        raise typer.Exit(code=1)

    exts_set = _parse_contrast_exts(exts)
    if not exts_set:
        logger.error("Lista de extensiones vacia: %s", exts)
        raise typer.Exit(code=1)

    files = [p for p in input_path.rglob("*") if p.is_file()]
    if not files:
        logger.error("No se encontraron archivos en %s", input_dir)
        raise typer.Exit(code=1)

    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("CROP DATASET")
    logger.info("-" * 40)
    logger.info("Entrada: %s", input_dir)
    logger.info("Salida: %s", output_dir)
    logger.info("Extensiones: %s", ", ".join(sorted(exts_set)))
    logger.info("Background threshold: %d", background_threshold)
    logger.info("Output size: %d", output_size)
    logger.info("Pad to square: %s (pad_value=%d)", pad_to_square, pad_value)
    logger.info("Copiar no imagenes: %s", copy_non_images)
    logger.info("-" * 40)

    processed = 0
    copied = 0
    skipped = 0
    failed = 0

    pbar = tqdm(files, desc="CROP", ncols=80)
    for path in pbar:
        rel_path = path.relative_to(input_path)
        out_path = output_path / rel_path

        if path.suffix.lower() in exts_set:
            if out_path.exists() and not overwrite:
                skipped += 1
                continue

            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                failed += 1
                continue

            try:
                gray = _to_gray(image)
            except Exception:
                failed += 1
                continue

            # Build foreground mask
            mask = _build_foreground_mask(gray, background_threshold)

            if not np.any(mask):
                # No foreground found, skip or copy original
                failed += 1
                continue

            # Crop to foreground bounding box and resize
            cropped = _crop_to_foreground(
                gray,
                mask=mask,
                output_size=output_size,
                pad_to_square=pad_to_square,
                pad_value=pad_value,
            )

            out_path.parent.mkdir(parents=True, exist_ok=True)
            if cv2.imwrite(str(out_path), cropped):
                processed += 1
            else:
                failed += 1
        else:
            if not copy_non_images:
                skipped += 1
                continue
            if out_path.exists() and not overwrite:
                skipped += 1
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out_path)
            copied += 1

    logger.info("-" * 40)
    logger.info("RESUMEN CROP")
    logger.info("-" * 40)
    logger.info("Procesadas: %d", processed)
    logger.info("Copiadas: %d", copied)
    logger.info("Saltadas: %d", skipped)
    logger.info("Fallidas: %d", failed)
    logger.info("Salida: %s", output_path)


@app.command("crop-dataset")
def crop_dataset(
    input_dir: str = typer.Argument(
        ...,
        help="Directorio de entrada con imagenes warped",
    ),
    output_dir: str = typer.Argument(
        ...,
        help="Directorio de salida para imagenes recortadas",
    ),
    exts: str = typer.Option(
        ".png,.jpg,.jpeg",
        "--exts",
        help="Extensiones de imagen separadas por coma",
    ),
    copy_non_images: bool = typer.Option(
        True,
        "--copy-non-images/--no-copy-non-images",
        help="Copiar archivos no imagen (CSV/JSON/etc.)",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Sobrescribir archivos existentes en output",
    ),
    background_threshold: int = typer.Option(
        0,
        "--background-threshold",
        help="Umbral para fondo (pixeles <= threshold)",
    ),
    output_size: int = typer.Option(
        DEFAULT_IMAGE_SIZE,
        "--output-size",
        help="Tamano final de imagen (default: 224)",
    ),
    pad_to_square: bool = typer.Option(
        True,
        "--pad-to-square/--stretch",
        help="Pad a cuadrado antes de redimensionar (vs stretch)",
    ),
    pad_value: int = typer.Option(
        0,
        "--pad-value",
        help="Valor de padding para el fondo (default: 0 negro)",
    ),
):
    """
    Crop warped images to remove black background and resize.

    This command processes warped images that have black background regions,
    crops to the bounding box of non-black pixels, and resizes to the
    specified output size.

    The command preserves directory structure and optionally copies
    non-image files (CSV/JSON/etc.).

    Examples:
        # Crop warped images removing black background
        python -m src_v2 crop-dataset outputs/warped_lung_best/session_warping \\
            outputs/warped_cropped/session_warping

        # With custom output size
        python -m src_v2 crop-dataset outputs/warped_lung_best/session_warping \\
            outputs/warped_cropped/session_warping --output-size 256
    """
    _crop_dataset_impl(
        input_dir=input_dir,
        output_dir=output_dir,
        exts=exts,
        copy_non_images=copy_non_images,
        overwrite=overwrite,
        background_threshold=background_threshold,
        output_size=output_size,
        pad_to_square=pad_to_square,
        pad_value=pad_value,
    )


# =============================================================================
# COMPARE ARCHITECTURES
# =============================================================================

# Arquitecturas soportadas por ImageClassifier
SUPPORTED_ARCHITECTURES = [
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "densenet121",
    "alexnet",
    "vgg16",
    "mobilenet_v2",
]

ARCHITECTURE_DISPLAY_NAMES = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "efficientnet_b0": "EfficientNet-B0",
    "densenet121": "DenseNet-121",
    "alexnet": "AlexNet",
    "vgg16": "VGG-16",
    "mobilenet_v2": "MobileNetV2",
}


def _train_single_architecture(
    model_name: str,
    train_loader,
    val_loader,
    test_loader,
    class_names: list,
    class_weights_tensor,
    torch_device,
    epochs: int,
    lr: float,
    patience: int,
) -> dict:
    """
    Entrena una arquitectura y retorna metricas.

    Returns:
        dict con resultados incluyendo metricas, historial y tiempo
    """
    import time
    from datetime import datetime

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        accuracy_score
    )

    from src_v2.models import ImageClassifier

    start_time = time.time()

    # Validar que los loaders tienen datos
    if len(train_loader) == 0:
        raise ValueError("train_loader está vacío: no hay datos para entrenar")
    if len(val_loader) == 0:
        raise ValueError("val_loader está vacío: no hay datos para validar")
    if len(test_loader) == 0:
        raise ValueError("test_loader está vacío: no hay datos para evaluar")

    # Crear modelo
    model = ImageClassifier(
        backbone=model_name,
        num_classes=len(class_names),
        pretrained=True,
        dropout=0.3,
    )
    model = model.to(torch_device)

    # Contar parametros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Loss y optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Historial de entrenamiento
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_f1_macro': [], 'val_f1_weighted': []
    }

    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(torch_device), labels.to(torch_device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(torch_device), labels.to(torch_device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(all_labels)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1_macro = f1_score(all_labels, all_preds, average='macro')
        val_f1_weighted = f1_score(all_labels, all_preds, average='weighted')

        # Scheduler step
        scheduler.step(val_f1_macro)

        # Guardar historial
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1_macro'].append(val_f1_macro)
        history['val_f1_weighted'].append(val_f1_weighted)

        # Early stopping
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("    Early stopping en epoch %d", epoch + 1)
                break

        if (epoch + 1) % 5 == 0:
            logger.info("    Epoch %3d: Val Acc=%.4f, F1=%.4f",
                       epoch + 1, val_acc, val_f1_macro)

    # Cargar mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(torch_device)

    # Evaluacion final en test
    model.eval()
    test_preds = []
    test_labels_list = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(torch_device), labels.to(torch_device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels_list.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_labels_list, test_preds)
    test_f1_macro = f1_score(test_labels_list, test_preds, average='macro')
    test_f1_weighted = f1_score(test_labels_list, test_preds, average='weighted')

    cm = confusion_matrix(test_labels_list, test_preds)
    per_class_report = classification_report(
        test_labels_list, test_preds,
        target_names=class_names,
        output_dict=True
    )

    training_time = time.time() - start_time

    # Calcular tamaño del modelo en MB
    model_size_mb = total_params * 4 / (1024 * 1024)  # float32 = 4 bytes

    results = {
        'model': model_name,
        'model_display_name': ARCHITECTURE_DISPLAY_NAMES.get(model_name, model_name),
        'epochs_trained': len(history['train_loss']),
        'best_val_f1': float(best_val_f1),
        'test_metrics': {
            'accuracy': float(test_acc),
            'f1_macro': float(test_f1_macro),
            'f1_weighted': float(test_f1_weighted),
        },
        'per_class_metrics': per_class_report,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'training_time_seconds': training_time,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb,
        'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        'timestamp': datetime.now().isoformat(),
    }

    # Retornar el mejor estado del modelo (ya está en CPU desde línea 4017)
    # Si nunca mejoró, usar el estado actual
    if best_model_state is None:
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Liberar memoria GPU
    del model
    if torch_device.type == 'cuda':
        torch.cuda.empty_cache()

    return results, best_model_state


def _generate_comparison_figures(
    results_list: list,
    output_dir: Path,
    dataset_label: str = "warped"
):
    """
    Genera figuras comparativas de los resultados.

    Args:
        results_list: Lista de diccionarios con resultados por arquitectura
        output_dir: Directorio para guardar figuras
        dataset_label: Etiqueta del dataset (warped/original)
    """
    import numpy as np

    try:
        import matplotlib
        matplotlib.use('Agg')  # Backend no interactivo
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn no disponible, saltando visualizaciones")
        return

    # Usar subdirectorio con el label para evitar sobrescribir
    figures_dir = output_dir / f"figures_{dataset_label}"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not results_list:
        return

    # Extraer datos
    models = [r['model_display_name'] for r in results_list]
    accuracies = [r['test_metrics']['accuracy'] * 100 for r in results_list]
    f1_scores = [r['test_metrics']['f1_macro'] * 100 for r in results_list]
    training_times = [r['training_time_seconds'] / 60 for r in results_list]  # en minutos

    # 1. Gráfico de barras: Accuracy comparativo
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    bars = ax.bar(x, accuracies, color='steelblue', alpha=0.8, edgecolor='navy')
    ax.set_xlabel('Arquitectura', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Comparación de Accuracy por Arquitectura ({dataset_label})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(max(0, min(accuracies) - 10), 100)
    ax.grid(axis='y', alpha=0.3)

    # Agregar etiquetas de valor
    for bar, val in zip(bars, accuracies):
        ax.annotate(f'{val:.1f}%',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(figures_dir / 'accuracy_comparison.png', dpi=150)
    plt.close()

    # 2. Gráfico de barras: F1 Score comparativo
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(x, f1_scores, color='darkorange', alpha=0.8, edgecolor='darkred')
    ax.set_xlabel('Arquitectura', fontsize=12)
    ax.set_ylabel('F1-Macro (%)', fontsize=12)
    ax.set_title(f'Comparación de F1-Macro por Arquitectura ({dataset_label})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(max(0, min(f1_scores) - 10), 100)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, f1_scores):
        ax.annotate(f'{val:.1f}%',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(figures_dir / 'f1_comparison.png', dpi=150)
    plt.close()

    # 3. Matrices de confusión combinadas
    n_models = len(results_list)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_models == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, r in enumerate(results_list):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        cm = np.array(r['confusion_matrix'])
        class_names = r['class_names']

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, cbar=False
        )
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Real')
        ax.set_title(f"{r['model_display_name']}\nAcc: {r['test_metrics']['accuracy']*100:.1f}%")

    # Ocultar ejes vacíos
    for idx in range(n_models, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis('off')

    plt.suptitle(f'Matrices de Confusión ({dataset_label})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Curvas de entrenamiento
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))

    for idx, r in enumerate(results_list):
        if 'history' not in r:
            continue
        history = r['history']
        epochs = range(1, len(history['train_loss']) + 1)
        label = r['model_display_name']
        color = colors[idx]

        # Train Loss
        axes[0, 0].plot(epochs, history['train_loss'], label=label, color=color)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

        # Val Loss
        axes[0, 1].plot(epochs, history['val_loss'], label=label, color=color)
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

        # Val Accuracy
        val_acc_pct = [v * 100 for v in history['val_acc']]
        axes[1, 0].plot(epochs, val_acc_pct, label=label, color=color)
        axes[1, 0].set_title('Validation Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

        # Val F1
        val_f1_pct = [v * 100 for v in history['val_f1_macro']]
        axes[1, 1].plot(epochs, val_f1_pct, label=label, color=color)
        axes[1, 1].set_title('Validation F1-Macro')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 (%)')
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Curvas de Entrenamiento ({dataset_label})', fontsize=14)
    plt.tight_layout()
    plt.savefig(figures_dir / 'training_curves.png', dpi=150)
    plt.close()

    logger.info("Figuras guardadas en: %s", figures_dir)


def _generate_comparison_reports(
    results_list: list,
    output_dir: Path,
) -> None:
    """
    Genera reportes JSON y CSV con resultados comparativos.

    Args:
        results_list: Lista de diccionarios con resultados
        output_dir: Directorio de salida
    """
    import json

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Guardar JSON completo
    json_path = output_dir / "comparison_results.json"
    with open(json_path, 'w') as f:
        json.dump(results_list, f, indent=2)
    logger.info("Resultados JSON guardados en: %s", json_path)

    # 2. Crear CSV resumido
    csv_path = output_dir / "comparison_results.csv"

    # Headers
    headers = [
        "Architecture", "Accuracy", "F1_Macro", "F1_Weighted",
        "Epochs", "Training_Time_Min", "Total_Params", "Model_Size_MB"
    ]

    with open(csv_path, 'w') as f:
        f.write(",".join(headers) + "\n")
        for r in results_list:
            row = [
                r['model_display_name'],
                f"{r['test_metrics']['accuracy'] * 100:.2f}",
                f"{r['test_metrics']['f1_macro'] * 100:.2f}",
                f"{r['test_metrics']['f1_weighted'] * 100:.2f}",
                str(r['epochs_trained']),
                f"{r['training_time_seconds'] / 60:.2f}",
                str(r['total_params']),
                f"{r['model_size_mb']:.2f}",
            ]
            f.write(",".join(row) + "\n")

    logger.info("Resultados CSV guardados en: %s", csv_path)


@app.command("compare-architectures")
def compare_architectures(
    data_dir: str = typer.Argument(
        ...,
        help="Directorio del dataset warped (con train/, val/, test/)"
    ),
    output_dir: str = typer.Option(
        "outputs/arch_comparison",
        "--output-dir",
        "-o",
        help="Directorio de salida para resultados"
    ),
    architectures: Optional[str] = typer.Option(
        None,
        "--architectures",
        "-a",
        help="Arquitecturas a comparar (separadas por coma). Default: todas"
    ),
    original_data_dir: Optional[str] = typer.Option(
        None,
        "--original-data-dir",
        help="Dataset original para comparar warped vs original"
    ),
    epochs: int = typer.Option(
        30,
        "--epochs",
        "-e",
        help="Numero de epocas por arquitectura"
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        help="Tamano de batch"
    ),
    lr: float = typer.Option(
        1e-4,
        "--lr",
        help="Learning rate"
    ),
    patience: int = typer.Option(
        10,
        "--patience",
        help="Paciencia para early stopping"
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Semilla aleatoria"
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo: auto, cuda, cpu, mps"
    ),
    quick: bool = typer.Option(
        False,
        "--quick",
        help="Modo rapido: solo 5 epocas (para pruebas)"
    ),
):
    """
    Comparar multiples arquitecturas CNN para clasificacion COVID-19.

    Entrena y evalua sistematicamente multiples arquitecturas en el mismo
    dataset, generando reportes comparativos y visualizaciones.

    Arquitecturas soportadas: resnet18, resnet50, efficientnet_b0,
    densenet121, alexnet, vgg16, mobilenet_v2

    Ejemplos:
        # Comparar todas las arquitecturas
        python -m src_v2 compare-architectures outputs/warped_lung_best/session_warping \\
            --epochs 30 --seed 42

        # Comparar arquitecturas especificas
        python -m src_v2 compare-architectures outputs/warped_lung_best/session_warping \\
            --architectures resnet18,efficientnet_b0,densenet121

        # Modo rapido para pruebas
        python -m src_v2 compare-architectures outputs/warped_lung_best/session_warping --quick

        # Comparar warped vs original
        python -m src_v2 compare-architectures outputs/warped_lung_best/session_warping \\
            --original-data-dir data/dataset/COVID-19_Radiography_Dataset
    """
    import json
    import time
    from collections import Counter
    from datetime import datetime

    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import datasets
    from PIL import Image
    from tqdm import tqdm

    from src_v2.models import get_classifier_transforms, get_class_weights

    logger.info("=" * 60)
    logger.info("COMPARE ARCHITECTURES - COVID-19 Classification")
    logger.info("=" * 60)

    # Modo rapido (usa constantes de constants.py para consistencia)
    # Nota: Usamos min() para respetar el valor del usuario si es menor
    if quick:
        epochs = min(epochs, QUICK_MODE_EPOCHS_COMPARE)
        logger.info("Modo rápido activado: %d épocas", epochs)

    # Parsear arquitecturas
    if architectures:
        arch_list = [a.strip().lower() for a in architectures.split(",")]
        # Validar arquitecturas
        invalid = [a for a in arch_list if a not in SUPPORTED_ARCHITECTURES]
        if invalid:
            logger.error("Arquitecturas no soportadas: %s", invalid)
            logger.info("Hint: Opciones válidas: %s", SUPPORTED_ARCHITECTURES)
            raise typer.Exit(code=1)
    else:
        arch_list = SUPPORTED_ARCHITECTURES.copy()

    logger.info("Arquitecturas a comparar: %s", arch_list)

    # Configurar semilla completa para reproducibilidad
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Verificar directorios
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    test_dir = data_path / "test"

    for d in [train_dir, val_dir, test_dir]:
        if not d.exists():
            logger.error("Directorio no existe: %s", d)
            raise typer.Exit(code=1)

    # Crear directorio de salida
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "training_logs").mkdir(exist_ok=True)
    (output_path / "checkpoints").mkdir(exist_ok=True)

    # Dispositivo
    torch_device = get_device(device)
    logger.info("Dispositivo: %s", torch_device)

    # Transforms
    train_transform = get_classifier_transforms(train=True, img_size=DEFAULT_IMAGE_SIZE)
    eval_transform = get_classifier_transforms(train=False, img_size=DEFAULT_IMAGE_SIZE)

    # Datasets warped
    logger.info("Cargando dataset warped...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    class_names = train_dataset.classes
    logger.info("Clases: %s", class_names)
    logger.info("Train: %d, Val: %d, Test: %d",
                len(train_dataset), len(val_dataset), len(test_dataset))

    # Class weights
    train_labels = [train_dataset.targets[i] for i in range(len(train_dataset))]
    class_weights_tensor = get_class_weights(train_labels, len(class_names)).to(torch_device)
    logger.info("Class weights: %s", class_weights_tensor.cpu().numpy())

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=get_optimal_num_workers())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=get_optimal_num_workers())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=get_optimal_num_workers())

    # -------------------------------------------------------------------------
    # Entrenar en dataset warped
    # -------------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("ENTRENAMIENTO EN DATASET WARPED")
    logger.info("-" * 60)

    warped_results = []
    total_start = time.time()

    for arch in tqdm(arch_list, desc="Arquitecturas", unit="modelo"):
        arch_display = ARCHITECTURE_DISPLAY_NAMES.get(arch, arch)
        logger.info("")
        logger.info("Entrenando %s...", arch_display)

        try:
            results, model_state = _train_single_architecture(
                model_name=arch,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                class_names=class_names,
                class_weights_tensor=class_weights_tensor,
                torch_device=torch_device,
                epochs=epochs,
                lr=lr,
                patience=patience,
            )

            results['dataset'] = 'warped'
            warped_results.append(results)

            # Guardar log individual
            log_path = output_path / "training_logs" / f"{arch}_warped_log.json"
            with open(log_path, 'w') as f:
                json.dump(results, f, indent=2)

            # Guardar checkpoint
            ckpt_path = output_path / "checkpoints" / f"{arch}_warped_best.pt"
            torch.save({
                'model_state_dict': model_state,
                'model_name': arch,
                'class_names': class_names,
                'best_val_f1': results['best_val_f1'],
            }, ckpt_path)

            logger.info("    Test Accuracy: %.2f%%, F1: %.2f%%, Tiempo: %.1f min",
                       results['test_metrics']['accuracy'] * 100,
                       results['test_metrics']['f1_macro'] * 100,
                       results['training_time_seconds'] / 60)

        except Exception as e:
            logger.error("    Error entrenando %s: %s", arch, e)
            import traceback
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # Entrenar en dataset original (si se especifica)
    # -------------------------------------------------------------------------
    original_results = []

    if original_data_dir:
        original_path = Path(original_data_dir)
        if not original_path.exists():
            logger.warning("Dataset original no encontrado: %s", original_path)
        else:
            logger.info("")
            logger.info("-" * 60)
            logger.info("ENTRENAMIENTO EN DATASET ORIGINAL")
            logger.info("-" * 60)

            # Dataset para cargar imagenes originales usando CSVs del warped
            class OriginalDataset(Dataset):
                """Dataset que carga imagenes originales usando splits del warped."""

                def __init__(self, split_csv, original_dir, class_names_list, transform=None):
                    self.original_dir = Path(original_dir)
                    self.transform = transform
                    # Usar class_names del dataset warped para consistencia
                    self.classes = class_names_list
                    self.class_to_idx = {c: i for i, c in enumerate(class_names_list)}

                    # Validar CSV antes de leer
                    try:
                        self.df = pd.read_csv(split_csv)
                        if 'image_name' not in self.df.columns or 'category' not in self.df.columns:
                            raise ValueError(f"CSV {split_csv} no tiene columnas 'image_name' o 'category'")
                    except Exception as e:
                        raise ValueError(f"Error leyendo CSV {split_csv}: {e}")

                    self.samples = []
                    self.targets = []
                    missing_count = 0

                    for _, row in self.df.iterrows():
                        image_name = row['image_name']
                        category = row['category']

                        if category not in self.class_to_idx:
                            continue  # Saltar categorías desconocidas

                        # Buscar imagen original (intentar múltiples extensiones)
                        found = False
                        for ext in ['.png', '.jpg', '.jpeg']:
                            img_path = self.original_dir / category / f"{image_name}{ext}"
                            if img_path.exists():
                                self.samples.append((img_path, self.class_to_idx[category]))
                                self.targets.append(self.class_to_idx[category])
                                found = True
                                break
                        if not found:
                            missing_count += 1

                    if missing_count > 0:
                        logger.warning("OriginalDataset: %d/%d imagenes no encontradas",
                                      missing_count, len(self.df))

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    img_path, label = self.samples[idx]
                    image = Image.open(img_path).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    return image, label

            # Cargar datasets originales
            try:
                train_csv = train_dir / "images.csv"
                val_csv = val_dir / "images.csv"
                test_csv = test_dir / "images.csv"

                if train_csv.exists() and val_csv.exists() and test_csv.exists():
                    orig_train = OriginalDataset(train_csv, original_path, class_names, train_transform)
                    orig_val = OriginalDataset(val_csv, original_path, class_names, eval_transform)
                    orig_test = OriginalDataset(test_csv, original_path, class_names, eval_transform)

                    logger.info("Original - Train: %d, Val: %d, Test: %d",
                               len(orig_train), len(orig_val), len(orig_test))

                    if len(orig_train) > 0:
                        orig_train_loader = DataLoader(orig_train, batch_size=batch_size,
                                                       shuffle=True, num_workers=get_optimal_num_workers())
                        orig_val_loader = DataLoader(orig_val, batch_size=batch_size,
                                                     shuffle=False, num_workers=get_optimal_num_workers())
                        orig_test_loader = DataLoader(orig_test, batch_size=batch_size,
                                                      shuffle=False, num_workers=get_optimal_num_workers())

                        # Class weights para original
                        orig_labels = orig_train.targets
                        orig_weights = get_class_weights(orig_labels, len(class_names)).to(torch_device)

                        for arch in tqdm(arch_list, desc="Arquitecturas (original)", unit="modelo"):
                            arch_display = ARCHITECTURE_DISPLAY_NAMES.get(arch, arch)
                            logger.info("")
                            logger.info("Entrenando %s (original)...", arch_display)

                            try:
                                results, model_state = _train_single_architecture(
                                    model_name=arch,
                                    train_loader=orig_train_loader,
                                    val_loader=orig_val_loader,
                                    test_loader=orig_test_loader,
                                    class_names=class_names,
                                    class_weights_tensor=orig_weights,
                                    torch_device=torch_device,
                                    epochs=epochs,
                                    lr=lr,
                                    patience=patience,
                                )

                                results['dataset'] = 'original'
                                original_results.append(results)

                                # Guardar log
                                log_path = output_path / "training_logs" / f"{arch}_original_log.json"
                                with open(log_path, 'w') as f:
                                    json.dump(results, f, indent=2)

                                # Guardar checkpoint
                                ckpt_path = output_path / "checkpoints" / f"{arch}_original_best.pt"
                                torch.save({
                                    'model_state_dict': model_state,
                                    'model_name': arch,
                                    'class_names': class_names,
                                    'best_val_f1': results['best_val_f1'],
                                }, ckpt_path)

                                logger.info("    Test Accuracy: %.2f%%, F1: %.2f%%",
                                           results['test_metrics']['accuracy'] * 100,
                                           results['test_metrics']['f1_macro'] * 100)

                            except Exception as e:
                                logger.error("    Error: %s", e)
                else:
                    logger.warning("CSVs de splits no encontrados en dataset warped")
            except Exception as e:
                logger.error("Error cargando dataset original: %s", e)

    total_time = time.time() - total_start

    # -------------------------------------------------------------------------
    # Generar reportes y visualizaciones
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("-" * 60)
    logger.info("GENERANDO REPORTES")
    logger.info("-" * 60)

    # Combinar todos los resultados
    all_results = warped_results + original_results

    # Generar reportes
    _generate_comparison_reports(all_results, output_path)

    # Generar figuras para warped
    if warped_results:
        _generate_comparison_figures(warped_results, output_path, "warped")

    # Generar figuras para original si existe
    if original_results:
        _generate_comparison_figures(original_results, output_path, "original")

    # -------------------------------------------------------------------------
    # Resumen final
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESUMEN COMPARATIVO")
    logger.info("=" * 60)

    if warped_results:
        logger.info("")
        logger.info("Dataset WARPED:")
        logger.info("%-15s %10s %10s %10s", "Arquitectura", "Accuracy", "F1-Macro", "Tiempo")
        logger.info("-" * 50)
        for r in sorted(warped_results, key=lambda x: -x['test_metrics']['f1_macro']):
            logger.info("%-15s %9.2f%% %9.2f%% %8.1f min",
                       r['model_display_name'][:15],
                       r['test_metrics']['accuracy'] * 100,
                       r['test_metrics']['f1_macro'] * 100,
                       r['training_time_seconds'] / 60)

        best_warped = max(warped_results, key=lambda x: x['test_metrics']['f1_macro'])
        logger.info("")
        logger.info("Mejor (warped): %s con F1=%.2f%%",
                   best_warped['model_display_name'],
                   best_warped['test_metrics']['f1_macro'] * 100)

    if original_results:
        logger.info("")
        logger.info("Dataset ORIGINAL:")
        logger.info("%-15s %10s %10s %10s", "Arquitectura", "Accuracy", "F1-Macro", "Tiempo")
        logger.info("-" * 50)
        for r in sorted(original_results, key=lambda x: -x['test_metrics']['f1_macro']):
            logger.info("%-15s %9.2f%% %9.2f%% %8.1f min",
                       r['model_display_name'][:15],
                       r['test_metrics']['accuracy'] * 100,
                       r['test_metrics']['f1_macro'] * 100,
                       r['training_time_seconds'] / 60)

    logger.info("")
    logger.info("Tiempo total: %.1f minutos", total_time / 60)
    logger.info("Resultados guardados en: %s", output_path)
    logger.info("=" * 60)
    logger.info("Comparacion completada!")
    logger.info("=" * 60)


@app.command("gradcam")
def gradcam(
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint",
        "-c",
        help="Path al checkpoint del clasificador"
    ),
    image: Optional[str] = typer.Option(
        None,
        "--image",
        "-i",
        help="Imagen individual a visualizar"
    ),
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        "-d",
        help="Directorio con imagenes (modo batch)"
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Archivo de salida (imagen individual)"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Directorio de salida (modo batch)"
    ),
    layer: str = typer.Option(
        "auto",
        "--layer",
        "-l",
        help="Capa para Grad-CAM (auto detecta segun arquitectura)"
    ),
    num_samples: int = typer.Option(
        10,
        "--num-samples",
        "-n",
        help="Numero de muestras por clase (modo batch)"
    ),
    colormap: str = typer.Option(
        "jet",
        "--colormap",
        help="Mapa de colores: jet, hot, viridis"
    ),
    alpha: float = typer.Option(
        0.5,
        "--alpha",
        help="Transparencia del heatmap (0-1)"
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo: auto, cuda, cpu, mps"
    ),
):
    """
    Generar visualizaciones Grad-CAM para explicabilidad del clasificador.

    Muestra que regiones de la imagen influyen en la prediccion del modelo.

    Ejemplo imagen individual:
        python -m src_v2 gradcam \\
            --checkpoint outputs/classifier/best.pt \\
            --image test.png --output gradcam.png

    Ejemplo batch:
        python -m src_v2 gradcam \\
            --checkpoint outputs/classifier/best.pt \\
            --data-dir outputs/warped_lung_best/session_warping/test \\
            --output-dir outputs/gradcam_analysis \\
            --num-samples 20
    """
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import datasets

    from src_v2.models import create_classifier, get_classifier_transforms
    from src_v2.visualization.gradcam import (
        GradCAM,
        get_target_layer,
        overlay_heatmap,
        create_gradcam_visualization,
        TARGET_LAYER_MAP,
    )

    logger.info("=" * 60)
    logger.info("Grad-CAM Visualization")
    logger.info("=" * 60)

    # Validar parametros
    if image is None and data_dir is None:
        logger.error("Debe especificar --image o --data-dir")
        raise typer.Exit(code=1)

    if image is not None and data_dir is not None:
        logger.error("Especifique solo --image o --data-dir, no ambos")
        raise typer.Exit(code=1)

    if image is not None and output is None:
        logger.error("--image requiere --output")
        raise typer.Exit(code=1)

    if data_dir is not None and output_dir is None:
        logger.error("--data-dir requiere --output-dir")
        raise typer.Exit(code=1)

    if not Path(checkpoint).exists():
        logger.error("Checkpoint no existe: %s", checkpoint)
        raise typer.Exit(code=1)

    if image is not None and not Path(image).exists():
        logger.error("Imagen no existe: %s", image)
        raise typer.Exit(code=1)

    if data_dir is not None and not Path(data_dir).exists():
        logger.error("Directorio no existe: %s", data_dir)
        raise typer.Exit(code=1)

    # Validar parametros numericos
    if not (0.0 <= alpha <= 1.0):
        logger.error("--alpha debe estar entre 0.0 y 1.0, recibido: %f", alpha)
        raise typer.Exit(code=1)

    if num_samples <= 0:
        logger.error("--num-samples debe ser mayor que 0, recibido: %d", num_samples)
        raise typer.Exit(code=1)

    # Validar layer
    if layer != "auto" and layer not in TARGET_LAYER_MAP.values():
        valid_layers = list(set(TARGET_LAYER_MAP.values()))
        logger.warning("Capa '%s' no es estandar. Capas comunes: %s", layer, valid_layers)

    # Dispositivo
    torch_device = get_device(device)
    logger.info("Dispositivo: %s", torch_device)

    # Cargar modelo
    logger.info("Cargando modelo desde: %s", checkpoint)
    model = create_classifier(checkpoint=checkpoint, device=torch_device)
    model.eval()

    # Obtener metadata del checkpoint
    ckpt_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    class_names = ckpt_data.get("class_names", CLASSIFIER_CLASSES)
    backbone_name = model.backbone_name
    logger.info("Arquitectura detectada: %s", backbone_name)
    logger.info("Clases: %s", class_names)

    # Obtener target layer
    try:
        layer_name = layer if layer != "auto" else None
        target_layer = get_target_layer(model, backbone_name, layer_name)
        logger.info("Capa objetivo: %s", TARGET_LAYER_MAP.get(backbone_name, layer))
    except ValueError as e:
        logger.error("Error obteniendo capa: %s", e)
        raise typer.Exit(code=1)

    # Transforms
    transform = get_classifier_transforms(train=False, img_size=DEFAULT_IMAGE_SIZE)

    # Crear GradCAM
    gradcam = GradCAM(model, target_layer)

    try:
        if image is not None:
            # Modo imagen individual
            logger.info("Procesando imagen: %s", image)

            # Cargar imagen y convertir a RGB
            img_pil = Image.open(image).convert('RGB')
            img_array = np.array(img_pil)

            # Transformar para modelo
            img_tensor = transform(img_pil).unsqueeze(0).to(torch_device)

            # Generar GradCAM
            heatmap, pred_class, confidence = gradcam(img_tensor)

            # Crear visualizacion
            visualization = create_gradcam_visualization(
                image=img_array,
                heatmap=heatmap,
                prediction=class_names[pred_class],
                confidence=confidence,
                alpha=alpha,
                colormap=colormap,
            )

            # Guardar
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

            logger.info("Prediccion: %s (%.1f%%)", class_names[pred_class], confidence * 100)
            logger.info("Visualizacion guardada: %s", output_path)

        else:
            # Modo batch
            logger.info("Procesando directorio: %s", data_dir)

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Cargar dataset
            dataset = datasets.ImageFolder(data_dir, transform=transform)
            dataset_classes = dataset.classes

            logger.info("Clases encontradas: %s", dataset_classes)
            logger.info("Total imagenes: %d", len(dataset))

            # Agrupar por clase
            from collections import defaultdict
            samples_by_class = defaultdict(list)
            for idx, (path, label) in enumerate(dataset.samples):
                samples_by_class[dataset_classes[label]].append((idx, path, label))

            # Procesar muestras por clase
            total_processed = 0
            for class_name in dataset_classes:
                class_samples = samples_by_class[class_name][:num_samples]
                class_dir = output_path / class_name
                class_dir.mkdir(parents=True, exist_ok=True)

                logger.info("Procesando clase %s (%d muestras)...", class_name, len(class_samples))

                for idx, img_path, label in class_samples:
                    try:
                        # Cargar imagen original y convertir a RGB
                        img_pil = Image.open(img_path).convert('RGB')
                        img_array = np.array(img_pil)

                        # Transformar
                        img_tensor, _ = dataset[idx]
                        img_tensor = img_tensor.unsqueeze(0).to(torch_device)

                        # Generar GradCAM
                        heatmap, pred_class, confidence = gradcam(img_tensor)

                        # Crear visualizacion
                        visualization = create_gradcam_visualization(
                            image=img_array,
                            heatmap=heatmap,
                            prediction=class_names[pred_class],
                            confidence=confidence,
                            true_label=class_name,
                            alpha=alpha,
                            colormap=colormap,
                        )

                        # Guardar
                        img_name = Path(img_path).stem
                        out_file = class_dir / f"{img_name}_gradcam.png"
                        cv2.imwrite(str(out_file), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
                        total_processed += 1

                    except Exception as e:
                        logger.warning("Error procesando %s: %s", img_path, e)

            logger.info("Total procesadas: %d imagenes", total_processed)
            logger.info("Resultados guardados en: %s", output_path)

    finally:
        # Limpiar hooks para evitar memory leaks
        gradcam.remove_hooks()
        # Limpiar memoria GPU
        if torch_device.type == "cuda":
            del model
            torch.cuda.empty_cache()

    logger.info("=" * 60)
    logger.info("Grad-CAM completado!")
    logger.info("=" * 60)


@app.command("analyze-errors")
def analyze_errors(
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint",
        "-c",
        help="Path al checkpoint del clasificador"
    ),
    data_dir: str = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Directorio del dataset de test"
    ),
    output_dir: str = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Directorio de salida para reportes"
    ),
    visualize: bool = typer.Option(
        False,
        "--visualize/--no-visualize",
        help="Generar visualizaciones"
    ),
    gradcam_errors: bool = typer.Option(
        False,
        "--gradcam/--no-gradcam",
        help="Generar Grad-CAM para errores"
    ),
    top_k: int = typer.Option(
        20,
        "--top-k",
        "-k",
        help="Top K errores por categoria a analizar"
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo: auto, cuda, cpu, mps"
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Tamano de batch"
    ),
):
    """
    Analizar errores de clasificacion.

    Genera reportes detallados de errores, incluyendo:
    - error_summary.json: Resumen de errores
    - error_details.csv: Detalles por imagen
    - confusion_analysis.json: Analisis de confusion
    - Visualizaciones opcionales

    Ejemplo:
        python -m src_v2 analyze-errors \\
            --checkpoint outputs/classifier/best.pt \\
            --data-dir outputs/warped_lung_best/session_warping/test \\
            --output-dir outputs/error_analysis \\
            --visualize --gradcam
    """
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from tqdm import tqdm

    from src_v2.models import create_classifier, get_classifier_transforms
    from src_v2.visualization.error_analysis import (
        ErrorAnalyzer,
        create_error_visualizations,
    )

    logger.info("=" * 60)
    logger.info("Classification Error Analysis")
    logger.info("=" * 60)

    # Validar paths
    if not Path(checkpoint).exists():
        logger.error("Checkpoint no existe: %s", checkpoint)
        raise typer.Exit(code=1)

    if not Path(data_dir).exists():
        logger.error("Directorio de datos no existe: %s", data_dir)
        raise typer.Exit(code=1)

    # Validar parametros numericos
    if batch_size <= 0:
        logger.error("--batch-size debe ser mayor que 0, recibido: %d", batch_size)
        raise typer.Exit(code=1)

    if top_k <= 0:
        logger.error("--top-k debe ser mayor que 0, recibido: %d", top_k)
        raise typer.Exit(code=1)

    # Dispositivo
    torch_device = get_device(device)
    logger.info("Dispositivo: %s", torch_device)

    # Cargar modelo
    logger.info("Cargando modelo desde: %s", checkpoint)
    model = create_classifier(checkpoint=checkpoint, device=torch_device)
    model.eval()

    # Obtener metadata del checkpoint
    ckpt_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    class_names = ckpt_data.get("class_names", CLASSIFIER_CLASSES)
    backbone_name = model.backbone_name
    logger.info("Arquitectura: %s", backbone_name)
    logger.info("Clases: %s", class_names)

    # Transforms
    transform = get_classifier_transforms(train=False, img_size=DEFAULT_IMAGE_SIZE)

    # Cargar dataset
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=get_optimal_num_workers())
    logger.info("Total muestras: %d", len(dataset))

    # Crear analizador
    analyzer = ErrorAnalyzer(class_names)

    # Evaluar
    logger.info("Analizando predicciones...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Evaluando")):
            images = images.to(torch_device)
            outputs = model(images)

            # Obtener paths para este batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            batch_paths = [dataset.samples[i][0] for i in range(start_idx, end_idx)]

            # Agregar al analizador
            analyzer.add_batch(outputs, labels, batch_paths)

    # Obtener resumen
    summary = analyzer.get_summary()

    logger.info("")
    logger.info("-" * 60)
    logger.info("RESUMEN DE ERRORES")
    logger.info("-" * 60)
    logger.info("Total muestras: %d", summary.total_samples)
    logger.info("Total errores: %d (%.1f%%)", summary.total_errors, summary.error_rate * 100)
    logger.info("Confianza promedio (errores): %.2f%%", summary.avg_confidence_errors * 100)
    logger.info("Confianza promedio (correctos): %.2f%%", summary.avg_confidence_correct * 100)

    logger.info("")
    logger.info("Errores por clase verdadera:")
    for cls, count in sorted(summary.errors_by_true_class.items()):
        logger.info("  %s: %d", cls, count)

    logger.info("")
    logger.info("Pares de confusion mas frecuentes:")
    for pair, count in sorted(summary.confusion_pairs.items(), key=lambda x: -x[1])[:5]:
        logger.info("  %s: %d", pair, count)

    # Guardar reportes
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("")
    logger.info("-" * 60)
    logger.info("GUARDANDO REPORTES")
    logger.info("-" * 60)

    saved_files = analyzer.save_reports(output_path)
    for name, path in saved_files.items():
        logger.info("  %s: %s", name, path)

    # Visualizaciones opcionales
    if visualize:
        logger.info("")
        logger.info("Generando visualizaciones...")
        viz_files = create_error_visualizations(
            analyzer,
            output_path,
            copy_images=True,
        )
        for name, path in viz_files.items():
            logger.info("  %s: %s", name, path)

    # Grad-CAM para errores
    if gradcam_errors:
        logger.info("")
        logger.info("-" * 60)
        logger.info("GENERANDO GRAD-CAM PARA ERRORES")
        logger.info("-" * 60)

        from src_v2.visualization.gradcam import (
            GradCAM,
            get_target_layer,
            create_gradcam_visualization,
        )

        # Obtener target layer
        target_layer = get_target_layer(model, backbone_name)
        gradcam = GradCAM(model, target_layer)

        gradcam_dir = output_path / "gradcam_errors"

        # Procesar top-K errores por par de confusion
        top_errors = analyzer.get_top_errors(k=top_k, descending=True)
        logger.info("Procesando top %d errores con mayor confianza...", min(top_k, len(top_errors)))

        processed = 0
        try:
            for error in top_errors:
                try:
                    # Crear directorio por par de confusion
                    pair_dir = gradcam_dir / f"{error.true_class}_as_{error.predicted_class}"
                    pair_dir.mkdir(parents=True, exist_ok=True)

                    # Cargar imagen y convertir a RGB
                    img_pil = Image.open(error.image_path).convert('RGB')
                    img_array = np.array(img_pil)
                    img_tensor = transform(img_pil).unsqueeze(0).to(torch_device)

                    # Generar GradCAM
                    heatmap, pred_class, confidence = gradcam(img_tensor)

                    # Crear visualizacion
                    visualization = create_gradcam_visualization(
                        image=img_array,
                        heatmap=heatmap,
                        prediction=error.predicted_class,
                        confidence=confidence,
                        true_label=error.true_class,
                    )

                    # Guardar
                    img_name = Path(error.image_path).stem
                    out_file = pair_dir / f"{img_name}_conf{error.confidence:.2f}_gradcam.png"
                    cv2.imwrite(str(out_file), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
                    processed += 1

                except Exception as e:
                    logger.warning("Error procesando %s: %s", error.image_path, e)
        finally:
            # Asegurar limpieza de hooks incluso si hay errores
            gradcam.remove_hooks()

        logger.info("Grad-CAM generados: %d imagenes en %s", processed, gradcam_dir)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Analisis de errores completado!")
    logger.info("=" * 60)
    logger.info("Resultados en: %s", output_path)

    # Limpieza de memoria GPU
    if torch_device.type == "cuda":
        del model
        torch.cuda.empty_cache()
        logger.debug("Memoria GPU liberada")


@app.command("pfs-analysis")
def pfs_analysis(
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint",
        "-c",
        help="Path al checkpoint del clasificador"
    ),
    data_dir: str = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Directorio del dataset de test"
    ),
    output_dir: str = typer.Option(
        "outputs/pfs_analysis",
        "--output-dir",
        "-o",
        help="Directorio de salida para reportes"
    ),
    mask_dir: Optional[str] = typer.Option(
        None,
        "--mask-dir",
        "-m",
        help="Directorio con mascaras pulmonares"
    ),
    num_samples: int = typer.Option(
        50,
        "--num-samples",
        "-n",
        help="Muestras por clase a analizar"
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        "-t",
        help="Umbral PFS minimo aceptable"
    ),
    approximate_masks: bool = typer.Option(
        False,
        "--approximate/--no-approximate",
        help="Usar mascaras rectangulares aproximadas si no hay mascaras"
    ),
    margin: float = typer.Option(
        0.15,
        "--margin",
        help="Margen para mascaras aproximadas (0-0.5)"
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        "-b",
        help="Tamano de batch (recomendado 1 para GradCAM)"
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo: auto, cuda, cpu, mps"
    ),
):
    """
    Analizar Pulmonary Focus Score (PFS) del clasificador.

    PFS mide que fraccion de la atencion del modelo se enfoca en
    las regiones pulmonares. Valores mas altos indican mejor enfoque.

    Interpretacion:
    - PFS > 0.7: Excelente enfoque pulmonar
    - PFS 0.5-0.7: Aceptable
    - PFS < 0.5: Preocupante, el modelo mira otras regiones

    Ejemplo basico:
        python -m src_v2 pfs-analysis \\
            --checkpoint outputs/classifier/best.pt \\
            --data-dir outputs/warped_lung_best/session_warping/test \\
            --mask-dir data/dataset/COVID-19_Radiography_Dataset

    Con mascaras aproximadas (sin mascaras reales):
        python -m src_v2 pfs-analysis \\
            --checkpoint outputs/classifier/best.pt \\
            --data-dir outputs/warped_lung_best/session_warping/test \\
            --approximate --margin 0.15
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from tqdm import tqdm

    from src_v2.models import create_classifier, get_classifier_transforms
    from src_v2.visualization.pfs_analysis import (
        PFSAnalyzer,
        PFSResult,
        run_pfs_analysis,
        create_pfs_visualizations,
        save_low_pfs_gradcam_samples,
        load_lung_mask,
        find_mask_for_image,
        generate_approximate_mask,
    )
    from src_v2.visualization.gradcam import GradCAM, get_target_layer, calculate_pfs

    logger.info("=" * 60)
    logger.info("PULMONARY FOCUS SCORE (PFS) ANALYSIS")
    logger.info("=" * 60)

    # Validar paths
    if not Path(checkpoint).exists():
        logger.error("Checkpoint no existe: %s", checkpoint)
        raise typer.Exit(code=1)

    if not Path(data_dir).exists():
        logger.error("Directorio de datos no existe: %s", data_dir)
        raise typer.Exit(code=1)

    # Validar que hay manera de obtener mascaras
    if mask_dir is None and not approximate_masks:
        logger.error("Debe especificar --mask-dir o usar --approximate")
        raise typer.Exit(code=1)

    if mask_dir and not Path(mask_dir).exists():
        logger.error("Directorio de mascaras no existe: %s", mask_dir)
        raise typer.Exit(code=1)

    # Validar parametros numericos
    if num_samples <= 0:
        logger.error("--num-samples debe ser mayor que 0, recibido: %d", num_samples)
        raise typer.Exit(code=1)

    if not (0.0 <= threshold <= 1.0):
        logger.error("--threshold debe estar en [0, 1], recibido: %.2f", threshold)
        raise typer.Exit(code=1)

    if not (0.0 <= margin < 0.5):
        logger.error("--margin debe estar en [0, 0.5), recibido: %.2f", margin)
        raise typer.Exit(code=1)

    if batch_size <= 0:
        logger.error("--batch-size debe ser mayor que 0, recibido: %d", batch_size)
        raise typer.Exit(code=1)

    # Dispositivo
    torch_device = get_device(device)
    logger.info("Dispositivo: %s", torch_device)

    # Cargar modelo
    logger.info("Cargando modelo desde: %s", checkpoint)
    model = create_classifier(checkpoint=checkpoint, device=torch_device)
    model.eval()

    # Obtener metadata del checkpoint
    ckpt_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    class_names = ckpt_data.get("class_names", CLASSIFIER_CLASSES)
    backbone_name = model.backbone_name

    logger.info("Arquitectura: %s", backbone_name)
    logger.info("Clases: %s", class_names)
    logger.info("Umbral PFS: %.2f", threshold)
    logger.info("Muestras por clase: %d", num_samples)

    if mask_dir:
        logger.info("Directorio mascaras: %s", mask_dir)
    if approximate_masks:
        logger.info("Mascaras aproximadas habilitadas (margen: %.2f)", margin)

    # Transforms
    transform = get_classifier_transforms(train=False, img_size=DEFAULT_IMAGE_SIZE)

    # Cargar dataset
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=get_optimal_num_workers())
    logger.info("Total muestras disponibles: %d", len(dataset))

    # Obtener target layer para GradCAM
    target_layer = get_target_layer(model, backbone_name)

    # Session 31 fix: Inicializar GradCAM (faltaba)
    gradcam = GradCAM(model, target_layer)

    # ADVERTENCIA: PFS con imágenes warped
    if "warped" in str(data_dir).lower():
        # Session 31 fix: Usar typer.echo en lugar de console.print
        typer.echo(
            "⚠️  ADVERTENCIA: Las imágenes parecen ser warped."
        )
        typer.echo(
            "   Las máscaras pulmonares NO están warped, lo que puede causar"
        )
        typer.echo(
            "   desalineación geométrica y PFS inexacto."
        )
        typer.echo(
            "   Para PFS preciso, use imágenes originales (no warped)."
        )
        typer.echo("")

    # Crear analizador
    analyzer = PFSAnalyzer(class_names, threshold=threshold)
    detailed_results = []

    # Contadores por clase
    # Dividir num_samples entre las clases para obtener el total deseado
    samples_per_class = {c: 0 for c in class_names}
    max_per_class = max(1, num_samples // len(class_names))

    logger.info("")
    logger.info("-" * 60)
    logger.info("PROCESANDO IMAGENES")
    logger.info("-" * 60)

    try:
        total_processed = 0
        total_skipped = 0

        # Total esperado: max_per_class por cada clase
        expected_total = max_per_class * len(class_names)
        with tqdm(total=expected_total, desc="Analizando PFS") as pbar:
            for batch_idx, (images, labels) in enumerate(dataloader):
                # Procesar una imagen a la vez (requerimiento de GradCAM)
                for i in range(images.shape[0]):
                    image = images[i:i+1].to(torch_device)
                    label = labels[i].item()
                    true_class = class_names[label]

                    # Verificar limite por clase
                    if samples_per_class[true_class] >= max_per_class:
                        continue

                    # Obtener path de imagen
                    start_idx = batch_idx * batch_size
                    img_path = dataset.samples[start_idx + i][0]

                    try:
                        # Generar GradCAM
                        heatmap, pred_class, confidence = gradcam(image)
                        predicted_class = class_names[pred_class]

                        # Obtener o generar mascara
                        mask = None
                        if mask_dir:
                            mask_path = find_mask_for_image(img_path, mask_dir, true_class)
                            if mask_path:
                                mask = load_lung_mask(mask_path)

                        if mask is None:
                            if approximate_masks:
                                mask = generate_approximate_mask(heatmap.shape, margin)
                            else:
                                logger.debug("Sin mascara para %s, saltando", img_path)
                                total_skipped += 1
                                continue

                        # Calcular PFS
                        pfs = calculate_pfs(heatmap, mask)

                        # Guardar resultado
                        result = PFSResult(
                            image_path=str(img_path),
                            true_class=true_class,
                            predicted_class=predicted_class,
                            confidence=confidence,
                            pfs=pfs,
                            correct=(pred_class == label),
                        )
                        analyzer.add_result(result)

                        detailed_results.append({
                            "image_path": str(img_path),
                            "heatmap": heatmap,
                            "mask": mask,
                            "pfs": pfs,
                            "result": result,
                        })

                        samples_per_class[true_class] += 1
                        total_processed += 1
                        pbar.update(1)

                    except Exception as e:
                        logger.warning("Error procesando %s: %s", img_path, str(e))
                        total_skipped += 1
                        continue

                # Verificar si tenemos suficientes muestras
                if all(c >= max_per_class for c in samples_per_class.values()):
                    break

        logger.info("Procesadas: %d imagenes", total_processed)
        if total_skipped > 0:
            logger.info("Saltadas: %d imagenes (sin mascara)", total_skipped)

    finally:
        # Limpiar hooks
        gradcam.remove_hooks()

    # Obtener resumen
    if not analyzer.results:
        logger.error("No se procesaron imagenes. Verifique las mascaras.")
        raise typer.Exit(code=1)

    summary = analyzer.get_summary()

    logger.info("")
    logger.info("-" * 60)
    logger.info("RESUMEN PFS")
    logger.info("-" * 60)
    logger.info("Total muestras: %d", summary.total_samples)
    logger.info("PFS promedio: %.4f (+/- %.4f)", summary.mean_pfs, summary.std_pfs)
    logger.info("PFS mediana: %.4f", summary.median_pfs)
    logger.info("PFS rango: [%.4f, %.4f]", summary.min_pfs, summary.max_pfs)
    logger.info("")
    logger.info("Muestras con PFS < %.2f: %d (%.1f%%)",
                threshold, summary.low_pfs_count, summary.low_pfs_rate * 100)

    logger.info("")
    logger.info("PFS por clase:")
    for cls, stats in summary.pfs_by_class.items():
        status = "OK" if stats["mean"] >= threshold else "BAJO"
        logger.info("  %s: %.4f (+/- %.4f) [%s] (n=%d)",
                    cls, stats["mean"], stats["std"], status, stats["count"])

    logger.info("")
    logger.info("PFS predicciones correctas vs incorrectas:")
    logger.info("  Correctas: %.4f (+/- %.4f) (n=%d)",
                summary.pfs_correct_vs_incorrect["correct"]["mean"],
                summary.pfs_correct_vs_incorrect["correct"]["std"],
                summary.pfs_correct_vs_incorrect["correct"]["count"])
    logger.info("  Incorrectas: %.4f (+/- %.4f) (n=%d)",
                summary.pfs_correct_vs_incorrect["incorrect"]["mean"],
                summary.pfs_correct_vs_incorrect["incorrect"]["std"],
                summary.pfs_correct_vs_incorrect["incorrect"]["count"])

    # Guardar reportes
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("")
    logger.info("-" * 60)
    logger.info("GUARDANDO REPORTES")
    logger.info("-" * 60)

    saved_files = analyzer.save_reports(output_path)
    for name, path in saved_files.items():
        logger.info("  %s: %s", name, path)

    # Crear visualizaciones
    logger.info("")
    logger.info("Generando visualizaciones...")
    viz_files = create_pfs_visualizations(detailed_results, output_path, summary)
    for name, path in viz_files.items():
        logger.info("  %s: %s", name, path)

    # Guardar muestras de bajo PFS
    if summary.low_pfs_count > 0:
        logger.info("")
        logger.info("Guardando muestras con PFS bajo...")
        n_saved = save_low_pfs_gradcam_samples(
            detailed_results, output_path, threshold, max_samples=20
        )
        logger.info("  Guardadas %d muestras en low_pfs_samples/", n_saved)

    # Alertas
    logger.info("")
    logger.info("-" * 60)
    if summary.mean_pfs >= 0.7:
        logger.info("RESULTADO: EXCELENTE - El modelo enfoca bien en pulmones")
    elif summary.mean_pfs >= threshold:
        logger.info("RESULTADO: ACEPTABLE - Enfoque pulmonar satisfactorio")
    else:
        logger.warning("RESULTADO: PREOCUPANTE - El modelo enfoca poco en pulmones")
        logger.warning("Revise las visualizaciones en low_pfs_samples/")
    logger.info("-" * 60)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Analisis PFS completado!")
    logger.info("=" * 60)
    logger.info("Resultados en: %s", output_path)

    # Limpieza de memoria GPU
    if torch_device.type == "cuda":
        del model
        torch.cuda.empty_cache()
        logger.debug("Memoria GPU liberada")


@app.command("generate-lung-masks")
def generate_lung_masks(
    data_dir: str = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Directorio del dataset"
    ),
    output_dir: str = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Directorio de salida para mascaras"
    ),
    method: str = typer.Option(
        "rectangular",
        "--method",
        "-m",
        help="Metodo de generacion: rectangular"
    ),
    margin: float = typer.Option(
        0.15,
        "--margin",
        help="Margen para metodo rectangular (0-0.5)"
    ),
):
    """
    Generar mascaras pulmonares aproximadas.

    Crea mascaras binarias rectangulares como aproximacion de
    la region pulmonar cuando no hay mascaras de segmentacion.

    Metodos disponibles:
    - rectangular: Region central de la imagen (mas simple)

    Ejemplo:
        python -m src_v2 generate-lung-masks \\
            --data-dir outputs/warped_lung_best/session_warping \\
            --output-dir outputs/lung_masks \\
            --method rectangular --margin 0.15
    """
    import cv2
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    from src_v2.visualization.pfs_analysis import generate_approximate_mask

    logger.info("=" * 60)
    logger.info("GENERAR MASCARAS PULMONARES")
    logger.info("=" * 60)

    # Validar paths
    if not Path(data_dir).exists():
        logger.error("Directorio de datos no existe: %s", data_dir)
        raise typer.Exit(code=1)

    # Validar metodo
    if method not in ["rectangular"]:
        logger.error("Metodo no soportado: %s. Usar: rectangular", method)
        raise typer.Exit(code=1)

    # Validar margen
    if not (0.0 <= margin < 0.5):
        logger.error("--margin debe estar en [0, 0.5), recibido: %.2f", margin)
        raise typer.Exit(code=1)

    logger.info("Directorio entrada: %s", data_dir)
    logger.info("Directorio salida: %s", output_dir)
    logger.info("Metodo: %s", method)
    logger.info("Margen: %.2f", margin)

    # Buscar imagenes
    data_path = Path(data_dir)
    output_path = Path(output_dir)

    # Buscar todas las imagenes (incluyendo subdirectorios)
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    images = []
    for ext in image_extensions:
        images.extend(data_path.rglob(f"*{ext}"))
        images.extend(data_path.rglob(f"*{ext.upper()}"))

    if not images:
        logger.error("No se encontraron imagenes en %s", data_dir)
        raise typer.Exit(code=1)

    logger.info("Imagenes encontradas: %d", len(images))

    # Generar mascaras
    logger.info("")
    logger.info("-" * 60)
    logger.info("GENERANDO MASCARAS")
    logger.info("-" * 60)

    processed = 0
    errors = 0

    for img_path in tqdm(images, desc="Generando mascaras"):
        try:
            # Cargar imagen para obtener dimensiones
            img = Image.open(img_path)
            width, height = img.size

            # Generar mascara
            if method == "rectangular":
                mask = generate_approximate_mask((height, width), margin)

            # Calcular ruta de salida manteniendo estructura de subdirectorios
            relative_path = img_path.relative_to(data_path)
            mask_path = output_path / relative_path.parent / f"{img_path.stem}_mask.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)

            # Guardar mascara (convertir a 0-255)
            mask_uint8 = (mask * 255).astype(np.uint8)
            cv2.imwrite(str(mask_path), mask_uint8)

            processed += 1

        except Exception as e:
            logger.warning("Error procesando %s: %s", img_path, str(e))
            errors += 1

    logger.info("")
    logger.info("-" * 60)
    logger.info("RESUMEN")
    logger.info("-" * 60)
    logger.info("Mascaras generadas: %d", processed)
    if errors > 0:
        logger.warning("Errores: %d", errors)
    logger.info("Salida: %s", output_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Generacion de mascaras completada!")
    logger.info("=" * 60)


# =============================================================================
# COMANDO: optimize-margin
# =============================================================================


@app.command("optimize-margin")
def optimize_margin(
    data_dir: str = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Directorio del dataset original (COVID-19_Radiography_Dataset)",
    ),
    landmarks_csv: str = typer.Option(
        ...,
        "--landmarks-csv",
        "-l",
        help="Archivo CSV con landmarks predichos o ground truth",
    ),
    margins: str = typer.Option(
        "1.00,1.05,1.10,1.15,1.20,1.25,1.30",
        "--margins",
        "-m",
        help="Lista de márgenes a probar, separados por comas",
    ),
    epochs: int = typer.Option(
        10,
        "--epochs",
        "-e",
        help="Número de épocas por entrenamiento (default: 10 para rapidez)",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Batch size para entrenamiento",
    ),
    architecture: str = typer.Option(
        "resnet18",
        "--architecture",
        "-a",
        help="Arquitectura del clasificador (resnet18, efficientnet_b0, densenet121)",
    ),
    output_dir: str = typer.Option(
        "outputs/margin_optimization",
        "--output-dir",
        "-o",
        help="Directorio de salida para resultados",
    ),
    checkpoint: Optional[str] = typer.Option(
        None,
        "--checkpoint",
        "-c",
        help="Checkpoint del modelo de landmarks (para predecir landmarks on-the-fly)",
    ),
    canonical: Optional[str] = typer.Option(
        None,
        "--canonical",
        help="Archivo JSON con forma canónica (default: outputs/shape_analysis/canonical_shape.json)",
    ),
    triangles: Optional[str] = typer.Option(
        None,
        "--triangles",
        help="Archivo JSON con triangulación Delaunay (default: outputs/shape_analysis/delaunay_triangles.json)",
    ),
    quick: bool = typer.Option(
        False,
        "--quick",
        help="Modo rápido: usa menos epochs (3) y subconjunto de datos",
    ),
    keep_datasets: bool = typer.Option(
        False,
        "--keep-datasets",
        help="Mantener datasets warped temporales después de la optimización",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Semilla para reproducibilidad",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Dispositivo: auto, cuda, cpu, mps",
    ),
    patience: int = typer.Option(
        5,
        "--patience",
        help="Epochs sin mejora para early stopping",
    ),
    splits: str = typer.Option(
        "0.75,0.15,0.10",
        "--splits",
        help="Ratios para train,val,test (default: 0.75,0.15,0.10)",
    ),
):
    """
    Buscar automáticamente el margen óptimo para warping.

    Este comando itera sobre una lista de valores de margen, genera datasets
    warped para cada uno, entrena un clasificador rápido, y determina cuál
    margen produce la mejor accuracy de clasificación.

    El margen (margin_scale) controla cuánto del pulmón se captura:
    - margin > 1.0: Expande el ROI, captura más área
    - margin = 1.0: Sin cambio (baseline)
    - margin < 1.0: Contrae el ROI

    Ejemplo:
        python -m src_v2 optimize-margin \\
            --data-dir data/dataset/COVID-19_Radiography_Dataset \\
            --landmarks-csv data/landmarks.csv \\
            --margins 1.00,1.05,1.10,1.15,1.20,1.25,1.30 \\
            --epochs 10 \\
            --output-dir outputs/margin_optimization
    """
    import json
    import warnings
    import shutil
    import time
    from collections import Counter
    from datetime import datetime

    import cv2
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from PIL import Image
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from tqdm import tqdm

    from src_v2.data.utils import load_coordinates_csv
    from src_v2.models import ImageClassifier, get_classifier_transforms
    from src_v2.processing.warp import (
        clip_landmarks_to_image,
        compute_fill_rate,
        piecewise_affine_warp,
        scale_landmarks_from_centroid,
    )

    # Silenciar warning de joblib cuando el entorno no permite multiprocesamiento
    warnings.filterwarnings(
        "ignore",
        message=".*joblib will operate in serial mode.*",
        category=UserWarning,
        module="joblib._multiprocessing_helpers",
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("OPTIMIZE-MARGIN: Búsqueda de Margen Óptimo")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # Validaciones
    # -------------------------------------------------------------------------
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error("Directorio de datos no existe: %s", data_dir)
        raise typer.Exit(code=1)

    landmarks_path = Path(landmarks_csv)
    if not landmarks_path.exists():
        logger.error("Archivo de landmarks no existe: %s", landmarks_csv)
        raise typer.Exit(code=1)

    # Parsear márgenes
    try:
        margin_list = [float(m.strip()) for m in margins.split(",")]
    except ValueError:
        logger.error("Formato de márgenes inválido: %s", margins)
        raise typer.Exit(code=1)

    if len(margin_list) < 1:
        logger.error("Se requiere al menos un margen")
        raise typer.Exit(code=1)

    for m in margin_list:
        if m <= 0:
            logger.error("Los márgenes deben ser positivos, recibido: %.2f", m)
            raise typer.Exit(code=1)

    # Parsear splits
    try:
        split_values = [float(s.strip()) for s in splits.split(",")]
        if len(split_values) != 3:
            raise ValueError("Se requieren exactamente 3 valores")
        train_ratio, val_ratio, test_ratio = split_values
        if abs(sum(split_values) - 1.0) > 0.01:
            raise ValueError("Los splits deben sumar 1.0")
    except ValueError as e:
        logger.error("Formato de splits inválido: %s (%s)", splits, str(e))
        raise typer.Exit(code=1)

    # Validar arquitectura (debe coincidir con ImageClassifier.SUPPORTED_BACKBONES)
    valid_archs = ["resnet18", "resnet50", "efficientnet_b0", "densenet121", "alexnet", "vgg16", "mobilenet_v2"]
    if architecture not in valid_archs:
        logger.error(
            "Arquitectura no soportada: %s. Opciones: %s",
            architecture,
            valid_archs,
        )
        raise typer.Exit(code=1)

    # Configurar epochs para modo quick (usa constantes de constants.py)
    if quick:
        epochs = min(epochs, QUICK_MODE_EPOCHS_OPTIMIZE)
        logger.info("Modo rápido activado: epochs=%d", epochs)

    # Paths por defecto para canonical y triangles
    shape_analysis_dir = Path("outputs/shape_analysis")
    if canonical is None:
        # Intentar nombres alternativos en orden de preferencia
        canonical_candidates = [
            shape_analysis_dir / "canonical_shape.json",
            shape_analysis_dir / "canonical_shape_gpa.json",
        ]
        for candidate in canonical_candidates:
            if candidate.exists():
                canonical = str(candidate)
                break
        else:
            canonical = str(canonical_candidates[0])  # Default para mensaje de error
    if triangles is None:
        # Intentar nombres alternativos en orden de preferencia
        triangles_candidates = [
            shape_analysis_dir / "delaunay_triangles.json",
            shape_analysis_dir / "canonical_delaunay_triangles.json",
        ]
        for candidate in triangles_candidates:
            if candidate.exists():
                triangles = str(candidate)
                break
        else:
            triangles = str(triangles_candidates[0])  # Default para mensaje de error

    canonical_path = Path(canonical)
    triangles_path = Path(triangles)

    if not canonical_path.exists():
        logger.error("Archivo de forma canónica no existe: %s", canonical)
        raise typer.Exit(code=1)

    if not triangles_path.exists():
        logger.error("Archivo de triangulación no existe: %s", triangles)
        raise typer.Exit(code=1)

    # -------------------------------------------------------------------------
    # Configuración
    # -------------------------------------------------------------------------
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch_device = get_device(device)

    # Configurar semilla completa para reproducibilidad
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info("")
    logger.info("Configuración:")
    logger.info("  Data dir: %s", data_dir)
    logger.info("  Landmarks: %s", landmarks_csv)
    logger.info("  Márgenes a probar: %s", margin_list)
    logger.info("  Arquitectura: %s", architecture)
    logger.info("  Epochs: %d", epochs)
    logger.info("  Batch size: %d", batch_size)
    logger.info("  Patience: %d", patience)
    logger.info("  Splits: %.2f/%.2f/%.2f", train_ratio, val_ratio, test_ratio)
    logger.info("  Device: %s", torch_device)
    logger.info("  Output: %s", output_path)

    # -------------------------------------------------------------------------
    # Cargar datos
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("-" * 60)
    logger.info("Cargando datos...")
    logger.info("-" * 60)

    # Cargar forma canónica (soporta diferentes formatos de JSON)
    with open(canonical_path, "r") as f:
        canonical_data = json.load(f)
    # Intentar diferentes claves posibles
    if "canonical_shape" in canonical_data:
        canonical_shape = np.array(canonical_data["canonical_shape"])
    elif "canonical_shape_normalized" in canonical_data:
        canonical_shape = np.array(canonical_data["canonical_shape_normalized"])
    elif "mean_landmarks" in canonical_data:
        canonical_shape = np.array(canonical_data["mean_landmarks"])
    else:
        logger.error("No se encontró forma canónica en JSON. Claves disponibles: %s", list(canonical_data.keys()))
        raise typer.Exit(code=1)
    logger.info("Forma canónica cargada: %s", canonical_shape.shape)

    # Cargar triangulación
    with open(triangles_path, "r") as f:
        triangles_data = json.load(f)
    delaunay_triangles = np.array(triangles_data["triangles"])
    logger.info("Triángulos Delaunay cargados: %d", len(delaunay_triangles))

    # Cargar landmarks CSV (soporta formato con y sin headers)
    # Primero intentamos leer con headers, si no tiene columna category
    # usamos load_coordinates_csv para formato sin headers
    landmarks_df = pd.read_csv(landmarks_path)

    # Verificar si tiene columna de categoría
    has_category = "category" in landmarks_df.columns or "class" in landmarks_df.columns

    if not has_category:
        # Intentar con formato sin headers (coordenadas_maestro.csv)
        logger.debug("CSV sin columna category, intentando load_coordinates_csv")
        try:
            landmarks_df = load_coordinates_csv(str(landmarks_path))
            logger.info("Landmarks cargados con load_coordinates_csv: %d imágenes", len(landmarks_df))
        except (ValueError, Exception) as e:
            logger.error("No se pudo cargar CSV: %s", e)
            raise typer.Exit(code=1)
    else:
        logger.info("Landmarks cargados: %d imágenes", len(landmarks_df))

    # Detectar columnas de landmarks
    landmark_cols = [c for c in landmarks_df.columns if c.startswith("L") and "_" in c]
    n_landmarks = len(landmark_cols) // 2
    logger.info("Número de landmarks: %d", n_landmarks)

    # Validar número de landmarks (esperamos 15)
    if n_landmarks == 0:
        logger.error("No se encontraron columnas de landmarks (formato esperado: L1_x, L1_y, ...)")
        raise typer.Exit(code=1)
    if n_landmarks != 15:
        logger.warning("Número de landmarks (%d) diferente al esperado (15)", n_landmarks)

    # Detectar clases disponibles
    if "category" in landmarks_df.columns:
        category_col = "category"
    elif "class" in landmarks_df.columns:
        category_col = "class"
    else:
        logger.error("No se encontró columna de categoría en el CSV")
        raise typer.Exit(code=1)

    classes = sorted(landmarks_df[category_col].unique().tolist())
    logger.info("Clases detectadas: %s", classes)

    # -------------------------------------------------------------------------
    # Dataset class para warping on-the-fly
    # -------------------------------------------------------------------------
    class WarpedOnFlyDataset(Dataset):
        """Dataset que aplica warping on-the-fly con un margen específico."""

        def __init__(
            self,
            df: pd.DataFrame,
            data_root: Path,
            canonical_shape: np.ndarray,
            triangles: np.ndarray,
            margin_scale: float,
            transform=None,
            class_names: list = None,
        ):
            self.df = df.reset_index(drop=True)
            self.data_root = Path(data_root)
            self.canonical_shape = canonical_shape
            self.triangles = triangles
            self.margin_scale = margin_scale
            self.transform = transform
            self.class_names = class_names or classes
            self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        def __len__(self):
            return len(self.df)

        def _get_landmarks(self, row):
            """Extraer landmarks de una fila del DataFrame."""
            landmarks = []
            for i in range(1, n_landmarks + 1):
                x = row.get(f"L{i}_x", row.get(f"l{i}_x", 0))
                y = row.get(f"L{i}_y", row.get(f"l{i}_y", 0))
                landmarks.append([x, y])
            return np.array(landmarks, dtype=np.float32)

        def _find_image(self, image_name: str, category: str) -> Optional[Path]:
            """Buscar imagen en diferentes ubicaciones posibles."""
            # Asegurar extensión
            if not image_name.endswith((".png", ".jpg", ".jpeg")):
                image_name = f"{image_name}.png"

            # Posibles ubicaciones
            candidates = [
                self.data_root / category / "images" / image_name,
                self.data_root / category / image_name,
                self.data_root / image_name,
            ]

            for path in candidates:
                if path.exists():
                    return path
            return None

        def __getitem__(self, idx):
            row = self.df.iloc[idx]

            # Obtener metadata (manejar diferentes formatos de columna)
            if "image_name" in row.index:
                image_name = str(row["image_name"])
            elif "filename" in row.index:
                image_name = str(row["filename"])
            else:
                image_name = ""
            category = row[category_col]
            class_idx = self.class_to_idx.get(category, 0)

            # Buscar imagen
            img_path = self._find_image(image_name, category)
            if img_path is None:
                # Retornar imagen negra si no se encuentra (con warning)
                logger.warning("Imagen no encontrada: %s/%s - usando fallback negro", category, image_name)
                image = np.zeros((224, 224), dtype=np.uint8)
            else:
                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    logger.warning("No se pudo leer imagen: %s - usando fallback negro", img_path)
                    image = np.zeros((224, 224), dtype=np.uint8)

            # Resize si es necesario
            if image.shape[0] != 224 or image.shape[1] != 224:
                image = cv2.resize(image, (224, 224))

            # Obtener landmarks
            landmarks = self._get_landmarks(row)

            # Aplicar margin_scale
            scaled = scale_landmarks_from_centroid(landmarks, self.margin_scale)
            scaled = clip_landmarks_to_image(scaled, image_size=224)

            # Aplicar warping
            try:
                warped = piecewise_affine_warp(
                    image=image,
                    source_landmarks=scaled,
                    target_landmarks=self.canonical_shape,
                    triangles=self.triangles,
                    use_full_coverage=False,
                )
            except Exception as e:
                # Fallback si warping falla (landmarks inválidos, etc.)
                logger.debug("Warping fallido para imagen %s: %s", image_name, str(e))
                warped = image

            # Convertir a PIL para transforms
            img = Image.fromarray(warped)

            if self.transform:
                img = self.transform(img)

            return img, class_idx

    # -------------------------------------------------------------------------
    # Función de entrenamiento
    # -------------------------------------------------------------------------
    def train_and_evaluate_margin(
        margin: float,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict:
        """Entrena y evalúa un modelo para un margen específico."""
        logger.info("")
        logger.info("=" * 50)
        logger.info("MARGEN: %.2f", margin)
        logger.info("=" * 50)

        start_time = time.time()

        # Transforms
        train_transform = get_classifier_transforms(train=True, img_size=224)
        eval_transform = get_classifier_transforms(train=False, img_size=224)

        # Crear datasets
        train_dataset = WarpedOnFlyDataset(
            train_df,
            data_path,
            canonical_shape,
            delaunay_triangles,
            margin,
            train_transform,
            classes,
        )
        val_dataset = WarpedOnFlyDataset(
            val_df,
            data_path,
            canonical_shape,
            delaunay_triangles,
            margin,
            eval_transform,
            classes,
        )
        test_dataset = WarpedOnFlyDataset(
            test_df,
            data_path,
            canonical_shape,
            delaunay_triangles,
            margin,
            eval_transform,
            classes,
        )

        logger.info(
            "  Datasets: train=%d, val=%d, test=%d",
            len(train_dataset),
            len(val_dataset),
            len(test_dataset),
        )

        # DataLoaders (Session 33: num_workers dinamico segun OS)
        # pin_memory=True solo beneficia con CUDA
        use_pin_memory = torch_device.type == "cuda"
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=get_optimal_num_workers(),
            pin_memory=use_pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=get_optimal_num_workers(),
            pin_memory=use_pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=get_optimal_num_workers(),
            pin_memory=use_pin_memory,
        )

        # Crear modelo
        model = ImageClassifier(
            backbone=architecture,
            num_classes=len(classes),
            pretrained=True,
            dropout=0.3,
        )
        model = model.to(torch_device)

        # Class weights para balanceo
        train_labels = train_df[category_col].map(
            {c: i for i, c in enumerate(classes)}
        ).tolist()
        class_counts = Counter(train_labels)
        n_samples = len(train_labels)
        weights = torch.FloatTensor([
            n_samples / (len(classes) * class_counts.get(i, 1))
            for i in range(len(classes))
        ])

        criterion = nn.CrossEntropyLoss(weight=weights.to(torch_device))
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

        # Training loop
        best_val_acc = 0
        patience_counter = 0
        best_state = None
        epoch = 0  # Inicializar para evitar UnboundLocalError si epochs=0

        for epoch in range(epochs):
            # Train
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(torch_device), labels.to(torch_device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validate
            model.eval()
            val_preds, val_labels_list = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(torch_device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels_list.extend(labels.numpy())

            if val_labels_list:
                val_acc = accuracy_score(val_labels_list, val_preds)
            else:
                logger.warning("  Split val vacío; val_acc=0.0")
                val_acc = 0.0

            if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
                logger.info("  Epoch %d/%d: Val Acc=%.2f%%", epoch + 1, epochs, val_acc * 100)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("  Early stopping en epoch %d", epoch + 1)
                    break

        # Cargar mejor modelo
        if best_state is not None:
            model.load_state_dict(best_state)

        # Test evaluation
        model.eval()
        test_preds, test_labels_list = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(torch_device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_preds.extend(predicted.cpu().numpy())
                test_labels_list.extend(labels.numpy())

        if test_labels_list:
            test_acc = accuracy_score(test_labels_list, test_preds)
            test_f1 = f1_score(test_labels_list, test_preds, average="macro")
        else:
            logger.warning("  Split test vacío; métricas asignadas a 0.0")
            test_acc = 0.0
            test_f1 = 0.0

        elapsed = time.time() - start_time

        logger.info("")
        logger.info("  RESULTADO margin=%.2f:", margin)
        logger.info("    Val Accuracy: %.2f%%", best_val_acc * 100)
        logger.info("    Test Accuracy: %.2f%%", test_acc * 100)
        logger.info("    Test F1 (macro): %.2f%%", test_f1 * 100)
        logger.info("    Tiempo: %.1f segundos", elapsed)

        # Guardar checkpoint si se requiere
        margin_dir = output_path / "per_margin" / f"margin_{margin:.2f}"
        margin_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "margin": margin,
            "architecture": architecture,
            "class_names": classes,
            "val_accuracy": best_val_acc,
            "test_accuracy": test_acc,
        }
        torch.save(checkpoint_data, margin_dir / "checkpoint.pt")

        return {
            "margin": margin,
            "val_accuracy": best_val_acc * 100,
            "test_accuracy": test_acc * 100,
            "test_f1": test_f1 * 100,
            "epochs_trained": epoch + 1,
            "time_seconds": elapsed,
        }

    # -------------------------------------------------------------------------
    # Crear splits
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("-" * 60)
    logger.info("Creando splits estratificados...")
    logger.info("-" * 60)

    # Shuffle y split
    df_shuffled = landmarks_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    train_dfs = []
    val_dfs = []
    test_dfs = []

    for cat in classes:
        cat_df = df_shuffled[df_shuffled[category_col] == cat]
        n = len(cat_df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_dfs.append(cat_df.iloc[:n_train])
        val_dfs.append(cat_df.iloc[n_train : n_train + n_val])
        test_dfs.append(cat_df.iloc[n_train + n_val :])

    train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1, random_state=seed)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    logger.info("  Train: %d imágenes", len(train_df))
    logger.info("  Val: %d imágenes", len(val_df))
    logger.info("  Test: %d imágenes", len(test_df))

    # Si quick mode, usar subconjunto (constantes definidas en constants.py)
    if quick:
        max_train = min(QUICK_MODE_MAX_TRAIN, len(train_df))
        max_val = min(QUICK_MODE_MAX_VAL, len(val_df))
        max_test = min(QUICK_MODE_MAX_TEST, len(test_df))
        train_df = train_df.head(max_train)
        val_df = val_df.head(max_val)
        test_df = test_df.head(max_test)
        logger.info("  [Quick mode] Usando subconjunto: train=%d, val=%d, test=%d",
                    len(train_df), len(val_df), len(test_df))

    # -------------------------------------------------------------------------
    # Ejecutar optimización
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("EJECUTANDO OPTIMIZACIÓN DE MARGEN")
    logger.info("=" * 60)
    logger.info("Márgenes a probar: %s", margin_list)

    all_results = []
    total_start = time.time()

    for margin in tqdm(margin_list, desc="Márgenes", unit="margen"):
        result = train_and_evaluate_margin(margin, train_df, val_df, test_df)
        all_results.append(result)

    total_elapsed = time.time() - total_start

    # -------------------------------------------------------------------------
    # Análisis de resultados
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESUMEN DE RESULTADOS")
    logger.info("=" * 60)

    # Encontrar mejor margen
    best_result = max(all_results, key=lambda x: x["test_accuracy"])
    best_margin = best_result["margin"]
    best_accuracy = best_result["test_accuracy"]

    logger.info("")
    logger.info("%-10s %-12s %-12s %-12s", "Margen", "Val Acc", "Test Acc", "Test F1")
    logger.info("-" * 50)
    for r in all_results:
        marker = " ***" if r["margin"] == best_margin else ""
        logger.info(
            "%-10.2f %-12.2f %-12.2f %-12.2f%s",
            r["margin"],
            r["val_accuracy"],
            r["test_accuracy"],
            r["test_f1"],
            marker,
        )

    logger.info("")
    logger.info("*** MEJOR MARGEN: %.2f con %.2f%% accuracy ***", best_margin, best_accuracy)

    # -------------------------------------------------------------------------
    # Guardar resultados
    # -------------------------------------------------------------------------
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "data_dir": str(data_dir),
            "landmarks_csv": str(landmarks_csv),
            "margins_tested": margin_list,
            "architecture": architecture,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "splits": [train_ratio, val_ratio, test_ratio],
            "seed": seed,
            "quick_mode": quick,
        },
        "results": all_results,
        "best_margin": best_margin,
        "best_accuracy": best_accuracy,
        "total_time_seconds": total_elapsed,
        "total_time_minutes": total_elapsed / 60,
    }

    results_path = output_path / "margin_optimization_results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    logger.info("")
    logger.info("Resultados guardados: %s", results_path)

    # Guardar CSV resumen
    df_results = pd.DataFrame(all_results)
    csv_path = output_path / "summary.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info("Resumen CSV: %s", csv_path)

    # -------------------------------------------------------------------------
    # Generar gráfico
    # -------------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        margins_arr = [r["margin"] for r in all_results]
        val_accs = [r["val_accuracy"] for r in all_results]
        test_accs = [r["test_accuracy"] for r in all_results]

        ax.plot(margins_arr, val_accs, "b-o", label="Validation Accuracy", linewidth=2)
        ax.plot(margins_arr, test_accs, "r-s", label="Test Accuracy", linewidth=2)

        # Marcar mejor margen
        ax.axvline(x=best_margin, color="green", linestyle="--", alpha=0.7, label=f"Best margin={best_margin:.2f}")

        ax.set_xlabel("Margin Scale", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title(f"Margin Optimization - {architecture}\nBest: margin={best_margin:.2f}, acc={best_accuracy:.2f}%", fontsize=14)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Ajustar límites Y
        all_accs = val_accs + test_accs
        y_min = max(0, min(all_accs) - 5)
        y_max = min(100, max(all_accs) + 5)
        ax.set_ylim(y_min, y_max)

        plt.tight_layout()
        plot_path = output_path / "accuracy_vs_margin.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Gráfico guardado: %s", plot_path)

    except ImportError:
        logger.warning("matplotlib no disponible, omitiendo generación de gráfico")

    # -------------------------------------------------------------------------
    # Limpieza (si no se quieren mantener datasets)
    # -------------------------------------------------------------------------
    if not keep_datasets:
        per_margin_dir = output_path / "per_margin"
        if per_margin_dir.exists():
            # Mantener solo checkpoints, eliminar datasets si existieran
            for margin_subdir in per_margin_dir.iterdir():
                dataset_dir = margin_subdir / "dataset"
                if dataset_dir.exists():
                    shutil.rmtree(dataset_dir)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Optimización completada!")
    logger.info("=" * 60)
    logger.info("Mejor margen: %.2f", best_margin)
    logger.info("Mejor accuracy: %.2f%%", best_accuracy)
    logger.info("Tiempo total: %.1f minutos", total_elapsed / 60)
    logger.info("Resultados en: %s", output_path)


@app.command()
def extract_dataset_splits(
    output_dir: Path = typer.Option(
        Path("outputs/dataset_splits_for_gui"),
        "--output-dir",
        help="Directorio de salida para el dataset extraído"
    ),
    warped_dir: Path = typer.Option(
        Path("outputs/warped_lung_best/session_warping"),
        "--warped-dir",
        help="Directorio del dataset warped fuente"
    ),
    original_dir: Path = typer.Option(
        Path("data/dataset/COVID-19_Radiography_Dataset"),
        "--original-dir",
        help="Directorio del dataset original"
    ),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Verificar integridad de archivos copiados con checksums MD5"
    ),
    verbose_errors: bool = typer.Option(
        False,
        "--verbose-errors",
        help="Mostrar todos los errores detalladamente"
    )
):
    """
    Extrae los splits exactos train/val/test del modelo warped_lung_best.

    Copia tanto las imágenes originales como las warped a un directorio
    separado, listo para usar en la GUI o validación. Este dataset contiene
    los splits EXACTOS usados para entrenar el modelo que alcanzó 98.05% accuracy.

    Ejemplo de uso:

        python -m src_v2 extract-dataset-splits

        python -m src_v2 extract-dataset-splits \\
            --output-dir outputs/my_dataset \\
            --no-verify

    El comando crea la siguiente estructura:

        outputs/dataset_splits_for_gui/
        ├── original/           # Imágenes originales
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── warped/            # Imágenes normalizadas (las que vio el modelo)
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── metadata/          # CSVs y metadata
        └── README.md          # Documentación
    """
    import subprocess

    logger.info("=" * 60)
    logger.info("Extracción de Dataset Splits - warped_lung_best")
    logger.info("=" * 60)

    # Construir comando para el script standalone
    script_path = Path(__file__).parent.parent / "scripts" / "extract_dataset_splits.py"

    if not script_path.exists():
        logger.error("Script no encontrado: %s", script_path)
        raise typer.Exit(code=1)

    cmd = [
        sys.executable,
        str(script_path),
        "--output-dir", str(output_dir),
        "--warped-dir", str(warped_dir),
        "--original-dir", str(original_dir),
    ]

    if not verify:
        cmd.append("--no-verify")

    if verbose_errors:
        cmd.append("--verbose")

    logger.info("Ejecutando script de extracción...")
    logger.info("Comando: %s", " ".join(cmd))
    logger.info("")

    # Ejecutar script
    result = subprocess.run(cmd)

    if result.returncode != 0:
        logger.error("Error al ejecutar el script de extracción")
        raise typer.Exit(code=result.returncode)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Dataset extraído exitosamente!")
    logger.info("=" * 60)
    logger.info("Directorio de salida: %s", output_dir)
    logger.info("")
    logger.info("Siguiente paso:")
    logger.info("  - Revisar: %s/README.md", output_dir)
    logger.info("  - Metadata: %s/metadata/dataset_info.json", output_dir)


@app.command("generate-landmark-visualization-dataset")
def generate_landmark_visualization_dataset(
    input_dir: Optional[str] = typer.Argument(
        None,
        help="Directorio del dataset original (estructura COVID/Normal/Viral_Pneumonia/images/)"
    ),
    output_dir: Optional[str] = typer.Argument(
        None,
        help="Directorio de salida para visualizaciones"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path a JSON con parametros"
    ),
    predictions: Optional[str] = typer.Option(
        None,
        "--predictions",
        help="Archivo NPZ con landmarks predichos (requerido)"
    ),
    splits: str = typer.Option(
        "0.75,0.125,0.125",
        "--splits",
        help="Ratios train,val,test separados por coma"
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Semilla para reproducibilidad de splits (debe ser 42 para alinearse con warped dataset)"
    ),
    classes: str = typer.Option(
        "COVID,Normal,Viral Pneumonia",
        "--classes",
        help="Clases a procesar separadas por coma"
    ),
    cross_size: int = typer.Option(
        5,
        "--cross-size",
        help="Tamaño medio de cruces en píxeles"
    ),
    cross_thickness: int = typer.Option(
        2,
        "--cross-thickness",
        help="Grosor de líneas de cruces"
    ),
    cross_color: str = typer.Option(
        "red",
        "--cross-color",
        help="Color de las cruces (red, green, blue, yellow, etc.)"
    ),
):
    """
    Generar dataset de visualización con landmarks dibujados sobre imágenes originales.

    Este comando:
    1. Carga predicciones de landmarks desde archivo NPZ
    2. Usa los MISMOS splits que generate-dataset (con seed=42)
    3. Dibuja cruces rojas científicas en cada landmark
    4. Guarda imágenes originales (299x299, sin preprocesamiento) con landmarks marcados

    IMPORTANTE: Usar seed=42 para garantizar que los splits coincidan exactamente
    con el dataset warped usado para clasificación.

    Estructura de salida:
        output_dir/
        ├── train/
        │   ├── COVID/
        │   │   └── COVID-1_landmarks_viz.png
        │   ├── Normal/
        │   └── Viral_Pneumonia/
        ├── val/
        ├── test/
        └── dataset_summary.json

    Ejemplo:
        python -m src_v2 generate-landmark-visualization-dataset \\
            --config configs/landmark_viz_best.json

        python -m src_v2 generate-landmark-visualization-dataset \\
            data/dataset/COVID-19_Radiography_Dataset \\
            outputs/landmark_visualizations/session_warping \\
            --predictions outputs/landmark_predictions/session_warping/predictions.npz
    """
    import json
    import time
    from collections import defaultdict

    import cv2
    import numpy as np
    from tqdm import tqdm

    from src_v2.visualization.utils import draw_scientific_crosses_on_image

    logger.info("=" * 60)
    logger.info("Generate Landmark Visualization Dataset")
    logger.info("=" * 60)

    # Load config if provided
    argv = sys.argv[1:]

    def option_present(*names: str) -> bool:
        for name in names:
            if name in argv:
                return True
            prefix = f"{name}="
            if any(arg.startswith(prefix) for arg in argv):
                return True
        return False

    config_data = {}
    if config:
        config_path = Path(config)
        if not config_path.exists():
            logger.error("Config no existe: %s", config)
            raise typer.Exit(code=1)
        with open(config_path, "r") as f:
            config_data = json.load(f)

        def override_param(
            param_name: str,
            current_value,
            config_key: Optional[str] = None,
            option_names: Optional[tuple[str, ...]] = None,
        ):
            key = config_key or param_name
            if key not in config_data:
                return current_value
            if option_names and option_present(*option_names):
                return current_value
            if option_names is None and current_value is not None:
                return current_value
            return config_data[key]

        input_dir = override_param("input_dir", input_dir, "input_dir")
        output_dir = override_param("output_dir", output_dir, "output_dir")
        predictions = override_param("predictions", predictions, "predictions", ("--predictions",))
        splits = override_param("splits", splits, "splits", ("--splits",))
        seed = override_param("seed", seed, "seed", ("--seed", "-s"))
        classes = override_param("classes", classes, "classes", ("--classes",))
        cross_size = override_param("cross_size", cross_size, "cross_size", ("--cross-size",))
        cross_thickness = override_param("cross_thickness", cross_thickness, "cross_thickness", ("--cross-thickness",))
        cross_color = override_param("cross_color", cross_color, "cross_color", ("--cross-color",))

    if isinstance(splits, (list, tuple)):
        splits = ",".join(str(x) for x in splits)
    if isinstance(classes, (list, tuple)):
        classes = ",".join(str(x) for x in classes)

    # Validate inputs
    if input_dir is None or output_dir is None:
        logger.error("input_dir y output_dir son requeridos (usa argumentos o --config)")
        raise typer.Exit(code=1)

    if predictions is None:
        logger.error("--predictions es requerido (archivo NPZ con landmarks)")
        raise typer.Exit(code=1)

    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error("Directorio de entrada no existe: %s", input_dir)
        raise typer.Exit(code=1)

    predictions_path = Path(predictions)
    if not predictions_path.exists():
        logger.error("Archivo de predicciones no existe: %s", predictions)
        raise typer.Exit(code=1)

    # Parse splits
    try:
        split_ratios = [float(x) for x in splits.split(",")]
        if len(split_ratios) != 3:
            raise ValueError("Se requieren exactamente 3 valores")
        if abs(sum(split_ratios) - 1.0) > 0.01:
            raise ValueError(f"Los ratios deben sumar 1.0, suman {sum(split_ratios)}")
        train_ratio, val_ratio, test_ratio = split_ratios
    except Exception as e:
        logger.error("Error parseando splits '%s': %s", splits, e)
        raise typer.Exit(code=1)

    # Parse classes
    class_list = [c.strip() for c in classes.split(",")]
    class_mapping = {c: c.replace(" ", "_") for c in class_list}
    logger.info("Clases a procesar: %s", class_list)

    # Load predictions from NPZ
    logger.info("Cargando predicciones: %s", predictions_path)
    data = np.load(predictions_path, allow_pickle=True)

    if "landmarks" not in data:
        logger.error("NPZ no contiene clave 'landmarks'")
        raise typer.Exit(code=1)

    landmarks_224 = data["landmarks"]  # Shape: (N, 15, 2)
    image_names = data["image_names"] if "image_names" in data else None
    categories = data["categories"] if "categories" in data else None

    if image_names is None:
        logger.error("NPZ debe incluir 'image_names'")
        raise typer.Exit(code=1)

    # Build lookup dictionary: (image_name, category) -> landmarks
    predictions_dict = {}
    for i in range(len(landmarks_224)):
        name = str(image_names[i])
        cat = str(categories[i]).replace(" ", "_") if categories is not None else None
        predictions_dict[(name, cat)] = landmarks_224[i]

    logger.info("Predicciones cargadas: %d", len(predictions_dict))

    # Scale factor for landmarks: 224 -> 299
    SCALE_FACTOR = ORIGINAL_IMAGE_SIZE / DEFAULT_IMAGE_SIZE
    logger.info("Factor de escala para landmarks: %.4f (224 -> %d)", SCALE_FACTOR, ORIGINAL_IMAGE_SIZE)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect images (SAME logic as generate-dataset)
    logger.info("Recolectando imágenes del dataset...")
    all_images = []

    for class_name in class_list:
        # Try class_name/images/ structure first
        class_dir = input_path / class_name / "images"
        if not class_dir.exists():
            # Try without images subdirectory
            class_dir = input_path / class_name
        if not class_dir.exists():
            logger.warning("Directorio de clase no existe: %s", class_dir)
            continue

        images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        mapped_class = class_mapping[class_name]
        all_images.extend([(img, mapped_class) for img in images])
        logger.info("  %s: %d imágenes", class_name, len(images))

    if not all_images:
        logger.error("No se encontraron imágenes")
        raise typer.Exit(code=1)

    logger.info("Total: %d imágenes", len(all_images))

    # Create splits (EXACT SAME logic as generate-dataset)
    logger.info("Creando splits (train=%.0f%%, val=%.0f%%, test=%.0f%%)...",
                train_ratio * 100, val_ratio * 100, test_ratio * 100)
    logger.info("Usando seed=%d (CRÍTICO: debe ser 42 para alinearse con warped dataset)", seed)

    # CRITICAL: Set seed for reproducibility (SAME as generate-dataset)
    random.seed(seed)
    np.random.seed(seed)

    # Group by class
    by_class = defaultdict(list)
    for path, class_name in all_images:
        by_class[class_name].append(path)

    split_data = {'train': [], 'val': [], 'test': []}

    # Split each class (EXACT SAME logic)
    for class_name, paths in by_class.items():
        np.random.shuffle(paths)
        n = len(paths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        split_data['train'].extend([(p, class_name) for p in paths[:n_train]])
        split_data['val'].extend([(p, class_name) for p in paths[n_train:n_train + n_val]])
        split_data['test'].extend([(p, class_name) for p in paths[n_train + n_val:]])

    # Shuffle each split
    for split_name in split_data:
        np.random.shuffle(split_data[split_name])
        by_class_count = defaultdict(int)
        for _, c in split_data[split_name]:
            by_class_count[c] += 1
        logger.info("  %s: %d imágenes %s", split_name, len(split_data[split_name]), dict(by_class_count))

    # Process images
    start_time = time.time()
    stats = {
        'processed': 0,
        'failed': 0,
        'missing_predictions': 0,
        'wrong_size': 0,
    }

    logger.info("")
    logger.info("=" * 60)
    logger.info("Procesando imágenes...")
    logger.info("=" * 60)

    for split_name, split_images in split_data.items():
        logger.info("")
        logger.info("=== Procesando %s ===", split_name.upper())

        for image_path, class_name in tqdm(split_images, desc=f"  {split_name}"):
            # 1. Load original image (NO preprocessing)
            image_orig = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image_orig is None:
                stats['failed'] += 1
                continue

            # Verify image size (should be 299x299)
            if image_orig.shape[:2] != (ORIGINAL_IMAGE_SIZE, ORIGINAL_IMAGE_SIZE):
                logger.warning("Imagen %s tiene tamaño %s (esperado %dx%d), aplicando resize",
                              image_path.name, image_orig.shape[:2], ORIGINAL_IMAGE_SIZE, ORIGINAL_IMAGE_SIZE)
                image_orig = cv2.resize(image_orig, (ORIGINAL_IMAGE_SIZE, ORIGINAL_IMAGE_SIZE))
                stats['wrong_size'] += 1

            # 2. Look up predicted landmarks
            image_name = image_path.stem
            landmarks_key = (image_name, class_name)
            landmarks_224_img = predictions_dict.get(landmarks_key)

            if landmarks_224_img is None:
                logger.warning("Predicción no encontrada para: %s (clase: %s)", image_name, class_name)
                stats['missing_predictions'] += 1
                continue

            # 3. Scale landmarks: 224 -> 299
            landmarks_299 = landmarks_224_img * SCALE_FACTOR

            # 4. Clip to valid range [0, 298]
            landmarks_299 = np.clip(landmarks_299, 0, ORIGINAL_IMAGE_SIZE - 1)

            # 5. Draw scientific crosses
            image_with_landmarks = draw_scientific_crosses_on_image(
                image_orig,
                landmarks_299,
                cross_size=cross_size,
                thickness=cross_thickness,
                color=cross_color,
                return_rgb=True
            )

            # 6. Save visualization
            output_filename = f"{image_path.stem}_landmarks_viz.png"
            output_file_path = output_path / split_name / class_name / output_filename
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(output_file_path), image_with_landmarks)
            stats['processed'] += 1

    elapsed = time.time() - start_time

    # Save dataset summary
    summary = {
        "source_dataset": str(input_path),
        "predictions": str(predictions_path),
        "classes": class_list,
        "splits": {
            "train": {
                "count": len(split_data['train']),
                "by_class": {c: sum(1 for _, cls in split_data['train'] if cls == c) for c in class_mapping.values()}
            },
            "val": {
                "count": len(split_data['val']),
                "by_class": {c: sum(1 for _, cls in split_data['val'] if cls == c) for c in class_mapping.values()}
            },
            "test": {
                "count": len(split_data['test']),
                "by_class": {c: sum(1 for _, cls in split_data['test'] if cls == c) for c in class_mapping.values()}
            },
        },
        "seed": seed,
        "image_size": ORIGINAL_IMAGE_SIZE,
        "landmarks_scale_factor": float(SCALE_FACTOR),
        "cross_size": cross_size,
        "cross_thickness": cross_thickness,
        "cross_color": cross_color,
        "processing_stats": stats,
        "processing_time_seconds": elapsed,
    }

    # Add metadata from predictions if available
    if "metadata_json" in data:
        try:
            pred_meta = json.loads(str(data["metadata_json"].item()))
            summary["landmark_ensemble"] = pred_meta
        except (json.JSONDecodeError, AttributeError):
            pass

    summary_path = output_path / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Dataset de visualización completado!")
    logger.info("=" * 60)
    logger.info("Imágenes procesadas: %d", stats['processed'])
    logger.info("Fallos: %d", stats['failed'])
    logger.info("Predicciones faltantes: %d", stats['missing_predictions'])
    logger.info("Imágenes con tamaño incorrecto: %d", stats['wrong_size'])
    logger.info("Tiempo total: %.1f minutos", elapsed / 60)
    logger.info("")
    logger.info("Directorio de salida: %s", output_path)
    logger.info("Resumen guardado: %s", summary_path)
    logger.info("")
    logger.info("Siguiente paso:")
    logger.info("  - Verificar alineación: python scripts/verify_landmark_viz_dataset.py")


@app.command("generate-delaunay-mesh-dataset")
def generate_delaunay_mesh_dataset(
    input_dir: Optional[str] = typer.Argument(
        None,
        help="Directorio del dataset original (estructura COVID/Normal/Viral Pneumonia/images/)"
    ),
    output_dir: Optional[str] = typer.Argument(
        None,
        help="Directorio de salida para visualizaciones con malla"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path a JSON con parametros"
    ),
    predictions: Optional[str] = typer.Option(
        None,
        "--predictions",
        help="Archivo NPZ con landmarks predichos (requerido)"
    ),
    triangles: str = typer.Option(
        "outputs/shape_analysis/canonical_delaunay_triangles.json",
        "--triangles",
        help="Archivo JSON con triangulacion Delaunay canonica"
    ),
    warped_dataset_dir: Optional[str] = typer.Option(
        None,
        "--warped-dataset",
        help="Dataset warped para usar sus splits (train/val/test/images.csv)"
    ),
    splits: str = typer.Option(
        "0.75,0.125,0.125",
        "--splits",
        help="Ratios train,val,test separados por coma (si no se usa --warped-dataset)"
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Semilla para reproducibilidad (si no se usa --warped-dataset)"
    ),
    classes: str = typer.Option(
        "COVID,Normal,Viral Pneumonia",
        "--classes",
        help="Clases a procesar separadas por coma"
    ),
    image_size: int = typer.Option(
        ORIGINAL_IMAGE_SIZE,
        "--image-size",
        help="Tamaño de imagen para visualización (default: 299)"
    ),
    preserve_size: bool = typer.Option(
        True,
        "--preserve-size/--resize",
        help="Preservar tamaño original o forzar resize al image-size"
    ),
    mesh_color: str = typer.Option(
        "cyan",
        "--mesh-color",
        help="Color de la malla (cyan, red, green, blue, yellow, white o #RRGGBB)"
    ),
    mesh_thickness: int = typer.Option(
        1,
        "--mesh-thickness",
        help="Grosor de líneas de la malla"
    ),
):
    """
    Generar dataset con malla de Delaunay sobre imágenes originales.

    Este comando:
    1. Carga predicciones NPZ del ensemble (3.61 px)
    2. Carga triangulación canónica (Procrustes/GPA)
    3. Usa splits del dataset warped o genera splits reproducibles
    4. Dibuja la malla de Delaunay sobre cada imagen original
    5. Guarda imágenes en train/val/test con la misma estructura del clasificador
    """
    import csv
    import json
    import time
    from collections import defaultdict

    import cv2
    import numpy as np
    from tqdm import tqdm

    logger.info("=" * 60)
    logger.info("Generate Delaunay Mesh Dataset")
    logger.info("=" * 60)

    argv = sys.argv[1:]

    def option_present(*names: str) -> bool:
        for name in names:
            if name in argv:
                return True
            prefix = f"{name}="
            if any(arg.startswith(prefix) for arg in argv):
                return True
        return False

    config_data = {}
    if config:
        config_path = Path(config)
        if not config_path.exists():
            logger.error("Config no existe: %s", config)
            raise typer.Exit(code=1)
        with open(config_path, "r") as f:
            config_data = json.load(f)

        def override_param(
            param_name: str,
            current_value,
            config_key: Optional[str] = None,
            option_names: Optional[tuple[str, ...]] = None,
        ):
            key = config_key or param_name
            if key not in config_data:
                return current_value
            if option_names and option_present(*option_names):
                return current_value
            if option_names is None and current_value is not None:
                return current_value
            return config_data[key]

        input_dir = override_param("input_dir", input_dir, "input_dir")
        output_dir = override_param("output_dir", output_dir, "output_dir")
        predictions = override_param("predictions", predictions, "predictions", ("--predictions",))
        triangles = override_param("triangles", triangles, "triangles", ("--triangles",))
        warped_dataset_dir = override_param(
            "warped_dataset_dir",
            warped_dataset_dir,
            "warped_dataset_dir",
            ("--warped-dataset",),
        )
        splits = override_param("splits", splits, "splits", ("--splits",))
        seed = override_param("seed", seed, "seed", ("--seed", "-s"))
        classes = override_param("classes", classes, "classes", ("--classes",))
        image_size = override_param("image_size", image_size, "image_size", ("--image-size",))
        preserve_size = override_param(
            "preserve_size",
            preserve_size,
            "preserve_size",
            ("--preserve-size", "--resize"),
        )
        mesh_color = override_param("mesh_color", mesh_color, "mesh_color", ("--mesh-color",))
        mesh_thickness = override_param(
            "mesh_thickness",
            mesh_thickness,
            "mesh_thickness",
            ("--mesh-thickness",),
        )

    if isinstance(splits, (list, tuple)):
        splits = ",".join(str(x) for x in splits)
    if isinstance(classes, (list, tuple)):
        classes = ",".join(str(x) for x in classes)

    if input_dir is None or output_dir is None:
        logger.error("input_dir y output_dir son requeridos (usa argumentos o --config)")
        raise typer.Exit(code=1)

    if predictions is None:
        logger.error("--predictions es requerido (archivo NPZ con landmarks)")
        raise typer.Exit(code=1)

    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error("Directorio de entrada no existe: %s", input_dir)
        raise typer.Exit(code=1)

    predictions_path = Path(predictions)
    if not predictions_path.exists():
        logger.error("Archivo de predicciones no existe: %s", predictions_path)
        raise typer.Exit(code=1)

    triangles_path = Path(triangles)
    if not triangles_path.exists():
        logger.error("Archivo de triangulacion no existe: %s", triangles_path)
        raise typer.Exit(code=1)

    def parse_color(color_value: str) -> tuple[int, int, int]:
        color_map = {
            "cyan": (255, 255, 0),
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
            "white": (255, 255, 255),
        }
        value = color_value.strip().lower()
        if value in color_map:
            return color_map[value]
        if value.startswith("#") and len(value) == 7:
            r = int(value[1:3], 16)
            g = int(value[3:5], 16)
            b = int(value[5:7], 16)
            return (b, g, r)
        raise ValueError(f"Color no soportado: {color_value}")

    try:
        mesh_color_bgr = parse_color(mesh_color)
    except ValueError as exc:
        logger.error("%s", exc)
        raise typer.Exit(code=1)

    class_list = [c.strip() for c in classes.split(",")]
    class_mapping = {c: c.replace(" ", "_") for c in class_list}
    logger.info("Clases a procesar: %s", class_list)

    def resolve_class_dir(class_name: str) -> Optional[Path]:
        candidates = [
            input_path / class_name / "images",
            input_path / class_name,
            input_path / class_name.replace("_", " ") / "images",
            input_path / class_name.replace("_", " "),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    class_dirs = {mapped: resolve_class_dir(mapped) for mapped in class_mapping.values()}
    for class_name, class_dir in class_dirs.items():
        if class_dir is None:
            logger.warning("Directorio de clase no existe para: %s", class_name)

    logger.info("Cargando predicciones: %s", predictions_path)
    data = np.load(predictions_path, allow_pickle=True)
    if "landmarks" not in data:
        logger.error("NPZ no contiene clave 'landmarks'")
        raise typer.Exit(code=1)

    landmarks_224 = data["landmarks"]
    image_names = data["image_names"] if "image_names" in data else None
    categories = data["categories"] if "categories" in data else None
    if image_names is None or categories is None:
        logger.error("NPZ debe incluir 'image_names' y 'categories'")
        raise typer.Exit(code=1)

    predictions_dict = {}
    for i in range(len(landmarks_224)):
        name = str(image_names[i])
        cat = str(categories[i]).replace(" ", "_")
        predictions_dict[(name, cat)] = landmarks_224[i]

    logger.info("Predicciones cargadas: %d", len(predictions_dict))

    with triangles_path.open("r", encoding="utf-8") as f:
        triangles_data = json.load(f)
    triangles_list = np.array(triangles_data["triangles"], dtype=np.int32)

    scale_factor = image_size / DEFAULT_IMAGE_SIZE
    if preserve_size:
        logger.info("Escala landmarks: se calcula por imagen (224 -> tamaño original)")
    else:
        logger.info("Factor de escala para landmarks: %.4f (224 -> %d)", scale_factor, image_size)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    split_data = {}
    if warped_dataset_dir:
        warped_path = Path(warped_dataset_dir)
        if not warped_path.exists():
            logger.error("Warped dataset no existe: %s", warped_dataset_dir)
            raise typer.Exit(code=1)
        for split_name in ("train", "val", "test"):
            csv_path = warped_path / split_name / "images.csv"
            if not csv_path.exists():
                logger.error("No se encontro: %s", csv_path)
                raise typer.Exit(code=1)
            entries = []
            with csv_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    entries.append((row["image_name"], row["category"]))
            split_data[split_name] = entries
        logger.info("Splits cargados desde: %s", warped_path)
    else:
        try:
            split_ratios = [float(x) for x in splits.split(",")]
            if len(split_ratios) != 3:
                raise ValueError("Se requieren exactamente 3 valores")
            if abs(sum(split_ratios) - 1.0) > 0.01:
                raise ValueError(f"Los ratios deben sumar 1.0, suman {sum(split_ratios)}")
            train_ratio, val_ratio, test_ratio = split_ratios
        except Exception as exc:
            logger.error("Error parseando splits '%s': %s", splits, exc)
            raise typer.Exit(code=1)

        logger.info("Recolectando imágenes del dataset...")
        all_images = []
        for class_name in class_list:
            class_dir = resolve_class_dir(class_name)
            if class_dir is None:
                logger.warning("Directorio de clase no existe: %s", class_name)
                continue
            images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
            mapped_class = class_mapping[class_name]
            for img in images:
                all_images.append((img.stem, mapped_class))
            logger.info("  %s: %d imágenes", class_name, len(images))

        if not all_images:
            logger.error("No se encontraron imágenes")
            raise typer.Exit(code=1)

        random.seed(seed)
        np.random.seed(seed)

        by_class = defaultdict(list)
        for image_name, class_name in all_images:
            by_class[class_name].append(image_name)

        split_data = {"train": [], "val": [], "test": []}
        for class_name, names in by_class.items():
            np.random.shuffle(names)
            n = len(names)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            split_data["train"].extend([(name, class_name) for name in names[:n_train]])
            val_names = names[n_train:n_train + n_val]
            split_data["val"].extend([(name, class_name) for name in val_names])
            split_data["test"].extend([(name, class_name) for name in names[n_train + n_val:]])

        for split_name in split_data:
            np.random.shuffle(split_data[split_name])
            counts = defaultdict(int)
            for _, class_name in split_data[split_name]:
                counts[class_name] += 1
            logger.info(
                "  %s: %d imágenes %s",
                split_name,
                len(split_data[split_name]),
                dict(counts),
            )

    def to_bgr(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image.copy()

    def draw_mesh(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        image_bgr = to_bgr(image)
        points = landmarks.astype(np.int32)
        for triangle in triangles_list:
            tri_points = points[triangle].reshape((-1, 1, 2))
            cv2.polylines(
                image_bgr,
                [tri_points],
                isClosed=True,
                color=mesh_color_bgr,
                thickness=mesh_thickness,
                lineType=cv2.LINE_AA,
            )
        return image_bgr

    start_time = time.time()
    stats = {
        "processed": 0,
        "failed": 0,
        "missing_predictions": 0,
        "wrong_size": 0,
    }

    for split_name, split_images in split_data.items():
        logger.info("")
        logger.info("=== Procesando %s ===", split_name.upper())

        for image_name, class_name in tqdm(split_images, desc=f"  {split_name}"):
            class_dir = class_dirs.get(class_name)
            if class_dir is None:
                stats["failed"] += 1
                continue

            image_path = class_dir / f"{image_name}.png"
            if not image_path.exists():
                image_path = class_dir / f"{image_name}.jpg"
            if not image_path.exists():
                stats["failed"] += 1
                continue

            image_orig = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image_orig is None:
                stats["failed"] += 1
                continue

            if image_orig.shape[:2] != (image_size, image_size):
                stats["wrong_size"] += 1
                if not preserve_size:
                    interp = cv2.INTER_AREA
                    if image_orig.shape[0] < image_size or image_orig.shape[1] < image_size:
                        interp = cv2.INTER_CUBIC
                    image_orig = cv2.resize(image_orig, (image_size, image_size), interpolation=interp)

            landmarks_224_img = predictions_dict.get((image_name, class_name))
            if landmarks_224_img is None:
                stats["missing_predictions"] += 1
                continue

            if preserve_size:
                target_h, target_w = image_orig.shape[:2]
                landmarks_scaled = landmarks_224_img.copy()
                landmarks_scaled[:, 0] *= target_w / DEFAULT_IMAGE_SIZE
                landmarks_scaled[:, 1] *= target_h / DEFAULT_IMAGE_SIZE
                landmarks_scaled[:, 0] = np.clip(landmarks_scaled[:, 0], 0, target_w - 1)
                landmarks_scaled[:, 1] = np.clip(landmarks_scaled[:, 1], 0, target_h - 1)
            else:
                landmarks_scaled = landmarks_224_img * scale_factor
                landmarks_scaled = np.clip(landmarks_scaled, 0, image_size - 1)

            image_with_mesh = draw_mesh(image_orig, landmarks_scaled)

            output_filename = f"{image_name}_mesh.png"
            output_file_path = output_path / split_name / class_name / output_filename
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_file_path), image_with_mesh)
            stats["processed"] += 1

    elapsed = time.time() - start_time

    summary = {
        "source_dataset": str(input_path),
        "predictions": str(predictions_path),
        "triangles": str(triangles_path),
        "classes": class_list,
        "splits": {
            split: {
                "count": len(split_data[split]),
                "by_class": {c: sum(1 for _, cls in split_data[split] if cls == c)
                             for c in class_mapping.values()}
            }
            for split in split_data
        },
        "seed": seed,
        "image_size": image_size,
        "preserve_size": preserve_size,
        "landmarks_scale_factor": float(scale_factor),
        "mesh_color": mesh_color,
        "mesh_thickness": mesh_thickness,
        "processing_stats": stats,
        "processing_time_seconds": elapsed,
        "warped_dataset_dir": str(warped_dataset_dir) if warped_dataset_dir else None,
    }

    if "metadata_json" in data:
        try:
            pred_meta = json.loads(str(data["metadata_json"].item()))
            summary["landmark_ensemble"] = pred_meta
        except (json.JSONDecodeError, AttributeError):
            pass

    summary_path = output_path / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Dataset con malla de Delaunay completado!")
    logger.info("=" * 60)
    logger.info("Imágenes procesadas: %d", stats["processed"])
    logger.info("Fallos: %d", stats["failed"])
    logger.info("Predicciones faltantes: %d", stats["missing_predictions"])
    logger.info("Imágenes con tamaño incorrecto: %d", stats["wrong_size"])
    logger.info("Tiempo total: %.1f minutos", elapsed / 60)
    logger.info("")
    logger.info("Directorio de salida: %s", output_path)
    logger.info("Resumen guardado: %s", summary_path)


@app.command("generate-landmark-comparison-dataset")
def generate_landmark_comparison_dataset(
    input_dir: str = typer.Argument(
        ...,
        help="Directorio del dataset original"
    ),
    output_dir: str = typer.Argument(
        ...,
        help="Directorio de salida para comparaciones"
    ),
    ground_truth_csv: str = typer.Option(
        "data/coordenadas/coordenadas_maestro.csv",
        "--ground-truth-csv",
        help="CSV con landmarks ground truth (299×299)"
    ),
    predictions: str = typer.Option(
        "outputs/landmark_predictions/session_warping/predictions.npz",
        "--predictions",
        help="NPZ con predicciones del ensemble (224×224)"
    ),
    seed: int = typer.Option(
        42,
        "--seed", "-s",
        help="Semilla para splits reproducibles"
    ),
    pred_color: str = typer.Option(
        "red",
        "--pred-color",
        help="Color para predicciones (red, blue, green, etc.)"
    ),
    gt_color: str = typer.Option(
        "green",
        "--gt-color",
        help="Color para ground truth"
    ),
    cross_size: int = typer.Option(
        5,
        "--cross-size",
        help="Tamaño de cruces en píxeles"
    ),
    cross_thickness: int = typer.Option(
        2,
        "--cross-thickness",
        help="Grosor de líneas"
    ),
    show_error_lines: bool = typer.Option(
        False,
        "--show-error-lines/--no-error-lines",
        help="Dibujar líneas amarillas entre pares correspondientes"
    ),
    max_per_split: Optional[int] = typer.Option(
        None,
        "--max-per-split",
        help="Máximo de imágenes por split (para testing)"
    ),
):
    """
    Generar dataset de visualizaciones comparativas (predicciones vs ground truth).

    Este comando:
    1. Carga landmarks ground truth desde CSV (957 imágenes)
    2. Carga predicciones desde NPZ
    3. Hace matching entre ambos
    4. Crea splits estratificados (75/15/10)
    5. Genera visualizaciones duales:
       - Cruces rojas: predicciones
       - Cruces verdes: ground truth
    6. Calcula errores por imagen y landmark
    7. Guarda metadata con estadísticas detalladas

    Ejemplo:
        python -m src_v2 generate-landmark-comparison-dataset \\
            data/dataset/COVID-19_Radiography_Dataset \\
            outputs/landmark_comparisons/best_ensemble

        # Testing con subset pequeño
        python -m src_v2 generate-landmark-comparison-dataset \\
            data/dataset/COVID-19_Radiography_Dataset \\
            outputs/landmark_comparisons/test \\
            --max-per-split 5 \\
            --show-error-lines
    """
    from pathlib import Path
    from src_v2.visualization.comparison_viz import generate_comparison_dataset

    logger.info("=" * 60)
    logger.info("Generate Landmark Comparison Dataset")
    logger.info("=" * 60)

    # Validar inputs
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory no existe: {input_dir}")
        raise typer.Exit(code=1)

    gt_csv_path = Path(ground_truth_csv)
    if not gt_csv_path.exists():
        logger.error(f"Ground truth CSV no existe: {ground_truth_csv}")
        raise typer.Exit(code=1)

    pred_npz_path = Path(predictions)
    if not pred_npz_path.exists():
        logger.error(f"Predictions NPZ no existe: {predictions}")
        raise typer.Exit(code=1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Ejecutar pipeline
    try:
        metadata = generate_comparison_dataset(
            input_dir=input_path,
            output_dir=output_path,
            ground_truth_csv=gt_csv_path,
            predictions_npz=pred_npz_path,
            seed=seed,
            pred_color=pred_color,
            gt_color=gt_color,
            cross_size=cross_size,
            cross_thickness=cross_thickness,
            show_error_lines=show_error_lines,
            max_per_split=max_per_split
        )

        logger.info("")
        logger.info("Archivos generados:")
        logger.info(f"  - {output_path / 'metadata.json'}")
        logger.info(f"  - {output_path / 'error_statistics.json'}")
        logger.info(f"  - {output_path / 'image_errors.csv'}")

    except Exception as e:
        logger.error(f"Error durante generación: {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command("generate-landmark-comparison-dataset-aligned")
def generate_landmark_comparison_dataset_aligned(
    warped_dataset_dir: str = typer.Argument(
        ...,
        help="Directorio del warped dataset (ej: outputs/warped_lung_best/session_warping)"
    ),
    output_dir: str = typer.Argument(
        ...,
        help="Directorio de salida para comparaciones"
    ),
    original_images_dir: str = typer.Option(
        "data/dataset/COVID-19_Radiography_Dataset",
        "--original-images",
        help="Directorio de imágenes originales 299×299"
    ),
    ground_truth_csv: str = typer.Option(
        "data/coordenadas/coordenadas_maestro.csv",
        "--ground-truth-csv",
        help="CSV con landmarks ground truth"
    ),
    predictions: str = typer.Option(
        "outputs/landmark_predictions/session_warping/predictions.npz",
        "--predictions",
        help="NPZ con predicciones del ensemble"
    ),
    pred_color: str = typer.Option(
        "red",
        "--pred-color",
        help="Color para predicciones (red, blue, green, etc.)"
    ),
    gt_color: str = typer.Option(
        "green",
        "--gt-color",
        help="Color para ground truth"
    ),
    cross_size: int = typer.Option(
        5,
        "--cross-size",
        help="Tamaño de cruces en píxeles"
    ),
    cross_thickness: int = typer.Option(
        2,
        "--cross-thickness",
        help="Grosor de líneas"
    ),
    show_error_lines: bool = typer.Option(
        False,
        "--show-error-lines/--no-error-lines",
        help="Dibujar líneas amarillas entre pares correspondientes"
    ),
):
    """
    Generar comparaciones alineadas con el dataset warped usado para clasificación.

    Este comando genera un dataset de comparaciones que mantiene EXACTAMENTE la misma
    estructura de carpetas, splits y nombres de archivo que el warped dataset usado
    para clasificación, permitiendo un análisis perfecto de la relación entre
    errores de landmarks y performance de clasificación.

    Diferencias vs generate-landmark-comparison-dataset:
    - Usa splits del warped dataset (NO crea nuevos splits)
    - Mantiene estructura de carpetas idéntica al warped dataset
    - Proporciona mapeo 1:1 con el dataset de clasificación
    - Solo procesa ~957 imágenes que tienen ground truth (~6.3% de 15,153)

    Este comando:
    1. Lee la estructura del warped dataset (train/val/test splits)
    2. Para cada imagen en el warped dataset que tiene GT:
       - Carga imagen original (299×299)
       - Genera visualización comparativa (pred vs GT)
       - Guarda en la MISMA ubicación relativa que el warped dataset
    3. Mantiene consistencia perfecta con el dataset de clasificación

    Estructura de salida (espejo del warped dataset):
        output_dir/
        ├── train/
        │   ├── COVID/{name}_comparison.png
        │   ├── Normal/{name}_comparison.png
        │   └── Viral_Pneumonia/{name}_comparison.png
        ├── val/...
        ├── test/...
        ├── metadata.json
        ├── error_statistics.json
        └── image_errors.csv

    Ejemplo:
        python -m src_v2 generate-landmark-comparison-dataset-aligned \\
            outputs/warped_lung_best/session_warping \\
            outputs/landmark_comparisons/aligned_with_classifier

        # Con opciones personalizadas
        python -m src_v2 generate-landmark-comparison-dataset-aligned \\
            outputs/warped_lung_best/session_warping \\
            outputs/landmark_comparisons/aligned \\
            --show-error-lines \\
            --cross-size 7
    """
    from pathlib import Path
    from src_v2.visualization.comparison_viz import generate_comparison_dataset_aligned

    logger.info("=" * 60)
    logger.info("Generate Aligned Landmark Comparison Dataset")
    logger.info("=" * 60)

    # Validar inputs
    warped_path = Path(warped_dataset_dir)
    if not warped_path.exists():
        logger.error(f"Warped dataset directory no existe: {warped_dataset_dir}")
        raise typer.Exit(code=1)

    # Verificar que tiene la estructura esperada
    expected_splits = ['train', 'val', 'test']
    missing_splits = [s for s in expected_splits if not (warped_path / s / 'images.csv').exists()]
    if missing_splits:
        logger.error(f"Warped dataset missing expected splits: {missing_splits}")
        logger.error(f"Expected structure: {warped_dataset_dir}/{{train,val,test}}/images.csv")
        raise typer.Exit(code=1)

    original_images_path = Path(original_images_dir)
    if not original_images_path.exists():
        logger.error(f"Original images directory no existe: {original_images_dir}")
        raise typer.Exit(code=1)

    gt_csv_path = Path(ground_truth_csv)
    if not gt_csv_path.exists():
        logger.error(f"Ground truth CSV no existe: {ground_truth_csv}")
        raise typer.Exit(code=1)

    pred_npz_path = Path(predictions)
    if not pred_npz_path.exists():
        logger.error(f"Predictions NPZ no existe: {predictions}")
        raise typer.Exit(code=1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Ejecutar pipeline
    try:
        metadata = generate_comparison_dataset_aligned(
            warped_dataset_dir=warped_path,
            output_dir=output_path,
            ground_truth_csv=gt_csv_path,
            predictions_npz=pred_npz_path,
            original_images_dir=original_images_path,
            pred_color=pred_color,
            gt_color=gt_color,
            cross_size=cross_size,
            cross_thickness=cross_thickness,
            show_error_lines=show_error_lines,
        )

        logger.info("")
        logger.info("Archivos generados:")
        logger.info(f"  - {output_path / 'metadata.json'}")
        logger.info(f"  - {output_path / 'error_statistics.json'}")
        logger.info(f"  - {output_path / 'image_errors.csv'}")
        logger.info("")
        logger.info("Siguiente paso:")
        logger.info("  - Verificar alineación con el warped dataset")
        logger.info(f"  - Analizar relación entre errores de landmarks y clasificación")

    except Exception as e:
        logger.error(f"Error durante generación: {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command()
def generate_viz_dataset(
    input_dir: str = typer.Argument(
        ...,
        help="Directorio del dataset original (COVID-19_Radiography_Dataset)"
    ),
    output_dir: str = typer.Argument(
        ...,
        help="Directorio de salida para visualizaciones"
    ),
    predictions: str = typer.Option(
        "outputs/landmark_predictions/session_warping/predictions.npz",
        "--predictions",
        help="Cache NPZ de predicciones del ensemble"
    ),
    canonical: str = typer.Option(
        "outputs/shape_analysis/canonical_shape_gpa.json",
        "--canonical",
        help="Forma canónica de GPA"
    ),
    triangles: str = typer.Option(
        "outputs/shape_analysis/canonical_delaunay_triangles.json",
        "--triangles",
        help="Triangulación de Delaunay"
    ),
    csv_path: str = typer.Option(
        "data/coordenadas/coordenadas_maestro.csv",
        "--csv-path",
        help="CSV maestro para splits"
    ),
    viz_size: int = typer.Option(
        299,
        "--viz-size",
        help="Tamaño de visualización (píxeles)"
    ),
    model_size: int = typer.Option(
        224,
        "--model-size",
        help="Tamaño del modelo (píxeles)"
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Seed para splits"
    ),
    max_per_split: Optional[int] = typer.Option(
        None,
        "--max-per-split",
        help="Máximo de imágenes por split (para testing)"
    ),
    dpi: int = typer.Option(
        300,
        "--dpi",
        help="Resolución de visualizaciones"
    ),
    no_numbers: bool = typer.Option(
        False,
        "--no-numbers",
        help="No mostrar números en landmarks"
    ),
):
    """
    Generar dataset de visualizaciones científicas con landmarks y triangulación.

    Crea visualizaciones mostrando:
    - Radiografía original 299x299 (sin preprocesamiento)
    - Predicciones exactas del ensemble best (3.61 px)
    - Malla de triangulación de Delaunay sobre landmarks predichos

    Las visualizaciones usan:
    - Splits idénticos al clasificador (seed=42)
    - Predicciones cacheadas del ensemble
    - Coordenadas escaladas correctamente de 224x224 a 299x299

    Ejemplo:
        # Generar dataset completo
        python -m src_v2 generate-viz-dataset \\
            data/dataset/COVID-19_Radiography_Dataset \\
            outputs/viz_dataset

        # Muestra pequeña para testing
        python -m src_v2 generate-viz-dataset \\
            data/dataset/COVID-19_Radiography_Dataset \\
            outputs/viz_dataset_test \\
            --max-per-split 5
    """
    from src_v2.visualization.scientific_viz import generate_viz_dataset as gen_viz

    logger.info("Iniciando generación de dataset de visualizaciones")
    logger.info("")

    # Convertir paths
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    predictions_path = Path(predictions)
    canonical_path = Path(canonical)
    triangles_path = Path(triangles)
    csv_path_obj = Path(csv_path)

    # Validar que existan los archivos necesarios
    if not input_path.exists():
        logger.error(f"Directorio de entrada no existe: {input_path}")
        raise typer.Exit(code=1)

    if not predictions_path.exists():
        logger.error(f"Cache de predicciones no existe: {predictions_path}")
        logger.error("Genera predicciones primero con scripts/predict_landmarks_dataset.py")
        raise typer.Exit(code=1)

    if not canonical_path.exists():
        logger.error(f"Forma canónica no existe: {canonical_path}")
        logger.error("Genera forma canónica con: python -m src_v2 compute-canonical")
        raise typer.Exit(code=1)

    if not triangles_path.exists():
        logger.error(f"Triangulación no existe: {triangles_path}")
        logger.error("Genera triangulación con: python -m src_v2 compute-canonical")
        raise typer.Exit(code=1)

    if not csv_path_obj.exists():
        logger.error(f"CSV maestro no existe: {csv_path_obj}")
        raise typer.Exit(code=1)

    # Ejecutar generación
    try:
        metadata = gen_viz(
            input_dir=input_path,
            output_dir=output_path,
            predictions_npz=predictions_path,
            canonical_json=canonical_path,
            triangles_json=triangles_path,
            csv_path=csv_path_obj,
            viz_size=viz_size,
            model_size=model_size,
            seed=seed,
            splits=(0.75, 0.125, 0.125),
            max_per_split=max_per_split,
            dpi=dpi,
            show_numbers=not no_numbers
        )

        logger.info("")
        logger.info("Dataset de visualizaciones generado exitosamente!")
        logger.info(f"Output: {output_path}")
        logger.info(f"Metadata: {output_path / 'metadata.json'}")

    except Exception as e:
        logger.error(f"Error al generar dataset: {e}", exc_info=True)
        raise typer.Exit(code=1)


def main():
    """Entry point principal."""
    app()


if __name__ == "__main__":
    main()
