#!/usr/bin/env python3
"""
Predice landmarks para todo un dataset y guarda cache en JSON o NPZ.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src_v2.constants import (  # noqa: E402
    DEFAULT_IMAGE_SIZE,
    DEFAULT_CLAHE_CLIP_LIMIT,
    DEFAULT_CLAHE_TILE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_LANDMARKS,
    SYMMETRIC_PAIRS,
)
from src_v2.models import create_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predice landmarks de un dataset completo (ensemble + TTA + CLAHE)."
    )
    parser.add_argument(
        "--input-dir",
        default="data/dataset/COVID-19_Radiography_Dataset",
        help="Directorio del dataset original",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta de salida (.json o .npz)",
    )
    parser.add_argument(
        "--ensemble-config",
        default="configs/ensemble_best.json",
        help="Config JSON con lista de checkpoints del ensemble",
    )
    parser.add_argument(
        "--tta",
        dest="tta",
        action="store_true",
        help="Habilitar TTA (override del config)",
    )
    parser.add_argument(
        "--no-tta",
        dest="tta",
        action="store_false",
        help="Deshabilitar TTA (override del config)",
    )
    parser.set_defaults(tta=None)
    parser.add_argument(
        "--clahe",
        dest="clahe",
        action="store_true",
        help="Habilitar CLAHE (override del config)",
    )
    parser.add_argument(
        "--no-clahe",
        dest="clahe",
        action="store_false",
        help="Deshabilitar CLAHE (override del config)",
    )
    parser.set_defaults(clahe=None)
    parser.add_argument(
        "--clahe-clip",
        type=float,
        default=None,
        help="CLAHE clip limit",
    )
    parser.add_argument(
        "--clahe-tile",
        type=int,
        default=None,
        help="CLAHE tile size",
    )
    parser.add_argument(
        "--classes",
        default="COVID,Normal,Viral Pneumonia",
        help="Clases a procesar separadas por coma",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Dispositivo: auto, cuda, cpu, mps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Tamano de batch para inferencia",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limitar numero de imagenes (debug)",
    )
    return parser.parse_args()


def get_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def detect_architecture_from_checkpoint(state_dict: dict) -> dict:
    use_coord_attention = any("coord_attention" in k for k in state_dict.keys())
    deep_head = "head.9.weight" in state_dict or "head.9.bias" in state_dict

    if deep_head:
        hidden_dim = state_dict.get("head.5.weight", np.zeros((256, 1))).shape[0]
    else:
        weight = state_dict.get("head.2.weight")
        if weight is None:
            hidden_dim = 256
        elif len(weight.shape) == 2:
            hidden_dim = weight.shape[0]
        else:
            hidden_dim = 256

    return {
        "use_coord_attention": use_coord_attention,
        "deep_head": deep_head,
        "hidden_dim": hidden_dim,
    }


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_ensemble_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config no existe: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("Config debe incluir lista 'models'")
    return config


def load_models(checkpoints: list[str], device: torch.device) -> list[torch.nn.Module]:
    models = []
    for ckpt in checkpoints:
        ckpt_path = resolve_path(ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint no existe: {ckpt_path}")
        checkpoint_data = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)

        arch_params = detect_architecture_from_checkpoint(state_dict)
        model = create_model(
            pretrained=False,
            use_coord_attention=arch_params["use_coord_attention"],
            deep_head=arch_params["deep_head"],
            hidden_dim=arch_params["hidden_dim"],
        )
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        models.append(model)
    return models


def preprocess_image(
    image_path: Path,
    use_clahe: bool,
    clahe_clip: float,
    clahe_tile: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> torch.Tensor | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    if image.shape[0] != DEFAULT_IMAGE_SIZE or image.shape[1] != DEFAULT_IMAGE_SIZE:
        image = cv2.resize(image, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))

    img_array = image.copy()
    if use_clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
        img_array = clahe_obj.apply(img_array)

    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    img_normalized = (img_float - mean) / std

    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()
    return img_tensor


def collect_images(input_dir: Path, classes: list[str]) -> list[tuple[Path, str]]:
    class_mapping = {c: c.replace(" ", "_") for c in classes}
    all_images = []

    for class_name in classes:
        class_dir = input_dir / class_name / "images"
        if not class_dir.exists():
            class_dir = input_dir / class_name
        if not class_dir.exists():
            print(f"[warn] Directorio de clase no existe: {class_dir}")
            continue

        images = sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg"))
        mapped_class = class_mapping[class_name]
        all_images.extend([(img, mapped_class) for img in images])
        print(f"  {class_name}: {len(images)} imagenes")

    return all_images


def main() -> int:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Directorio de entrada no existe: {input_dir}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ensemble_path = Path(args.ensemble_config)
    ensemble_config = load_ensemble_config(ensemble_path)
    model_paths = ensemble_config["models"]

    use_tta = args.tta if args.tta is not None else bool(ensemble_config.get("tta", False))
    use_clahe = args.clahe if args.clahe is not None else bool(ensemble_config.get("clahe", True))
    clahe_clip = (
        args.clahe_clip
        if args.clahe_clip is not None
        else float(ensemble_config.get("clahe_clip", DEFAULT_CLAHE_CLIP_LIMIT))
    )
    clahe_tile = (
        args.clahe_tile
        if args.clahe_tile is not None
        else int(ensemble_config.get("clahe_tile", DEFAULT_CLAHE_TILE_SIZE))
    )

    class_list = [c.strip() for c in args.classes.split(",")]
    class_mapping = {c: c.replace(" ", "_") for c in class_list}

    print("=" * 70)
    print("PREDICCION DE LANDMARKS - DATASET COMPLETO")
    print("=" * 70)
    print(f"Input: {input_dir}")
    print(f"Output: {output_path}")
    print(f"Ensemble config: {ensemble_path}")
    print(f"TTA: {use_tta} | CLAHE: {use_clahe} (clip={clahe_clip}, tile={clahe_tile})")

    device = get_device(args.device)
    print(f"Dispositivo: {device}")

    print("\nCargando modelos...")
    models = load_models(model_paths, device)
    print(f"Modelos cargados: {len(models)}")

    print("\nRecolectando imagenes...")
    all_images = collect_images(input_dir, class_list)
    if args.limit:
        all_images = all_images[: args.limit]
    if not all_images:
        raise SystemExit("No se encontraron imagenes")

    print(f"Total imagenes: {len(all_images)}")

    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    @torch.no_grad()
    def predict_ensemble(images: torch.Tensor) -> torch.Tensor:
        preds = []
        for model in models:
            if use_tta:
                pred1 = model(images)
                images_flip = torch.flip(images, dims=[3])
                pred2 = model(images_flip)
                pred2 = pred2.view(-1, NUM_LANDMARKS, 2)
                pred2[:, :, 0] = 1 - pred2[:, :, 0]
                for left, right in SYMMETRIC_PAIRS:
                    pred2[:, [left, right]] = pred2[:, [right, left]]
                pred2 = pred2.view(-1, NUM_LANDMARKS * 2)
                pred = (pred1 + pred2) / 2
            else:
                pred = model(images)
            preds.append(pred)
        preds_stack = torch.stack(preds)
        return preds_stack.mean(dim=0)

    predictions = []
    image_paths = []
    image_names = []
    categories = []
    landmarks_list = []
    failed = []

    batch_tensors = []
    batch_records = []

    print("\nPrediciendo landmarks...")
    for image_path, class_name in tqdm(all_images, desc="Procesando", ncols=80):
        tensor = preprocess_image(
            image_path,
            use_clahe=use_clahe,
            clahe_clip=clahe_clip,
            clahe_tile=clahe_tile,
            mean=mean,
            std=std,
        )
        if tensor is None:
            failed.append(image_path)
            continue

        batch_tensors.append(tensor)
        batch_records.append((image_path, class_name))

        if len(batch_tensors) >= args.batch_size:
            images_tensor = torch.stack(batch_tensors).to(device)
            preds = predict_ensemble(images_tensor)
            preds = preds.view(-1, NUM_LANDMARKS, 2) * DEFAULT_IMAGE_SIZE
            preds_np = preds.cpu().numpy().astype(np.float32)

            for idx, (img_path, category) in enumerate(batch_records):
                rel_path = img_path.relative_to(input_dir).as_posix()
                image_paths.append(rel_path)
                image_names.append(img_path.stem)
                categories.append(category)
                landmarks_list.append(preds_np[idx])
                predictions.append({
                    "image_path": rel_path,
                    "image_name": img_path.stem,
                    "category": category,
                    "landmarks": preds_np[idx].tolist(),
                })

            batch_tensors.clear()
            batch_records.clear()

    if batch_tensors:
        images_tensor = torch.stack(batch_tensors).to(device)
        preds = predict_ensemble(images_tensor)
        preds = preds.view(-1, NUM_LANDMARKS, 2) * DEFAULT_IMAGE_SIZE
        preds_np = preds.cpu().numpy().astype(np.float32)

        for idx, (img_path, category) in enumerate(batch_records):
            rel_path = img_path.relative_to(input_dir).as_posix()
            image_paths.append(rel_path)
            image_names.append(img_path.stem)
            categories.append(category)
            landmarks_list.append(preds_np[idx])
            predictions.append({
                "image_path": rel_path,
                "image_name": img_path.stem,
                "category": category,
                "landmarks": preds_np[idx].tolist(),
            })

    metadata = {
        "schema_version": "v1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "input_dir": str(input_dir),
        "classes": [class_mapping[c] for c in class_list],
        "models": model_paths,
        "ensemble_config": str(ensemble_path),
        "tta": use_tta,
        "clahe": use_clahe,
        "clahe_clip": clahe_clip,
        "clahe_tile": clahe_tile,
        "image_size": DEFAULT_IMAGE_SIZE,
        "seed": args.seed,
        "device": str(device),
        "batch_size": args.batch_size,
        "total_images": len(all_images),
        "processed_images": len(image_paths),
        "failed_images": [p.relative_to(input_dir).as_posix() for p in failed],
    }

    if output_path.suffix.lower() == ".json":
        payload = {
            "metadata": metadata,
            "predictions": predictions,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    elif output_path.suffix.lower() == ".npz":
        np.savez(
            output_path,
            image_paths=np.array(image_paths, dtype=object),
            image_names=np.array(image_names, dtype=object),
            categories=np.array(categories, dtype=object),
            landmarks=np.array(landmarks_list, dtype=np.float32),
            metadata_json=json.dumps(metadata),
        )
    else:
        raise SystemExit("Formato de salida no soportado (usa .json o .npz)")

    print("\nResumen")
    print(f"- Total: {len(all_images)}")
    print(f"- Procesadas: {len(image_paths)}")
    print(f"- Fallidas: {len(failed)}")
    print(f"- Output: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
