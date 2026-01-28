#!/usr/bin/env python3
"""
Script para extraer los splits exactos train/val/test del modelo warped_lung_best.

Este script copia tanto las imágenes originales como las warped a un directorio
separado, preservando la estructura de splits y categorías usadas durante el
entrenamiento del modelo que alcanzó 98.05% accuracy.

Usage:
    python scripts/extract_dataset_splits.py [OPTIONS]
"""

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm


def compute_md5(file_path: Path) -> str:
    """Calcula el checksum MD5 de un archivo."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def map_category_to_original_dir(category: str) -> str:
    """
    Mapea el nombre de categoría del CSV al nombre del directorio original.

    Args:
        category: Nombre de categoría en el CSV (e.g., "Viral_Pneumonia")

    Returns:
        Nombre del directorio en el dataset original (e.g., "Viral Pneumonia")
    """
    if category == "Viral_Pneumonia":
        return "Viral Pneumonia"
    return category


def create_directory_structure(output_dir: Path) -> None:
    """Crea la estructura de directorios para el dataset extraído."""
    categories = ["COVID", "Normal", "Viral_Pneumonia"]
    splits = ["train", "val", "test"]

    # Crear directorios para imágenes originales y warped
    for img_type in ["original", "warped"]:
        for split in splits:
            for category in categories:
                dir_path = output_dir / img_type / split / category
                dir_path.mkdir(parents=True, exist_ok=True)

    # Crear directorio de metadata
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)


def copy_image(src: Path, dst: Path, verify: bool = True) -> Tuple[bool, str]:
    """
    Copia una imagen y opcionalmente verifica la integridad.

    Returns:
        Tuple[bool, str]: (éxito, mensaje de error si falla)
    """
    try:
        if not src.exists():
            return False, f"Archivo fuente no existe: {src}"

        # Copiar archivo
        shutil.copy2(src, dst)

        # Verificar integridad si se solicita
        if verify:
            src_md5 = compute_md5(src)
            dst_md5 = compute_md5(dst)
            if src_md5 != dst_md5:
                return False, f"Checksum no coincide: {src} -> {dst}"

        return True, ""
    except Exception as e:
        return False, str(e)


def process_split(
    split: str,
    csv_path: Path,
    original_dir: Path,
    warped_dir: Path,
    output_dir: Path,
    verify: bool = True
) -> Tuple[int, int, List[str]]:
    """
    Procesa un split (train/val/test) copiando imágenes originales y warped.

    Returns:
        Tuple[int, int, List[str]]: (imágenes copiadas, errores, lista de errores)
    """
    print(f"\nProcesando split: {split}")

    # Leer CSV
    df = pd.read_csv(csv_path)
    print(f"  Imágenes en CSV: {len(df)}")

    copied = 0
    errors = []

    # Procesar cada imagen
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  Copiando {split}"):
        image_name = row["image_name"]
        warped_filename = row["warped_filename"]
        category = row["category"]

        # Asegurar que image_name tenga extensión .png
        if not image_name.endswith('.png'):
            image_name_with_ext = f"{image_name}.png"
        else:
            image_name_with_ext = image_name

        # --- Copiar imagen ORIGINAL ---
        category_orig_dir = map_category_to_original_dir(category)
        src_original = original_dir / category_orig_dir / "images" / image_name_with_ext
        dst_original = output_dir / "original" / split / category / image_name_with_ext

        success, error_msg = copy_image(src_original, dst_original, verify=verify)
        if not success:
            errors.append(f"[{split}] Original {image_name_with_ext}: {error_msg}")
            continue

        # --- Copiar imagen WARPED ---
        src_warped = warped_dir / split / category / warped_filename
        dst_warped = output_dir / "warped" / split / category / warped_filename

        success, error_msg = copy_image(src_warped, dst_warped, verify=verify)
        if not success:
            errors.append(f"[{split}] Warped {warped_filename}: {error_msg}")
            continue

        copied += 1

    print(f"  ✓ Copiadas: {copied}/{len(df)} imágenes")
    if errors:
        print(f"  ✗ Errores: {len(errors)}")

    return copied, len(df), errors


def copy_metadata_csv(csv_path: Path, output_dir: Path, split: str) -> None:
    """Copia el archivo images.csv a metadata."""
    dst_path = output_dir / "metadata" / f"{split}_images.csv"
    shutil.copy2(csv_path, dst_path)
    print(f"  ✓ Copiado CSV de metadata: {dst_path.name}")


def generate_dataset_info(
    output_dir: Path,
    warped_dir: Path,
    counts: Dict[str, Dict[str, int]]
) -> None:
    """Genera el archivo dataset_info.json con metadata completa."""

    # Leer dataset_summary.json del dataset warped original
    summary_path = warped_dir / "dataset_summary.json"
    summary_data = {}
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary_data = json.load(f)

    # Generar metadata
    info = {
        "extraction_date": datetime.now().isoformat(),
        "source_model": "warped_lung_best",
        "model_accuracy": 0.9805,
        "model_checkpoint": "outputs/classifier_warped_lung_best/sweeps_2026-01-12/lr2e-4_seed321_on/best_classifier.pt",
        "dataset_seed": 42,
        "split_ratios": {
            "train": 0.75,
            "val": 0.125,
            "test": 0.125
        },
        "splits": counts,
        "total_images": sum(sum(split_counts.values()) for split_counts in counts.values()),
        "categories": ["COVID", "Normal", "Viral_Pneumonia"],
        "source_warped_dir": str(warped_dir),
        "warping_config": summary_data.get("warping_config", {}),
        "landmark_models": summary_data.get("landmark_models", []),
        "ground_truth_reference": "GROUND_TRUTH.json",
        "notes": [
            "Este dataset contiene los splits EXACTOS usados para entrenar/validar/testear el modelo warped_lung_best",
            "Seed fijo (42) garantiza reproducibilidad de los splits",
            "Las imágenes 'original' son las originales del dataset, las 'warped' son las normalizadas geométricamente",
            "NO modificar este dataset - es la referencia de entrenamiento. Para experimentos, hacer copias."
        ]
    }

    # Guardar archivo
    info_path = output_dir / "metadata" / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n✓ Generado: {info_path}")


def generate_readme(output_dir: Path, counts: Dict[str, Dict[str, int]]) -> None:
    """Genera el archivo README.md con documentación del dataset."""

    readme_content = f"""# Dataset Splits para warped_lung_best

## Descripción

Este directorio contiene los splits **exactos** de train/validation/test utilizados para entrenar el modelo `warped_lung_best` que alcanzó **98.05% accuracy** en la clasificación de COVID-19 desde radiografías de tórax.

**⚠️ IMPORTANTE:** Este dataset es una referencia inmutable. NO debe modificarse. Para experimentos, crear copias.

## Modelo Asociado

- **Nombre:** warped_lung_best
- **Accuracy:** 98.05%
- **Checkpoint:** `outputs/classifier_warped_lung_best/sweeps_2026-01-12/lr2e-4_seed321_on/best_classifier.pt`
- **Dataset fuente:** `outputs/warped_lung_best/session_warping`
- **Seed:** 42 (fijo, reproducible)

## Estructura del Dataset

```
dataset_splits_for_gui/
├── original/                   # Imágenes originales sin warping
│   ├── train/
│   │   ├── COVID/              # {counts['train']['COVID']} imágenes
│   │   ├── Normal/             # {counts['train']['Normal']} imágenes
│   │   └── Viral_Pneumonia/    # {counts['train']['Viral_Pneumonia']} imágenes
│   ├── val/
│   │   ├── COVID/              # {counts['val']['COVID']} imágenes
│   │   ├── Normal/             # {counts['val']['Normal']} imágenes
│   │   └── Viral_Pneumonia/    # {counts['val']['Viral_Pneumonia']} imágenes
│   └── test/
│       ├── COVID/              # {counts['test']['COVID']} imágenes
│       ├── Normal/             # {counts['test']['Normal']} imágenes
│       └── Viral_Pneumonia/    # {counts['test']['Viral_Pneumonia']} imágenes
├── warped/                     # Imágenes normalizadas (las que vio el modelo)
│   ├── train/
│   │   ├── COVID/
│   │   ├── Normal/
│   │   └── Viral_Pneumonia/
│   ├── val/
│   │   ├── COVID/
│   │   ├── Normal/
│   │   └── Viral_Pneumonia/
│   └── test/
│       ├── COVID/
│       ├── Normal/
│       └── Viral_Pneumonia/
├── metadata/
│   ├── train_images.csv        # Mapeo imagen original → warped para train
│   ├── val_images.csv          # Mapeo imagen original → warped para val
│   ├── test_images.csv         # Mapeo imagen original → warped para test
│   └── dataset_info.json       # Metadata completa (seed, splits, counts, etc.)
└── README.md                   # Este archivo
```

## Composición del Dataset

### Train Split (75% - {sum(counts['train'].values())} imágenes)
- Normal: {counts['train']['Normal']:,}
- COVID: {counts['train']['COVID']:,}
- Viral_Pneumonia: {counts['train']['Viral_Pneumonia']:,}

### Validation Split (12.5% - {sum(counts['val'].values())} imágenes)
- Normal: {counts['val']['Normal']:,}
- COVID: {counts['val']['COVID']:,}
- Viral_Pneumonia: {counts['val']['Viral_Pneumonia']:,}

### Test Split (12.5% - {sum(counts['test'].values())} imágenes)
- Normal: {counts['test']['Normal']:,}
- COVID: {counts['test']['COVID']:,}
- Viral_Pneumonia: {counts['test']['Viral_Pneumonia']:,}

**Total:** {sum(sum(split_counts.values()) for split_counts in counts.values()):,} imágenes

## Cómo Usar

### En la GUI
El directorio `test/` puede usarse directamente en la interfaz para demostración:
```bash
python -m src_v2 gui --test-dir outputs/dataset_splits_for_gui/original/test
```

### En PyTorch (ImageFolder)
```python
from torchvision import datasets, transforms

# Cargar test set original
test_dataset = datasets.ImageFolder(
    'outputs/dataset_splits_for_gui/original/test',
    transform=transforms.ToTensor()
)

# Cargar test set warped (el que vio el modelo)
test_warped = datasets.ImageFolder(
    'outputs/dataset_splits_for_gui/warped/test',
    transform=transforms.ToTensor()
)
```

### Validar Splits
```python
import pandas as pd

# Leer mapeos
train_df = pd.read_csv('outputs/dataset_splits_for_gui/metadata/train_images.csv')
val_df = pd.read_csv('outputs/dataset_splits_for_gui/metadata/val_images.csv')
test_df = pd.read_csv('outputs/dataset_splits_for_gui/metadata/test_images.csv')

# Ver distribución de clases
print(train_df['category'].value_counts())
```

## Notas Importantes

1. **Seed Fijo (42):** Los splits usan seed 42, garantizando reproducibilidad exacta.

2. **Stratified Splits:** Las divisiones están balanceadas por clase según la distribución original del dataset.

3. **Nombres de Archivos:**
   - Originales: mantienen nombres originales (e.g., `Normal-9639.png`, `COVID-29.png`)
   - Warped: tienen sufijo `_warped.png`

4. **Mapeo Original ↔ Warped:** Los archivos CSV en `metadata/` contienen el mapeo completo entre imágenes originales y sus versiones warped.

5. **Uso en Evaluación:** El test set NUNCA fue visto por el modelo durante entrenamiento (solo en evaluación final), por lo que es ideal para demos y validación.

## Metadata Adicional

Ver `metadata/dataset_info.json` para:
- Configuración de warping usada
- Modelos de landmarks utilizados
- Fecha de extracción
- Checksums (si se habilitó verificación)

## Referencias

- **Ground Truth:** Ver `GROUND_TRUTH.json` en la raíz del proyecto
- **Documentación completa:** `docs/REPRO_FULL_PIPELINE.md`
- **Resultados experimentales:** `docs/EXPERIMENTS.md`

---

**Generado:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)

    print(f"✓ Generado: {readme_path}")


def count_images_in_output(output_dir: Path) -> Dict[str, Dict[str, int]]:
    """Cuenta las imágenes copiadas en el directorio de salida."""
    counts = {}

    for split in ["train", "val", "test"]:
        counts[split] = {}
        for category in ["COVID", "Normal", "Viral_Pneumonia"]:
            # Contar en directorio original (asumiendo que original y warped tienen mismo count)
            img_dir = output_dir / "original" / split / category
            img_count = len(list(img_dir.glob("*.png")))
            counts[split][category] = img_count

    return counts


def print_summary(counts: Dict[str, Dict[str, int]], errors: List[str]) -> None:
    """Imprime un resumen final de la extracción."""
    print("\n" + "="*60)
    print("RESUMEN DE EXTRACCIÓN")
    print("="*60)

    for split in ["train", "val", "test"]:
        total = sum(counts[split].values())
        print(f"\n{split.upper()} ({total:,} imágenes):")
        for category, count in counts[split].items():
            print(f"  - {category}: {count:,}")

    total_images = sum(sum(split_counts.values()) for split_counts in counts.values())
    print(f"\nTOTAL: {total_images:,} imágenes copiadas (× 2 versiones = {total_images * 2:,} archivos)")

    if errors:
        print(f"\n⚠️  ERRORES ENCONTRADOS: {len(errors)}")
        print("Ver detalles arriba o ejecutar con --verbose")
    else:
        print("\n✅ Extracción completada sin errores")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Extrae los splits train/val/test del modelo warped_lung_best"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dataset_splits_for_gui"),
        help="Directorio de salida (default: outputs/dataset_splits_for_gui)"
    )
    parser.add_argument(
        "--warped-dir",
        type=Path,
        default=Path("outputs/warped_lung_best/session_warping"),
        help="Directorio del dataset warped fuente (default: outputs/warped_lung_best/session_warping)"
    )
    parser.add_argument(
        "--original-dir",
        type=Path,
        default=Path("data/dataset/COVID-19_Radiography_Dataset"),
        help="Directorio del dataset original (default: data/dataset/COVID-19_Radiography_Dataset)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Desactivar verificación de checksums (más rápido, pero sin garantía de integridad)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar todos los errores detalladamente"
    )

    args = parser.parse_args()

    # Validar directorios de entrada
    if not args.warped_dir.exists():
        print(f"❌ Error: Directorio warped no existe: {args.warped_dir}")
        return 1

    if not args.original_dir.exists():
        print(f"❌ Error: Directorio original no existe: {args.original_dir}")
        return 1

    # Crear estructura de directorios
    print("Creando estructura de directorios...")
    create_directory_structure(args.output_dir)
    print(f"✓ Estructura creada en: {args.output_dir}")

    # Procesar cada split
    all_errors = []
    splits = ["train", "val", "test"]

    for split in splits:
        csv_path = args.warped_dir / split / "images.csv"

        if not csv_path.exists():
            print(f"❌ Error: CSV no encontrado: {csv_path}")
            return 1

        # Copiar imágenes
        copied, total, errors = process_split(
            split=split,
            csv_path=csv_path,
            original_dir=args.original_dir,
            warped_dir=args.warped_dir,
            output_dir=args.output_dir,
            verify=not args.no_verify
        )

        all_errors.extend(errors)

        # Copiar CSV de metadata
        copy_metadata_csv(csv_path, args.output_dir, split)

    # Contar imágenes en el directorio de salida
    print("\nContando imágenes en directorio de salida...")
    counts = count_images_in_output(args.output_dir)

    # Generar metadata
    print("\nGenerando metadata...")
    generate_dataset_info(args.output_dir, args.warped_dir, counts)

    # Generar README
    print("Generando README...")
    generate_readme(args.output_dir, counts)

    # Mostrar errores detallados si se solicita
    if all_errors and args.verbose:
        print("\n" + "="*60)
        print("ERRORES DETALLADOS")
        print("="*60)
        for error in all_errors:
            print(f"  {error}")

    # Imprimir resumen final
    print_summary(counts, all_errors)

    return 0 if not all_errors else 1


if __name__ == "__main__":
    exit(main())
