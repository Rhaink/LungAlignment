#!/usr/bin/env python3
"""
Genera all_landmarks.npz desde coordenadas_maestro.csv con splits reproducibles.

Uso:
  python scripts/generate_all_landmarks_npz.py
  python scripts/generate_all_landmarks_npz.py --compare-to outputs/predictions/all_landmarks.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src_v2.constants import DEFAULT_IMAGE_SIZE, ORIGINAL_IMAGE_SIZE, NUM_LANDMARKS
from src_v2.data.utils import load_coordinates_csv


def build_arrays(df, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    landmarks = []
    image_names = []
    categories = []

    for _, row in df.iterrows():
        image_names.append(row["image_name"])
        categories.append(row["category"])
        coords = []
        for i in range(1, NUM_LANDMARKS + 1):
            coords.append([row[f"L{i}_x"], row[f"L{i}_y"]])
        lm = np.array(coords, dtype=np.float64) * scale
        landmarks.append(lm)

    return (
        np.array(landmarks, dtype=np.float64),
        np.array(image_names, dtype=object),
        np.array(categories, dtype=object),
    )


def compare_npz(reference_path: Path, candidate_path: Path, atol: float) -> bool:
    with np.load(reference_path, allow_pickle=True) as ref_data:
        ref_keys = set(ref_data.keys())
        with np.load(candidate_path, allow_pickle=True) as cand_data:
            cand_keys = set(cand_data.keys())

            if ref_keys != cand_keys:
                missing = ref_keys - cand_keys
                extra = cand_keys - ref_keys
                if missing:
                    print(f"[compare] Keys missing in candidate: {sorted(missing)}")
                if extra:
                    print(f"[compare] Extra keys in candidate: {sorted(extra)}")
                return False

            ok = True
            for key in sorted(ref_keys):
                ref_arr = ref_data[key]
                cand_arr = cand_data[key]

                if ref_arr.shape != cand_arr.shape:
                    print(
                        f"[compare] Shape mismatch {key}: "
                        f"{ref_arr.shape} vs {cand_arr.shape}"
                    )
                    ok = False
                    continue

                if ref_arr.dtype.kind in {"U", "S", "O"} or cand_arr.dtype.kind in {"U", "S", "O"}:
                    same = np.array_equal(ref_arr, cand_arr)
                    if not same:
                        print(f"[compare] Values mismatch {key} (object arrays)")
                        ok = False
                    continue

                max_diff = float(np.max(np.abs(ref_arr - cand_arr))) if ref_arr.size else 0.0
                if max_diff > atol:
                    print(f"[compare] Values mismatch {key}: max_diff={max_diff:.6f}")
                    ok = False

            if ok:
                print("[compare] OK: candidate matches reference")
            return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera all_landmarks.npz desde coordenadas_maestro.csv"
    )
    parser.add_argument(
        "--csv",
        default="data/coordenadas/coordenadas_maestro.csv",
        help="Ruta al CSV con coordenadas (sin headers)",
    )
    parser.add_argument(
        "--output",
        default="outputs/predictions/all_landmarks.npz",
        help="Ruta de salida para el .npz generado",
    )
    parser.add_argument("--train-ratio", type=float, default=0.75, help="Proporcion train")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Proporcion val")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Proporcion test")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para splits")
    parser.add_argument(
        "--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Tamano objetivo (px)"
    )
    parser.add_argument(
        "--original-size",
        type=int,
        default=ORIGINAL_IMAGE_SIZE,
        help="Tamano original de imagen (px)",
    )
    parser.add_argument(
        "--compare-to",
        default=None,
        help="Ruta a un all_landmarks.npz existente para comparar",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Tolerancia absoluta para comparar landmarks",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    output_path = Path(args.output)

    ratios = [args.train_ratio, args.val_ratio, args.test_ratio]
    ratio_sum = sum(ratios)
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Ratios deben sumar 1.0, suman {ratio_sum}")

    scale = args.image_size / args.original_size

    df = load_coordinates_csv(str(csv_path))
    all_landmarks, all_names, all_categories = build_arrays(df, scale)

    train_df, temp_df = train_test_split(
        df,
        test_size=(args.val_ratio + args.test_ratio),
        stratify=df["category"],
        random_state=args.seed,
    )
    val_ratio = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio),
        stratify=temp_df["category"],
        random_state=args.seed,
    )

    train_landmarks, train_names, train_categories = build_arrays(train_df, scale)
    val_landmarks, val_names, val_categories = build_arrays(val_df, scale)
    test_landmarks, test_names, test_categories = build_arrays(test_df, scale)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        all_landmarks=all_landmarks,
        all_image_names=all_names,
        all_categories=all_categories,
        train_landmarks=train_landmarks,
        train_image_names=train_names,
        train_categories=train_categories,
        val_landmarks=val_landmarks,
        val_image_names=val_names,
        val_categories=val_categories,
        test_landmarks=test_landmarks,
        test_image_names=test_names,
        test_categories=test_categories,
    )

    print(f"[generate] Output: {output_path}")
    print(
        f"[generate] Counts: all={len(all_names)}, "
        f"train={len(train_names)}, val={len(val_names)}, test={len(test_names)}"
    )

    if args.compare_to:
        return 0 if compare_npz(Path(args.compare_to), output_path, args.atol) else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
