#!/usr/bin/env python3
"""
Script de analisis del dataset de landmarks.
Ejecutar desde la raiz del proyecto:
    python scripts/analyze_data.py
"""

import sys
from pathlib import Path

# Agregar src_v2 al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

from src_v2.data.utils import (
    load_coordinates_csv,
    get_image_path,
    get_landmarks_array,
    compute_statistics,
    compute_symmetry_error,
    SYMMETRIC_PAIRS,
    CENTRAL_LANDMARKS,
    LANDMARK_NAMES
)


def main():
    # Rutas
    data_root = project_root / 'data'
    csv_path = data_root / 'coordenadas' / 'coordenadas_maestro.csv'

    print("=" * 60)
    print("ANALISIS DEL DATASET DE LANDMARKS")
    print("=" * 60)

    # Cargar datos
    print("\n1. Cargando datos...")
    df = load_coordinates_csv(csv_path)
    print(f"   Total de muestras: {len(df)}")

    # Distribucion por categoria
    print("\n2. Distribucion por categoria:")
    for cat, count in df['category'].value_counts().items():
        pct = count / len(df) * 100
        print(f"   {cat}: {count} ({pct:.1f}%)")

    # Verificar imagenes
    print("\n3. Verificando imagenes...")
    missing = 0
    sizes = []
    for idx, row in df.head(50).iterrows():
        img_path = get_image_path(row['image_name'], row['category'], data_root)
        if not img_path.exists():
            missing += 1
            print(f"   MISSING: {img_path}")
        else:
            img = Image.open(img_path)
            sizes.append(img.size)

    if missing == 0:
        print("   Todas las imagenes verificadas existen")
    else:
        print(f"   ADVERTENCIA: {missing} imagenes faltantes")

    # Tamanos de imagen
    if sizes:
        size_counts = Counter(sizes)
        print(f"\n4. Tamanos de imagen encontrados:")
        for size, count in size_counts.items():
            print(f"   {size}: {count} imagenes")

    # Estadisticas de coordenadas
    print("\n5. Estadisticas de coordenadas:")
    stats = compute_statistics(df)
    print(f"\n   Variabilidad por landmark (desviacion estandar total):")

    landmark_vars = []
    for name, s in stats['landmark_stats'].items():
        total_std = np.sqrt(s['x_std']**2 + s['y_std']**2)
        landmark_vars.append((name, total_std))

    # Ordenar por variabilidad
    landmark_vars.sort(key=lambda x: x[1], reverse=True)
    for name, var in landmark_vars:
        difficulty = "DIFICIL" if var > 30 else "MEDIO" if var > 25 else "FACIL"
        print(f"   {name}: σ={var:.1f} px ({difficulty})")

    # Analisis de simetria
    print("\n6. Analisis de simetria en Ground Truth:")
    all_sym_errors = {f'L{l+1}-L{r+1}': [] for l, r in SYMMETRIC_PAIRS}

    for idx, row in df.iterrows():
        landmarks = get_landmarks_array(row)
        sym_errors = compute_symmetry_error(landmarks)
        for pair, error in sym_errors.items():
            all_sym_errors[pair].append(error)

    print(f"\n   Error de simetria por par (promedio +/- std):")
    for pair, errors in all_sym_errors.items():
        mean = np.mean(errors)
        std = np.std(errors)
        print(f"   {pair}: {mean:.1f} +/- {std:.1f} px")

    # Rango de coordenadas
    print("\n7. Rango de coordenadas:")
    for i in range(1, 16):
        x_min, x_max = df[f'L{i}_x'].min(), df[f'L{i}_x'].max()
        y_min, y_max = df[f'L{i}_y'].min(), df[f'L{i}_y'].max()
        print(f"   L{i}: x=[{x_min:.0f}, {x_max:.0f}], y=[{y_min:.0f}, {y_max:.0f}]")

    # Alineacion de centrales
    print("\n8. Analisis de alineacion de landmarks centrales:")
    central_distances = []
    for idx, row in df.iterrows():
        landmarks = get_landmarks_array(row)
        L1, L2 = landmarks[0], landmarks[1]
        eje = L2 - L1
        eje_len = np.linalg.norm(eje)
        if eje_len < 1:
            continue
        eje_unit = eje / eje_len

        for ci in CENTRAL_LANDMARKS:
            vec = landmarks[ci] - L1
            proj = np.dot(vec, eje_unit) * eje_unit
            perp = vec - proj
            dist = np.linalg.norm(perp)
            central_distances.append(dist)

    mean_dist = np.mean(central_distances)
    std_dist = np.std(central_distances)
    print(f"   Distancia promedio de centrales al eje: {mean_dist:.2f} +/- {std_dist:.2f} px")

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Total muestras: {len(df)}")
    print(f"Landmarks mas dificiles: {landmark_vars[0][0]} (σ={landmark_vars[0][1]:.1f}), "
          f"{landmark_vars[1][0]} (σ={landmark_vars[1][1]:.1f})")
    print(f"Landmarks mas faciles: {landmark_vars[-1][0]} (σ={landmark_vars[-1][1]:.1f}), "
          f"{landmark_vars[-2][0]} (σ={landmark_vars[-2][1]:.1f})")
    print(f"Simetria promedio: {np.mean([np.mean(e) for e in all_sym_errors.values()]):.1f} px")
    print(f"Alineacion centrales: {mean_dist:.2f} px (casi perfecta)")

    print("\nDataset listo para entrenamiento.")


if __name__ == '__main__':
    main()
