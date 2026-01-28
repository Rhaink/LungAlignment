#!/usr/bin/env python3
"""
Script para calcular métricas agregadas de validación cruzada en test set.

Este script:
1. Carga test_results.json de los 5 folds
2. Calcula estadísticas agregadas (mean ± std)
3. Suma matrices de confusión del test set
4. Guarda resultados en cross_validation_test_results.json
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def load_fold_test_results(fold_num: int, base_dir: Path) -> Dict[str, Any]:
    """Carga resultados de test de un fold."""
    fold_dir = base_dir / f"fold_{fold_num:02d}"
    results_path = fold_dir / "test_results.json"

    if not results_path.exists():
        raise FileNotFoundError(f"No se encontró {results_path}")

    with open(results_path, 'r') as f:
        return json.load(f)


def compute_aggregated_metrics(base_dir: Path, n_folds: int = 5) -> Dict[str, Any]:
    """Calcula métricas agregadas de todos los folds en test set."""

    # Cargar resultados de todos los folds
    all_results = []
    for fold in range(1, n_folds + 1):
        results = load_fold_test_results(fold, base_dir)
        all_results.append(results)

    # Extraer métricas globales
    accuracies = [r["metrics"]["accuracy"] for r in all_results]
    f1_macros = [r["metrics"]["f1_macro"] for r in all_results]
    f1_weighteds = [r["metrics"]["f1_weighted"] for r in all_results]

    # Extraer matrices de confusión
    confusion_matrices = [np.array(r["confusion_matrix"]) for r in all_results]
    aggregated_cm = np.sum(confusion_matrices, axis=0)

    # Extraer métricas por clase
    class_names = all_results[0]["class_names"]
    per_class_metrics = {cls: {"precision": [], "recall": [], "f1-score": []}
                         for cls in class_names}

    for results in all_results:
        for cls in class_names:
            per_class_metrics[cls]["precision"].append(results["per_class"][cls]["precision"])
            per_class_metrics[cls]["recall"].append(results["per_class"][cls]["recall"])
            per_class_metrics[cls]["f1-score"].append(results["per_class"][cls]["f1-score"])

    # Calcular estadísticas agregadas
    aggregated = {
        "description": "Cross-validation (k=5) evaluation on fixed test set (1,895 samples)",
        "n_folds": n_folds,
        "test_set_size": all_results[0]["n_samples"],
        "total_evaluations": n_folds * all_results[0]["n_samples"],

        "global_metrics": {
            "accuracy": {
                "mean": float(np.mean(accuracies)),
                "std": float(np.std(accuracies)),
                "min": float(np.min(accuracies)),
                "max": float(np.max(accuracies)),
                "per_fold": accuracies
            },
            "f1_macro": {
                "mean": float(np.mean(f1_macros)),
                "std": float(np.std(f1_macros)),
                "min": float(np.min(f1_macros)),
                "max": float(np.max(f1_macros)),
                "per_fold": f1_macros
            },
            "f1_weighted": {
                "mean": float(np.mean(f1_weighteds)),
                "std": float(np.std(f1_weighteds)),
                "min": float(np.min(f1_weighteds)),
                "max": float(np.max(f1_weighteds)),
                "per_fold": f1_weighteds
            }
        },

        "per_class_metrics": {},

        "aggregated_confusion_matrix": aggregated_cm.tolist(),
        "class_names": class_names,

        "best_fold": {
            "fold_number": int(np.argmax(accuracies) + 1),
            "accuracy": float(np.max(accuracies)),
            "f1_macro": f1_macros[int(np.argmax(accuracies))]
        }
    }

    # Calcular estadísticas por clase
    for cls in class_names:
        aggregated["per_class_metrics"][cls] = {
            "precision": {
                "mean": float(np.mean(per_class_metrics[cls]["precision"])),
                "std": float(np.std(per_class_metrics[cls]["precision"])),
            },
            "recall": {
                "mean": float(np.mean(per_class_metrics[cls]["recall"])),
                "std": float(np.std(per_class_metrics[cls]["recall"])),
            },
            "f1-score": {
                "mean": float(np.mean(per_class_metrics[cls]["f1-score"])),
                "std": float(np.std(per_class_metrics[cls]["f1-score"])),
            }
        }

    return aggregated


def main():
    """Ejecuta el cálculo de métricas agregadas."""
    base_dir = Path("outputs/classifier_cv")
    output_path = base_dir / "cross_validation_test_results.json"

    print("=" * 60)
    print("Calculando métricas agregadas de CV en test set")
    print("=" * 60)

    # Calcular métricas
    aggregated = compute_aggregated_metrics(base_dir, n_folds=5)

    # Mostrar resultados
    print(f"\nTest set size: {aggregated['test_set_size']}")
    print(f"Total evaluations: {aggregated['total_evaluations']}")
    print()
    print("Global Metrics (mean ± std):")
    print(f"  Accuracy:    {aggregated['global_metrics']['accuracy']['mean']:.4f} ± {aggregated['global_metrics']['accuracy']['std']:.4f}")
    print(f"  F1-Macro:    {aggregated['global_metrics']['f1_macro']['mean']:.4f} ± {aggregated['global_metrics']['f1_macro']['std']:.4f}")
    print(f"  F1-Weighted: {aggregated['global_metrics']['f1_weighted']['mean']:.4f} ± {aggregated['global_metrics']['f1_weighted']['std']:.4f}")
    print()
    print(f"Best fold: {aggregated['best_fold']['fold_number']} (accuracy: {aggregated['best_fold']['accuracy']:.4f})")
    print()
    print("Aggregated Confusion Matrix (5 × 1,895 = 9,475 evaluations):")
    cm = np.array(aggregated['aggregated_confusion_matrix'])
    print(cm)
    print()

    # Guardar resultados
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)

    print(f"Resultados guardados en: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
