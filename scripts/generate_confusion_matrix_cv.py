#!/usr/bin/env python3
"""
Script para generar matriz de confusión visual de validación cruzada en test set.
Genera F5.7_matriz_confusion_cv.png con métricas agregadas de los 5 folds evaluados en el test set fijo.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_cv_results(cv_dir: Path) -> dict:
    """
    Carga resultados de validación cruzada de los 5 folds evaluados en test set.

    Args:
        cv_dir: Directorio con resultados de CV

    Returns:
        Dict con matrices de confusión y métricas agregadas del test set
    """
    confusion_matrices = []
    fold_metrics = []
    accuracies = []
    f1_macros = []
    f1_weighteds = []

    # Cargar resultados de test de cada fold
    for fold in range(1, 6):
        fold_path = cv_dir / f"fold_{fold:02d}" / "test_results.json"

        if not fold_path.exists():
            raise FileNotFoundError(f"No se encontró: {fold_path}")

        with open(fold_path) as f:
            results = json.load(f)

        cm = np.array(results["confusion_matrix"])
        confusion_matrices.append(cm)
        fold_metrics.append(results["metrics"])
        accuracies.append(results["metrics"]["accuracy"])
        f1_macros.append(results["metrics"]["f1_macro"])
        f1_weighteds.append(results["metrics"]["f1_weighted"])

    # Agregar matrices de confusión del test set (5 evaluaciones del mismo test set)
    aggregated_cm = np.sum(confusion_matrices, axis=0)

    # Calcular métricas promedio y desviación estándar
    accuracy_mean = np.mean(accuracies)
    accuracy_std = np.std(accuracies)
    f1_macro_mean = np.mean(f1_macros)
    f1_macro_std = np.std(f1_macros)
    f1_weighted_mean = np.mean(f1_weighteds)
    f1_weighted_std = np.std(f1_weighteds)

    # Calcular precision, recall, F1 por clase desde la matriz agregada
    per_class_metrics = {}
    class_names = ["COVID", "Normal", "Viral_Pneumonia"]

    precisions = []
    recalls = []
    f1_scores = []

    for i, class_name in enumerate(class_names):
        tp = aggregated_cm[i, i]
        fp = np.sum(aggregated_cm[:, i]) - tp
        fn = np.sum(aggregated_cm[i, :]) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        per_class_metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": int(np.sum(aggregated_cm[i, :]))
        }

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Total de evaluaciones (5 modelos × test set)
    test_set_size = int(np.sum(confusion_matrices[0]))
    total_evaluations = len(confusion_matrices) * test_set_size

    return {
        "confusion_matrix": aggregated_cm,
        "class_names": class_names,
        "accuracy": accuracy_mean,
        "accuracy_std": accuracy_std,
        "f1_macro": f1_macro_mean,
        "f1_macro_std": f1_macro_std,
        "f1_weighted": f1_weighted_mean,
        "f1_weighted_std": f1_weighted_std,
        "per_class_metrics": per_class_metrics,
        "test_set_size": test_set_size,
        "total_evaluations": total_evaluations,
        "n_folds": len(confusion_matrices)
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    title: str,
    output_path: Path,
    accuracy: float,
    accuracy_std: float,
    f1_macro: float,
    f1_macro_std: float,
    test_set_size: int,
    total_evaluations: int,
    n_folds: int,
    colorbar_label: str,
    x_label: str,
    y_label: str,
):
    """
    Genera un heatmap de la matriz de confusión agregada del test set.

    Args:
        cm: Matriz de confusión agregada (array 3x3)
        class_names: Nombres de las clases
        title: Título del gráfico
        output_path: Ruta de salida
        accuracy: Accuracy promedio
        accuracy_std: Desviación estándar de accuracy
        f1_macro: F1-Score macro promedio
        f1_macro_std: Desviación estándar de F1-macro
        test_set_size: Tamaño del test set
        total_evaluations: Total de evaluaciones (n_folds × test_set_size)
        n_folds: Número de folds
        colorbar_label: Etiqueta de la barra de color
        x_label: Etiqueta del eje X
        y_label: Etiqueta del eje Y
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    # Calcular porcentajes
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Crear heatmap
    heatmap = sns.heatmap(
        cm_percent,
        annot=False,
        fmt='.1f',
        cmap='Blues',
        cbar_kws={'label': colorbar_label},
        ax=ax,
        vmin=0,
        vmax=100
    )
    cbar = heatmap.collections[0].colorbar
    cbar.set_label(colorbar_label, fontsize=15, fontweight='bold')
    cbar.ax.tick_params(labelsize=13)

    # Anotar con valores absolutos y porcentajes
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = cm[i, j]
            percent = cm_percent[i, j]

            # Color del texto (blanco para celdas oscuras)
            if percent > 50:
                text_color = 'white'
            else:
                text_color = 'black'

            # Negrita para la diagonal
            if i == j:
                weight = 'bold'
                text = f'{value}\n({percent:.1f}%)'
            else:
                weight = 'normal'
                text = f'{value}\n({percent:.1f}%)'

            ax.text(
                j + 0.5, i + 0.5, text,
                ha='center', va='center',
                color=text_color,
                fontsize=15,
                weight=weight
            )

    # Configurar ejes
    ax.set_xlabel(x_label, fontsize=17, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=17, fontweight='bold')
    ax.set_title(
        f'{title}\n'
        f'Validación Cruzada (k={n_folds}) - Test Set ({test_set_size:,} muestras × {n_folds} modelos = {total_evaluations:,} evaluaciones)\n'
        f'Accuracy: {accuracy*100:.2f}% ± {accuracy_std*100:.2f}% | F1-Macro: {f1_macro*100:.2f}% ± {f1_macro_std*100:.2f}%',
        fontsize=18,
        fontweight='bold',
        pad=20
    )

    # Etiquetas de las clases
    ax.set_xticklabels(class_names, rotation=0, ha='center')
    ax.set_yticklabels(class_names, rotation=0)
    ax.tick_params(axis='both', labelsize=15)

    # Ajustar layout
    plt.tight_layout()

    # Guardar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Matriz de confusión guardada: {output_path}")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera matriz de confusión de validación cruzada."
    )
    parser.add_argument(
        "--cv-dir",
        type=Path,
        default=Path("outputs/classifier_cv"),
        help="Directorio con resultados de CV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/Tesis/Figures"),
        help="Directorio de salida para las figuras.",
    )
    parser.add_argument(
        "--lang",
        choices=["es", "en"],
        default="es",
        help="Idioma de los textos en la figura.",
    )
    return parser.parse_args()


def resolve_output_dir(base_dir: Path, output_dir: Path) -> Path:
    if output_dir.is_absolute():
        return output_dir
    return (base_dir / output_dir).resolve()


def main():
    # Rutas
    base_dir = Path(__file__).parent.parent
    args = parse_args()
    cv_dir = base_dir / args.cv_dir if not args.cv_dir.is_absolute() else args.cv_dir
    output_dir = resolve_output_dir(base_dir, args.output_dir)

    # Configuración de idiomas
    language_configs = {
        "es": {
            "colorbar": "Porcentaje (%)",
            "xlabel": "Predicción",
            "ylabel": "Categoría Real",
            "title": "Matriz de Confusión - Clasificador Normalizado + SAHS",
            "class_display_names": {
                "COVID": "COVID-19",
                "Normal": "Normal",
                "Viral_Pneumonia": "Neumonía Viral",
            },
        },
        "en": {
            "colorbar": "Percentage (%)",
            "xlabel": "Prediction",
            "ylabel": "True Label",
            "title": "Confusion Matrix - Warped Classifier + SAHS",
            "class_display_names": {
                "COVID": "COVID-19",
                "Normal": "Normal",
                "Viral_Pneumonia": "Viral Pneumonia",
            },
        },
    }
    lang_config = language_configs[args.lang]

    print("=" * 70)
    print("GENERACIÓN DE MATRIZ DE CONFUSIÓN - VALIDACIÓN CRUZADA")
    print("=" * 70)

    # Cargar resultados de CV
    print(f"\nCargando resultados de: {cv_dir}")
    cv_results = load_cv_results(cv_dir)

    cm = cv_results["confusion_matrix"]
    class_names = cv_results["class_names"]
    accuracy = cv_results["accuracy"]
    accuracy_std = cv_results["accuracy_std"]
    f1_macro = cv_results["f1_macro"]
    f1_macro_std = cv_results["f1_macro_std"]
    test_set_size = cv_results["test_set_size"]
    total_evaluations = cv_results["total_evaluations"]
    n_folds = cv_results["n_folds"]

    display_names = [
        lang_config["class_display_names"].get(name, name) for name in class_names
    ]

    print(f"\nNúmero de folds: {n_folds}")
    print(f"Test set size: {test_set_size:,}")
    print(f"Total evaluations: {total_evaluations:,}")
    print(f"Accuracy: {accuracy*100:.2f}% ± {accuracy_std*100:.2f}%")
    print(f"F1-Macro: {f1_macro*100:.2f}% ± {f1_macro_std*100:.2f}%")
    print(f"F1-Weighted: {cv_results['f1_weighted']*100:.2f}% ± {cv_results['f1_weighted_std']*100:.2f}%")
    print("\nMatriz de confusión agregada (test set):")
    print(cm)

    print("\nMétricas por clase (matriz agregada del test set):")
    for class_name in class_names:
        metrics = cv_results["per_class_metrics"][class_name]
        print(f"  {class_name}:")
        print(f"    Precision: {metrics['precision']*100:.2f}%")
        print(f"    Recall: {metrics['recall']*100:.2f}%")
        print(f"    F1-Score: {metrics['f1-score']*100:.2f}%")
        print(f"    Support: {metrics['support']}")

    # Generar figura
    output_path = output_dir / "F5.7_matriz_confusion_cv.png"
    plot_confusion_matrix(
        cm=cm,
        class_names=display_names,
        title=lang_config["title"],
        output_path=output_path,
        accuracy=accuracy,
        accuracy_std=accuracy_std,
        f1_macro=f1_macro,
        f1_macro_std=f1_macro_std,
        test_set_size=test_set_size,
        total_evaluations=total_evaluations,
        n_folds=n_folds,
        colorbar_label=lang_config["colorbar"],
        x_label=lang_config["xlabel"],
        y_label=lang_config["ylabel"],
    )

    print("\n" + "=" * 70)
    print("PROCESO COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
