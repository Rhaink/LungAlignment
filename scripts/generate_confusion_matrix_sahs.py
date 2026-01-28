#!/usr/bin/env python3
"""
Script para generar matriz de confusión visual del experimento SAHS.
Genera la matriz de confusión del clasificador warped_sahs_masked
con los valores correctos verificados.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    title: str,
    output_path: Path,
    accuracy: float,
    f1_macro: float,
    colorbar_label: str,
    x_label: str,
    y_label: str,
):
    """
    Genera un heatmap de la matriz de confusión.

    Args:
        cm: Matriz de confusión (array 3x3)
        class_names: Nombres de las clases
        title: Título del gráfico
        output_path: Ruta de salida
        accuracy: Accuracy del modelo
        f1_macro: F1-Score macro del modelo
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
        f'{title}\nAccuracy: {accuracy:.2f}% | F1-Macro: {f1_macro:.2f}%',
        fontsize=20,
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
        description="Genera matrices de confusion para el experimento SAHS."
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
    parser.add_argument(
        "--output-name",
        default="F5.7_matriz_confusion_sahs.png",
        help="Nombre del archivo de salida para la matriz de confusión.",
    )
    parser.add_argument(
        "--comparison-name",
        default="F5.8_comparacion_sahs.png",
        help="Nombre del archivo de salida para la comparación.",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Si se indica, no genera la comparación de configuraciones.",
    )
    parser.add_argument(
        "--override-accuracy",
        type=float,
        default=None,
        help="Override de accuracy (en porcentaje) para el título.",
    )
    parser.add_argument(
        "--override-f1-macro",
        type=float,
        default=None,
        help="Override de F1-macro (en porcentaje) para el título.",
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
    results_path = base_dir / "outputs/classifier_warped_sahs_masked/results.json"
    output_dir = resolve_output_dir(base_dir, args.output_dir)

    # Cargar resultados
    with open(results_path) as f:
        results = json.load(f)

    # Extraer datos
    cm = np.array(results["confusion_matrix"])
    class_names = results["class_names"]
    accuracy = results["test_metrics"]["accuracy"] * 100
    f1_macro = results["test_metrics"]["f1_macro"] * 100

    language_configs = {
        "es": {
            "colorbar": "Porcentaje (%)",
            "xlabel": "Predicción",
            "ylabel": "Categoría Real",
            "title": "Matriz de Confusión - Clasificador Normalizado + SAHS",
            "comparison_title": "Comparación de Configuraciones SAHS",
            "comparison_xlabel": "Predicción",
            "comparison_ylabel": "Real",
            "class_display_names": {
                "COVID": "COVID-19",
                "Normal": "Normal",
                "Viral_Pneumonia": "Neumonía Viral",
            },
            "config_names": {
                "original": "Original + SAHS",
                "normalized": "Normalizado + SAHS",
                "cropped": "Cropped + SAHS",
            },
        },
        "en": {
            "colorbar": "Percentage (%)",
            "xlabel": "Prediction",
            "ylabel": "True Label",
            "title": "Confusion Matrix - Warped Classifier + SAHS",
            "comparison_title": "SAHS Configuration Comparison",
            "comparison_xlabel": "Prediction",
            "comparison_ylabel": "True",
            "class_display_names": {
                "COVID": "COVID-19",
                "Normal": "Normal",
                "Viral_Pneumonia": "Viral Pneumonia",
            },
            "config_names": {
                "original": "Original + SAHS",
                "normalized": "Warped + SAHS",
                "cropped": "Cropped + SAHS",
            },
        },
    }
    lang_config = language_configs[args.lang]
    display_names = [
        lang_config["class_display_names"].get(name, name) for name in class_names
    ]

    print("=" * 70)
    print("GENERACIÓN DE MATRIZ DE CONFUSIÓN - EXPERIMENTO SAHS")
    print("=" * 70)
    print(f"\nArchivo de resultados: {results_path}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"F1-Macro: {f1_macro:.2f}%")
    print(f"Test samples: {results['test_samples']}")
    print("\nMatriz de confusión:")
    print(cm)

    display_accuracy = (
        args.override_accuracy if args.override_accuracy is not None else accuracy
    )
    display_f1_macro = (
        args.override_f1_macro if args.override_f1_macro is not None else f1_macro
    )
    if args.override_accuracy is not None or args.override_f1_macro is not None:
        print(
            f"Usando override en título -> "
            f"Accuracy: {display_accuracy:.2f}% | F1-Macro: {display_f1_macro:.2f}%"
        )

    # Generar figura
    output_path = output_dir / args.output_name
    plot_confusion_matrix(
        cm=cm,
        class_names=display_names,
        title=lang_config["title"],
        output_path=output_path,
        accuracy=display_accuracy,
        f1_macro=display_f1_macro,
        colorbar_label=lang_config["colorbar"],
        x_label=lang_config["xlabel"],
        y_label=lang_config["ylabel"],
    )

    if args.skip_comparison:
        print("\n" + "=" * 70)
        print("COMPARACIÓN OMITIDA (flag --skip-comparison)")
        print("=" * 70)
        return

    # Generar también comparación de las 3 configuraciones
    print("\n" + "=" * 70)
    print("GENERACIÓN DE COMPARACIÓN DE CONFIGURACIONES")
    print("=" * 70)

    # Cargar resultados de las 3 configuraciones
    configs = {
        lang_config["config_names"]["original"]: base_dir
        / "outputs/classifier_original_sahs/results.json",
        lang_config["config_names"]["normalized"]: base_dir
        / "outputs/classifier_warped_sahs_masked/results.json",
        lang_config["config_names"]["cropped"]: base_dir
        / "outputs/classifier_cropped_12_sahs/results.json",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (config_name, config_path) in enumerate(configs.items()):
        with open(config_path) as f:
            config_results = json.load(f)

        cm_config = np.array(config_results["confusion_matrix"])
        acc_config = config_results["test_metrics"]["accuracy"] * 100
        f1_config = config_results["test_metrics"]["f1_macro"] * 100

        ax = axes[idx]

        # Calcular porcentajes
        cm_percent = cm_config.astype('float') / cm_config.sum(axis=1)[:, np.newaxis] * 100

        # Crear heatmap
        sns.heatmap(
            cm_percent,
            annot=False,
            fmt='.1f',
            cmap='Blues',
            cbar=False,
            ax=ax,
            vmin=0,
            vmax=100
        )

        # Anotar valores
        for i in range(3):
            for j in range(3):
                value = cm_config[i, j]
                percent = cm_percent[i, j]

                if percent > 50:
                    text_color = 'white'
                else:
                    text_color = 'black'

                if i == j:
                    weight = 'bold'
                else:
                    weight = 'normal'

                ax.text(
                    j + 0.5, i + 0.5, f'{value}',
                    ha='center', va='center',
                    color=text_color,
                    fontsize=11,
                    weight=weight
                )

        # Configurar título y etiquetas
        ax.set_title(
            f'{config_name}\nAcc: {acc_config:.2f}% | F1: {f1_config:.2f}%',
            fontsize=13,
            fontweight='bold'
        )

        if idx == 0:
            ax.set_ylabel(
                lang_config["comparison_ylabel"], fontsize=12, fontweight='bold'
            )
            ax.set_yticklabels(display_names, rotation=0)
        else:
            ax.set_yticklabels([])

        ax.set_xlabel(
            lang_config["comparison_xlabel"], fontsize=12, fontweight='bold'
        )
        ax.set_xticklabels(display_names, rotation=45, ha='right')

    plt.suptitle(
        lang_config["comparison_title"],
        fontsize=17,
        fontweight='bold',
        y=1.02
    )
    plt.tight_layout()

    # Guardar comparación
    comparison_path = output_dir / args.comparison_name
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparación guardada: {comparison_path}")
    plt.close()

    print("\n" + "=" * 70)
    print("PROCESO COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
