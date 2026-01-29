# LungAlignment

Este repositorio implementa un flujo de trabajo completo para:
1) detectar puntos de referencia pulmonares,
2) normalizar la geometria por warping,
3) entrenar un clasificador CNN sobre imagenes warpeadas.

El estado actual reproduce el ensemble de puntos de referencia **3.61 px** y un
clasificador en warped con **~98% accuracy** (ver `docs/EXPERIMENTS.md`).

## Estructura rapida del repo
- `src_v2/`: CLI principal y modulos (data, models, training, processing).
- `src_v2/gui/`: **Interfaz gráfica Gradio para demostración** (nuevo).
- `configs/`: JSON con defaults (landmarks, ensemble, warping, classifier).
- `scripts/`: automatizaciones y wrappers.
- `docs/`: guias de reproduccion y reportes.
- `data/`, `checkpoints/`, `outputs/`: artefactos locales (no se suben a Git).

## Requisitos locales
- Python 3.9+
- Datos y checkpoints necesarios (provistos por el autor).
- Los comandos asumen la estructura de carpetas del repo (`data/`, `checkpoints/`, `outputs/`).

Instalacion rapida:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Si necesitas una build especifica de PyTorch, ajusta segun tu hardware.

## Flujo de trabajo actual (resumen)
1) Forma estándar (GPA):
```bash
python -m src_v2 compute-canonical data/coordenadas/coordenadas_maestro.csv \
  --output-dir outputs/shape_analysis --visualize
```

2) Predicciones de puntos de referencia (cache completo del dataset):
```bash
python scripts/predict_landmarks_dataset.py \
  --input-dir data/dataset/COVID-19_Radiography_Dataset \
  --output outputs/landmark_predictions/session_warping/predictions.npz \
  --ensemble-config configs/ensemble_best.json \
  --tta --clahe --clahe-clip 2.0 --clahe-tile 4
```

3) Warping del dataset (usa cache, no re-inferencia):
```bash
python -m src_v2 generate-dataset --config configs/warping_best.json
```

4) Entrenar clasificador en warped:
```bash
python -m src_v2 train-classifier --config configs/classifier_warped_base.json
```

5) Evaluar clasificador:
```bash
python -m src_v2 evaluate-classifier \
  outputs/classifier_warped_lung_best/best_classifier.pt \
  --data-dir outputs/warped_lung_best/session_warping --split test
```

## Interfaz Gráfica (Demo)
Lanzar interfaz web Gradio para demostración interactiva:
```bash
pip install gradio>=4.0.0
python scripts/run_demo.py
```
La interfaz se abrirá en `http://localhost:7860` con:
- Visualización completa del flujo de trabajo (4 etapas)
- GradCAM para explicabilidad
- Métricas validadas en pantalla
- Exportación a PDF

Ver documentación completa: `src_v2/gui/README.md`

## Documentos clave
- **GUI de demostración**: `src_v2/gui/README.md` (nuevo)
- Flujo completo (detalle): `docs/REPRO_FULL_PIPELINE.md`
- Ensemble de puntos de referencia 3.61: `docs/REPRO_ENSEMBLE_3_71.md`
- Warping quickstart: `docs/QUICKSTART_WARPING.md`
- Clasificador warped: `docs/REPRO_CLASSIFIER_RESNET18.md`

## Nota sobre artefactos
`data/`, `checkpoints/` y `outputs/` son locales y no se versionan. Los datos y
checkpoints necesarios para reproducir el flujo de trabajo son provistos por el autor.
