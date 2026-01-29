# Scripts Migration Notes

**Fecha:** 2026-01-29
**Proyecto:** LungAlignment (migrado de prediccion_warping_clasificacion)

## Scripts Migrados (20 scripts)

### Core Pipeline Scripts
1. `analyze_data.py` - Analizar estructura del dataset
2. `predict_landmarks_dataset.py` - Generar predicciones cacheadas
3. `evaluate_ensemble_from_config.py` - Evaluar ensemble de landmarks
4. `quickstart_landmarks.sh` - Pipeline rápido de landmarks
5. `quickstart_warping.sh` - Pipeline rápido de warping

### Evaluation & Metrics
6. `compute_cv_test_aggregated_metrics.py` - Métricas agregadas de CV
7. `generate_confusion_matrix_cv.py` - Matriz de confusión (CV)
8. `generate_confusion_matrix_sahs.py` - Matriz de confusión (SAHS)
9. `extract_predictions.py` - Extraer predicciones de modelos
10. `extract_dataset_splits.py` - Extraer info de splits

### Visualization & Figures
11. `generate_F5_8_comparison_improved_v2.py` - Figura de comparación mejorada
12. `generate_all_landmarks_npz.py` - Generar NPZ de landmarks
13. `generate_cropped_sahs_dataset.py` - Dataset SAHS recortado
14. `create_reference_image.py` - Crear imagen de referencia
15. `verify_landmark_viz_dataset.py` - Verificar dataset de visualización

### Performance & Benchmarking
16. `benchmark_inference.py` - Benchmark de inferencia
17. `run_benchmark.sh` - Script de benchmark

### GUI & Demo (Agregados 2026-01-29)
18. `run_demo.py` - Lanzar interfaz Gradio (CRÍTICO - mencionado en README)
19. `verify_gui_setup.py` - Verificar setup de GUI
20. `verify_gui.sh` - Script de verificación de GUI

---

## Scripts NO Migrados (Disponibles en Original)

### Build & Deployment (7 scripts)
- `build_portable_windows.py` - Build ejecutable portable Windows
- `build_windows_exe.py` - Build EXE para Windows
- `build_release.sh` - Script de release
- `prepare_models_for_build.py` - Preparar modelos para build
- `generate_icon.py` - Generar icono de aplicación
- `test_exe_startup.py` - Test de inicio de ejecutable
- `covid_demo.spec` - Spec de PyInstaller

**Razón de exclusión:** Scripts de build/deployment específicos para distribución Windows. No son necesarios para reproducción científica.

### Thesis-Specific Figures (20+ scripts)
- `generate_F2_clahe_vs_sahs.py`
- `generate_F5_3_scientific.py`
- `generate_F5_3_single_panel_fixed.py`
- `generate_F5_3_single_panel.py`
- `generate_F5_6_warping_sahs.py`
- `generate_F5_8_comparison_cv.py`
- `generate_F5_8_comparison_improved.py`
- `generate_F5_9_misclassified_cv.py`
- `generate_F5_9_misclassified_en.py`
- `generate_F5_9_misclassified.py`
- `generate_sahs_comparison_figure.py`
- `generate_thesis_figure.py`
- `generate_thesis_figures_master.py`
- `create_thesis_figures.py`
- `generate_cv_figures_master.py`
- `update_all_figures.py`
- `visualize_gpa_methodology_fixed.py`
- `visualize_gpa_methodology.py`

**Razón de exclusión:** Scripts específicos para generar figuras de tesis/artículo. Generan archivos en `docs/Tesis/Figures/` que no están en la migración. Pueden regenerarse del proyecto original si es necesario.

### Verification Scripts (9 scripts)
- `verify_canonical_delaunay.py` - Verificar triangulación Delaunay
- `verify_comparison_alignment.py` - Verificar alineación en comparaciones
- `verify_data_leakage.py` - Verificar fugas de datos
- `verify_dataset_splits.py` - Verificar splits del dataset
- `verify_gpa_correctness.py` - Verificar correctitud de GPA
- `verify_individual_models.py` - Verificar modelos individuales
- `verify_no_tta.py` - Verificar ejecución sin TTA
- `verify_val_vs_test.py` - Comparar val vs test

**Razón de exclusión:** Scripts de verificación para debugging durante desarrollo. Útiles pero no críticos para reproducción. Pueden agregarse si se necesita debugging profundo.

### Legacy/Alternative Approaches (7 scripts)
- `generate_full_warped_dataset.py` - Enfoque antiguo de warping (Session 25)
- `generate_warped_dataset.py` - Enfoque antiguo (Session 21)
- `generate_warped_sahs_dataset.py` - Warping específico de SAHS
- `train.py` - Entrenador antiguo de landmarks
- `train_classifier.py` - Wrapper antiguo de clasificador
- `train_hierarchical.py` - Entrenador de modelo jerárquico
- `predict.py` - Wrapper antiguo de predicción

**Razón de exclusión:** Enfoques legacy reemplazados por el CLI principal (`python -m src_v2`). Mantenidos en original por referencia histórica.

### Sweeps & Experiments (4 scripts)
- `run_classifier_sweep_accuracy.sh` - Sweep de accuracy de clasificador
- `run_seed_sweep.sh` - Sweep de seeds
- `sweep_ensemble_combos.py` - Sweep de combinaciones de ensemble
- `run_best_ensemble.sh` - Ejecutar mejor ensemble

**Razón de exclusión:** Scripts para experimentación/tuning. Resultados ya validados en `GROUND_TRUTH.json`.

### Analysis & Exploration (5 scripts)
- `analyze_hospital_marks.py` - Análisis de marcas hospitalarias
- `calculate_pfs_warped.py` - Calcular PFS en imágenes warpeadas
- `gpa_analysis.py` - Análisis de GPA
- `landmark_connections.py` - Conexiones de landmarks
- `visualize_predictions.py` - Visualizar predicciones

**Razón de exclusión:** Scripts de análisis exploratorio. Útiles para análisis ad-hoc pero no críticos para pipeline principal.

### Utilities (2 scripts)
- `piecewise_affine_warp.py` - Implementación standalone de warping
- `cleanup_checkpoints.sh` - Limpieza de checkpoints

**Razón de exclusión:** `piecewise_affine_warp.py` está integrado en `src_v2/processing/warp.py`. `cleanup_checkpoints.sh` fue usado en el proyecto original (2026-01-20) y ya no es necesario.

---

## Recuperación de Scripts No Migrados

Si necesitas algún script que no se migró:

```bash
# Proyecto original
ORIGINAL="/home/donrobot/Projects/prediccion_warping_clasificacion"

# Copiar script específico
cp "$ORIGINAL/scripts/nombre_script.py" /home/donrobot/Projects/LungAlignment/scripts/

# O copiar múltiples
cp "$ORIGINAL/scripts/{script1.py,script2.py,script3.sh}" /home/donrobot/Projects/LungAlignment/scripts/
```

**Ejemplos comunes:**

```bash
# Verificación profunda
cp "$ORIGINAL/scripts/verify_gpa_correctness.py" scripts/
cp "$ORIGINAL/scripts/verify_data_leakage.py" scripts/

# Figuras de tesis
cp "$ORIGINAL/scripts/generate_F5_3_scientific.py" scripts/

# Build de ejecutable
cp "$ORIGINAL/scripts/build_windows_exe.py" scripts/
cp "$ORIGINAL/scripts/prepare_models_for_build.py" scripts/
```

---

## Directorio `scripts/` en Original (No Migrados)

Además de los scripts de nivel superior, el proyecto original tiene subdirectorios:

- `scripts/archive/` - Experimentos antiguos (Session 1-9)
- `scripts/visualization/` - Figuras y visualizaciones específicas
- `scripts/fisher/` - Enfoque Fisher abandonado

**Total en original:** 71 scripts (top-level) + subdirectorios

**Razón de exclusión:** Contexto histórico/experimental. No necesarios para reproducción de resultados validados.

---

## Scripts Agregados Post-Migración

### 2026-01-29: GUI Scripts
- `run_demo.py` - Agregado porque README lo menciona
- `verify_gui_setup.py` - Para verificar setup de GUI
- `verify_gui.sh` - Script de verificación

**Razón:** Necesarios para que la GUI funcione según documentación del README.

---

## Recomendaciones

### Scripts Esenciales (Ya Migrados)
Estos 20 scripts son suficientes para:
- ✅ Reproducir pipeline completo (landmarks → warping → clasificación)
- ✅ Evaluar resultados validados (3.61 px, 98.05% accuracy)
- ✅ Ejecutar GUI de demostración
- ✅ Generar benchmarks y métricas

### Si Necesitas Más
**Agregar si vas a:**
- **Debugging profundo:** Scripts de `verify_*`
- **Generar figuras de paper:** Scripts de `generate_F*`
- **Build ejecutable Windows:** Scripts de `build_*`
- **Experimentación nueva:** Scripts de `sweep_*` y `run_*_sweep.sh`

### No Agregar
**Mantener fuera (legacy):**
- Scripts de `archive/`, `visualization/`, `fisher/`
- Wrappers antiguos (`train.py`, `predict.py`, `train_classifier.py`)
- Scripts de análisis exploratorio ad-hoc

---

## Conteo Final

| Categoría | Migrados | Disponibles en Original | Total Original |
|-----------|----------|-------------------------|----------------|
| Core Pipeline | 5 | 7 | 12 |
| Evaluation | 5 | 0 | 5 |
| Visualization | 5 | 20+ | 25+ |
| Performance | 2 | 0 | 2 |
| GUI & Demo | 3 | 0 | 3 |
| Build | 0 | 7 | 7 |
| Verification | 1 | 9 | 10 |
| Legacy | 0 | 7 | 7 |
| Sweeps | 0 | 4 | 4 |
| Analysis | 0 | 5 | 5 |
| **TOTAL** | **20** | **~50** | **~70** |

---

## Historial de Cambios

- **2026-01-28:** Migración inicial (17 scripts)
- **2026-01-29:** Agregados 3 scripts de GUI (`run_demo.py`, `verify_gui*.py/sh`)

---

**Nota:** Este documento se actualizará si se agregan más scripts durante la fase de validación o desarrollo futuro.
