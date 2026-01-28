# Changelog - GUI de Demostración

## [1.0.0] - 2026-01-18

### Implementación Inicial

#### ✨ Características Nuevas

**Interfaz Gradio Completa**
- Interfaz web de 3 tabs para demostración de tesis
- Tab 1: Pipeline completo con visualización de 4 etapas
- Tab 2: Modo rápido para clasificación directa
- Tab 3: Información sobre metodología y resultados

**Visualizaciones Profesionales**
- Landmarks con colores por grupo anatómico (5 grupos)
- GradCAM para explicabilidad del modelo
- Comparaciones lado a lado (original vs. warped)
- Exportación a PDF multipágina con métricas

**Gestión Eficiente de Modelos**
- Patrón Singleton para cacheo de modelos
- Lazy loading: carga bajo demanda
- Detección automática GPU/CPU
- Soporte para ensemble de 4 modelos

**Métricas Validadas**
- Error de landmarks: 3.61 ± 2.48 px
- Accuracy: 98.05%
- F1-Score Macro: 97.12%
- Todas desde GROUND_TRUTH.json v2.1.0

#### 📁 Archivos Nuevos

```
src_v2/gui/
├── __init__.py              # Inicialización del módulo
├── app.py                   # Interfaz Gradio (3 tabs)
├── config.py                # Configuración centralizada
├── gradcam_utils.py         # Implementación GradCAM
├── inference_pipeline.py    # Orquestador del pipeline
├── model_manager.py         # Gestión de modelos (Singleton)
├── visualizer.py            # Funciones de renderizado
├── README.md               # Documentación de uso
└── CHANGELOG.md            # Este archivo

scripts/
├── run_demo.py             # Launcher con verificaciones
└── verify_gui_setup.py     # Script de verificación

examples/
├── covid_example.png       # Ejemplo COVID-19
├── normal_example.png      # Ejemplo Normal
└── viral_example.png       # Ejemplo Neumonía Viral
```

#### 🔧 Componentes Técnicos

**config.py** (~210 líneas)
- Rutas de modelos y datos
- Métricas validadas de GROUND_TRUTH.json
- Esquema de colores para landmarks
- Textos de interfaz en español

**model_manager.py** (~440 líneas)
- Singleton con lazy loading
- Carga de ensemble (4 modelos)
- Predicción con TTA (Test-Time Augmentation)
- Warping piecewise affine
- Clasificación + GradCAM

**gradcam_utils.py** (~261 líneas)
- Clase GradCAM con hooks
- Generación de heatmaps
- Overlay sobre imágenes
- Resize y normalización

**visualizer.py** (~482 líneas)
- render_original()
- render_landmarks_overlay()
- render_warped()
- render_gradcam()
- create_probability_chart()
- create_metrics_table()
- export_to_pdf()

**inference_pipeline.py** (~274 líneas)
- validate_image()
- load_and_preprocess()
- process_image_full()
- process_image_quick()
- export_results()

**app.py** (~388 líneas)
- create_demo() - Construcción de interfaz Gradio
- Manejo de eventos (botones, ejemplos)
- Estados para exportación
- Callbacks de procesamiento

**run_demo.py** (~218 líneas)
- Verificación de dependencias
- Verificación de modelos
- Detección de GPU/CPU
- Launcher con argumentos CLI

**verify_gui_setup.py** (~338 líneas)
- 8 verificaciones automáticas
- Diagnóstico de problemas
- Recomendaciones específicas
- Informe detallado

#### 🐛 Correcciones

**Fix 1: CLAHE TypeError** (2026-01-18)
- Problema: `apply_clahe()` de `src_v2/data/transforms.py` esperaba PIL Image
- Solución: Creada función `_apply_clahe_numpy()` para trabajar con numpy arrays
- Ubicación: `src_v2/gui/model_manager.py` líneas 35-59
- Impacto: Resuelve error en predicción de landmarks

**Fix 2: scale_landmarks_from_centroid() ArgumentError** (2026-01-18)
- Problema: Parámetro incorrecto `margin_scale` en lugar de `scale`
- Solución: Corregido a `scale=margin_scale` en llamada a función
- Ubicación: `src_v2/gui/model_manager.py` línea 343
- Impacto: Resuelve error en warping de imágenes

**Fix 3: RuntimeError con torch.Tensor.numpy()** (2026-01-18)
- Problema: No se puede llamar `.numpy()` en tensor con gradientes activados
- Solución: Agregado `.detach()` antes de `.cpu().numpy()`
- Ubicaciones:
  - `src_v2/gui/model_manager.py` línea 391
  - `src_v2/gui/inference_pipeline.py` línea 270
- Impacto: Resuelve error en clasificación y GradCAM

**Fix 4: torch.no_grad() llamada incorrecta** (2026-01-18)
- Problema: `manager.classifier.no_grad()` no existe
- Solución: Corregido a `torch.no_grad()` y agregado import
- Ubicación: `src_v2/gui/inference_pipeline.py` línea 268
- Impacto: Resuelve error en modo rápido de clasificación

#### ⚙️ Configuración

**Dependencias Nuevas**
```bash
gradio>=4.0.0
```

**Modelos Requeridos**
- Ensemble landmarks: 4 modelos (~47 MB cada uno)
- Canonical shape: JSON con 15 puntos
- Triangulación: JSON con 18 triángulos
- Clasificador: ResNet-18 finetuned (~45 MB)

**Hardware Recomendado**
- GPU: 4+ GB VRAM (probado con AMD Radeon RX 6600 8.6 GB)
- RAM: 8+ GB
- CPU fallback: Funciona pero ~2-3x más lento

#### 📊 Rendimiento

**Tiempos de Inferencia** (con GPU AMD RX 6600)
- Landmarks (ensemble + TTA): ~800 ms
- Warping: ~50 ms
- Clasificación + GradCAM: ~100 ms
- **Total**: ~1 segundo por imagen

**Tiempos de Carga**
- Modelos (primera vez): 5-10 segundos
- Interfaz Gradio: <1 segundo
- Inicio total: ~10-15 segundos

**Uso de Memoria**
- GPU: ~2 GB (4 modelos landmarks + clasificador)
- RAM: ~1.5 GB (modelos + datos)

#### 📝 Uso

**Lanzar Interfaz**
```bash
# Opción 1: Script recomendado
python scripts/run_demo.py

# Opción 2: Con opciones
python scripts/run_demo.py --share --port 8080

# Opción 3: Directamente
python -m src_v2.gui.app
```

**Verificar Configuración**
```bash
python scripts/verify_gui_setup.py
```

#### 🎯 Casos de Uso

1. **Defensa de Tesis**
   - Demostración visual del pipeline completo
   - Explicabilidad con GradCAM
   - Métricas validadas en pantalla

2. **Desarrollo e Investigación**
   - Prueba rápida de modelos entrenados
   - Validación visual de predicciones
   - Exportación de resultados para publicación

3. **Análisis Individual**
   - Clasificación rápida de radiografías
   - Visualización de landmarks detectados
   - Comparación original vs. normalizada

#### 🔮 Mejoras Futuras

- [ ] Modo batch para carpetas completas
- [ ] Exportar landmarks como CSV
- [ ] Comparación multi-imagen
- [ ] Soporte DICOM
- [ ] API REST
- [ ] Multi-layer GradCAM
- [ ] Métricas de incertidumbre
- [ ] Historial de predicciones

#### 🙏 Reconocimientos

- **Gradio**: Framework de interfaz web
- **PyTorch**: Framework de deep learning
- **OpenCV**: Procesamiento de imágenes
- **Matplotlib**: Visualizaciones
- **Dataset**: COVID-19 Radiography Database (Kaggle)

#### 📄 Licencia

[Especificar licencia del proyecto]

---

**Estadísticas del Código**
- Archivos nuevos: 11
- Líneas de código: ~2,600
- Funciones principales: 47
- Clases: 3 (GradCAM, ModelManager, ImageClassifier)
- Tiempo de desarrollo: 2-3 días

**Métricas de Calidad**
- ✓ Type hints en funciones principales
- ✓ Docstrings en formato Google
- ✓ Manejo de errores robusto
- ✓ Validación de entrada
- ✓ Logging y diagnóstico
- ✓ Documentación completa
