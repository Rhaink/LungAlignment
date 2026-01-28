# GUI para Demostración de Tesis - Detección de COVID-19

Interfaz gráfica basada en Gradio para demostrar el sistema completo de detección de COVID-19 mediante landmarks anatómicos.

## Características

- **Demostración Completa**: Visualiza las 4 etapas del pipeline
  1. Imagen Original
  2. Landmarks Detectados (15 puntos con colores por grupo)
  3. Imagen Normalizada (Warped)
  4. GradCAM (Regiones de Atención)

- **Vista Rápida**: Clasificación directa sin visualizaciones intermedias

- **Explicabilidad**: GradCAM muestra qué regiones del pulmón atiende el modelo

- **Exportación**: Genera PDF con todas las visualizaciones y métricas

## Requisitos

### Dependencias
```bash
pip install gradio>=4.0.0
```

Todas las demás dependencias ya están en `requirements.txt`.

### Modelos Necesarios
El sistema requiere los siguientes archivos:

1. **Ensemble de Landmarks** (4 modelos):
   - `checkpoints/session10/ensemble/seed123/final_model.pt`
   - `checkpoints/session13/seed321/final_model.pt`
   - `checkpoints/repro_split111/session14/seed111/final_model.pt`
   - `checkpoints/repro_split666/session16/seed666/final_model.pt`

2. **Forma Canónica y Triangulación**:
   - `outputs/shape_analysis/canonical_shape_gpa.json`
   - `outputs/shape_analysis/canonical_delaunay_triangles.json`

3. **Clasificador**:
   - `outputs/classifier_warped_lung_best/sweeps_2026-01-12/lr2e-4_seed321_on/best_classifier.pt`

Ver `docs/REPRO_FULL_PIPELINE.md` para instrucciones de entrenamiento.

## Uso

### Lanzar Interfaz

**Opción 1: Launcher Script** (Recomendado)
```bash
python scripts/run_demo.py
```

Opciones:
- `--share`: Crear enlace público compartible
- `--port PORT`: Cambiar puerto (default: 7860)
- `--host HOST`: Cambiar host (default: localhost)

**Opción 2: Directamente**
```bash
python -m src_v2.gui.app
```

La interfaz se abrirá automáticamente en el navegador en `http://localhost:7860`.

### Uso de la Interfaz

#### Tab 1: Demostración Completa

1. **Cargar imagen**:
   - Arrastra y suelta una radiografía de tórax
   - O haz clic en "Cargar Radiografía de Tórax"
   - O selecciona un ejemplo precargado

2. **Procesar**:
   - Haz clic en "🔍 Procesar Imagen"
   - Espera 1-2 segundos (dependiendo del hardware)

3. **Resultados**:
   - Visualiza las 4 etapas del pipeline
   - Revisa probabilidades de clasificación
   - Expande "Métricas Detalladas" para ver error por landmark

4. **Exportar** (Opcional):
   - Haz clic en "💾 Exportar Resultados a PDF"
   - El PDF se guarda en el directorio actual

#### Tab 2: Vista Rápida

1. Cargar imagen
2. Haz clic en "🚀 Clasificar"
3. Obtén resultado inmediato (sin visualizaciones)

#### Tab 3: Acerca del Sistema

Información sobre metodología, arquitectura y resultados validados.

## Estructura del Código

```
src_v2/gui/
├── __init__.py              # Módulo GUI
├── app.py                   # Interfaz Gradio (3 tabs)
├── config.py                # Configuración (rutas, métricas, colores)
├── gradcam_utils.py         # GradCAM para explicabilidad
├── inference_pipeline.py    # Orquestador del pipeline
├── model_manager.py         # Singleton para gestión de modelos
├── visualizer.py            # Funciones de renderizado
└── README.md               # Este archivo

examples/
├── covid_example.png        # Ejemplo COVID-19
├── normal_example.png       # Ejemplo Normal
└── viral_example.png        # Ejemplo Neumonía Viral

scripts/
└── run_demo.py             # Launcher con verificaciones
```

## Arquitectura

### Patrón Singleton
`ModelManager` usa singleton para cargar modelos una sola vez:
- Lazy loading: modelos se cargan al primer uso
- Cacheo en memoria: no se recargan en cada inferencia
- GPU/CPU detection automática

### Pipeline de Inferencia

```
Usuario carga imagen
    ↓
validate_image() → Verificar formato y tamaño
    ↓
load_and_preprocess() → Cargar y redimensionar a 224×224
    ↓
ModelManager.predict_landmarks()
    ├─ Ensemble de 4 modelos
    ├─ CLAHE (clip=2.0, tile=4)
    ├─ TTA (horizontal flip + swap symmetric pairs)
    └─ Promedio → Landmarks (15, 2)
    ↓
ModelManager.warp_image()
    ├─ Scale landmarks (margin=1.05)
    ├─ Piecewise affine warp
    └─ Imagen normalizada (224, 224)
    ↓
ModelManager.classify_with_gradcam()
    ├─ Clasificación ResNet-18
    ├─ GradCAM en layer4
    └─ Probabilidades + Heatmap
    ↓
Visualizer → Renderizar 4 imágenes
    ↓
Gradio UI → Mostrar al usuario
```

## Métricas Validadas

| Métrica | Valor |
|---------|-------|
| Error de Landmarks | 3.61 ± 2.48 px |
| Accuracy Clasificación | 98.05% |
| F1-Score Macro | 97.12% |
| F1-Score Weighted | 98.04% |

Fuente: `GROUND_TRUTH.json` v2.1.0

## Colores de Landmarks

Los 15 landmarks se visualizan con colores por grupo anatómico:

| Grupo | Landmarks | Color | Descripción |
|-------|-----------|-------|-------------|
| Eje | L1, L2 | Verde | Puntos superior e inferior del eje central |
| Central | L9, L10, L11 | Cyan | Puntos intermedios del eje |
| Lateral | L3-L8 | Amarillo | Contornos laterales izquierdo y derecho |
| Borde | L12, L13 | Magenta | Puntos de borde superior |
| Costal | L14, L15 | Rojo | Puntos costales inferiores |

## Troubleshooting

### Error: "Modelos no encontrados"
- Verifica que los checkpoints existen en las rutas especificadas
- Ejecuta `python scripts/run_demo.py` para diagnóstico automático
- Ver `docs/REPRO_FULL_PIPELINE.md` para entrenar modelos

### Error: "GPU sin memoria suficiente"
- La interfaz automáticamente hará fallback a CPU
- Tiempo de inferencia será ~2-3x más lento
- Para forzar CPU: modifica `DEVICE_PREFERENCE = "cpu"` en `config.py`

### Interfaz no se abre
- Verifica que el puerto 7860 no esté en uso
- Usa `--port 8080` para cambiar puerto
- Revisa firewall si usas `--share`

### Imágenes muy pequeñas
- Mínimo: 100×100 píxeles
- Recomendado: 224×224 o mayor
- La interfaz redimensiona automáticamente

## Notas para Defensa de Tesis

### Puntos Clave
1. **Pipeline Visual**: Usar Tab 1 para mostrar las 4 etapas
2. **Ejemplos Precargados**: Hacer clic en ejemplos para velocidad
3. **Métricas Detalladas**: Expandir accordion para mostrar error por landmark
4. **GradCAM**: Enfatizar que el modelo atiende regiones pulmonares correctas
5. **Comparación**: Mostrar Original vs. Warped lado a lado

### Backup Plan
- Si hay problemas de red: usar screenshots en slides
- Si GPU falla: la interfaz hace fallback a CPU automáticamente
- Si hay errores: usar Tab 2 (Vista Rápida) que es más robusto

## Extensiones Futuras

- [ ] Modo batch para procesar carpetas completas
- [ ] Exportar landmarks como CSV
- [ ] Comparar múltiples imágenes lado a lado
- [ ] Soporte para DICOM (formato médico estándar)
- [ ] API REST para integración con sistemas hospitalarios
- [ ] Visualización de atención por capa (multi-layer GradCAM)

## Referencias

- **Gradio**: https://gradio.app/
- **GradCAM**: Selvaraju et al. (2017) - "Grad-CAM: Visual Explanations from Deep Networks"
- **Dataset**: COVID-19 Radiography Database (Kaggle)

## Contacto

[Agregar información del investigador/tesista]

---

**Versión**: 1.0.0
**Última actualización**: Enero 2026
**Licencia**: [Especificar licencia]
