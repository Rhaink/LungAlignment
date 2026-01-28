# Notas de Migración: LungAlignment

**Fecha:** 2026-01-28
**Origen:** `/home/donrobot/Projects/prediccion_warping_clasificacion`
**Destino:** `/home/donrobot/Projects/LungAlignment`

## Cambios Principales

### Reducción de Tamaño
- **Antes:** ~42 GB (outputs, venvs, build artifacts, tesis)
- **Después:** ~300 MB (código + checkpoints esenciales)
- **Ahorro:** 41.7 GB

### Componentes Migrados

**Código fuente (2.1 MB):**
- `src_v2/` completo (43 archivos Python)
- 11 configuraciones JSON
- 17 scripts activos (excluidos: archive/, visualization/, fisher/)

**Modelos (184 MB):**
- 4 checkpoints del ensemble 3.61 px (seeds 123, 321, 111, 666)
- Estructura aplanada: `checkpoints/ensemble_seed123/final_model.pt`

**Datos críticos:**
- `data/coordenadas/coordenadas_maestro.csv` (119 KB)
- `GROUND_TRUTH.json` (valores validados v2.1.0)

### Resultados Validados Preservados

- **Landmarks:** 3.61 px error (ensemble 4 modelos + TTA + CLAHE)
- **Clasificación:** 98.05% accuracy (warped_lung_best, fill_rate 47%)
- **Hiperparámetros:** margin=1.05, CLAHE tile_size=4

### Excluido de la Migración

**Artifacts regenerables (37.8 GB):**
- `outputs/` - Datasets warpeados, predicciones cacheadas
- `.venv/`, `.venv_build/` - Entornos virtuales
- `build/`, `dist/` - Compilaciones
- `results/` - Figuras experimentales
- `data/dataset/` - Imágenes (re-descargables de Kaggle)

**Documentación de tesis (>500 MB):**
- `docs/Tesis/` - LaTeX, PDFs
- `docs/articulo/` - Submissions
- `docs/manual/`, `docs/estancia/`, `docs/carta/`

**Código legacy:**
- `scripts/archive/` - Experimentos antiguos
- `scripts/visualization/` - Figuras específicas de tesis
- `scripts/fisher/` - Enfoque abandonado

### Paths Actualizados

1. **ensemble_best.json:** Rutas de checkpoints aplanadas
2. **pyproject.toml:** Package renombrado a `lung_alignment`
3. **CLAUDE.md:** Todas las rutas apuntan a nuevo proyecto

### Reestructuración de Checkpoints

```
ANTES: checkpoints/session10/ensemble/seed123/final_model.pt
AHORA: checkpoints/ensemble_seed123/final_model.pt
```

Razón: Jerarquía de sesiones es contexto de investigación, nombres descriptivos son mejores para producción.

### Post-Migración Setup

```bash
cd /home/donrobot/Projects/LungAlignment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Verificación de Reproducibilidad

```bash
# GPA (no requiere dataset)
python -m src_v2 compute-canonical data/coordenadas/coordenadas_maestro.csv \
  --output-dir outputs/shape_analysis --visualize

# Ensemble (requiere dataset)
python scripts/evaluate_ensemble_from_config.py --config configs/ensemble_best.json
# Esperado: ~3.61 px
```

### Dataset COVID-19 Radiography

NO incluido (3.9 GB). Descargar de:
- URL: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
- Extraer a: `data/dataset/COVID-19_Radiography_Dataset/`

### Repositorio Original

- GitHub: https://github.com/Rhaink/prediccion_warping_clasificacion
- Commit final: `12346f98 - docs(03): complete TTA Integration phase`
- Historia completa disponible para referencia
