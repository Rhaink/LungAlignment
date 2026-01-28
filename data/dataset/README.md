# COVID-19 Radiography Dataset

Este directorio debe contener el dataset de rayos X de Kaggle.

## Instrucciones de Descarga

1. Visitar: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
2. Descargar ZIP (~3.9 GB)
3. Extraer a: `data/dataset/COVID-19_Radiography_Dataset/`

## Estructura Esperada

```
data/
├── coordenadas/
│   └── coordenadas_maestro.csv  (119 KB, incluido en repo)
└── dataset/
    └── COVID-19_Radiography_Dataset/
        ├── COVID/images/*.png (3,616 imágenes)
        ├── Normal/images/*.png (10,192 imágenes)
        └── Viral Pneumonia/images/*.png (1,345 imágenes)
```

## Verificación

```bash
python scripts/analyze_data.py
```
