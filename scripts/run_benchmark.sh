#!/bin/bash
#
# Script auxiliar para ejecutar benchmark de inferencia
#

set -e

echo "🚀 Ejecutando Benchmark de Inferencia..."
echo ""

# Crear directorio temporal con imágenes de muestra
TEMP_DIR=$(mktemp -d)
echo "📁 Directorio temporal: $TEMP_DIR"

# Copiar imágenes de muestra de cada clase
echo "📋 Recopilando imágenes de muestra..."

# COVID (30 imágenes)
mkdir -p "$TEMP_DIR/COVID"
find data/dataset/COVID-19_Radiography_Dataset/COVID/images -name "*.png" | head -30 | while read img; do
    cp "$img" "$TEMP_DIR/COVID/"
done

# Normal (40 imágenes)
mkdir -p "$TEMP_DIR/Normal"
find data/dataset/COVID-19_Radiography_Dataset/Normal/images -name "*.png" | head -40 | while read img; do
    cp "$img" "$TEMP_DIR/Normal/"
done

# Viral Pneumonia (30 imágenes)
mkdir -p "$TEMP_DIR/Viral_Pneumonia"
find "data/dataset/COVID-19_Radiography_Dataset/Viral Pneumonia/images" -name "*.png" | head -30 | while read img; do
    cp "$img" "$TEMP_DIR/Viral_Pneumonia/"
done

TOTAL=$(find "$TEMP_DIR" -name "*.png" | wc -l)
echo "✓ Imágenes recopiladas: $TOTAL"
echo ""

# Ejecutar benchmark
python scripts/benchmark_inference.py \
    --sample-dir "$TEMP_DIR" \
    --num-samples $TOTAL \
    --ensemble-config configs/ensemble_best.json \
    --classifier-path outputs/classifier_cropped_10/best_classifier.pt \
    --gpa-output-dir outputs/shape_analysis \
    --output-json outputs/benchmark_results.json \
    --warmup 5 \
    --device cuda

# Limpiar
rm -rf "$TEMP_DIR"
echo ""
echo "✓ Benchmark completado!"
echo "📄 Resultados guardados en: outputs/benchmark_results.json"
