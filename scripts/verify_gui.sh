#!/bin/bash

echo "=========================================="
echo "Verificación Exhaustiva de la GUI COVID-19"
echo "=========================================="
echo ""

# Colores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Contador de tests
PASSED=0
FAILED=0

run_test() {
    local test_name=$1
    local test_command=$2

    echo -n "🔍 $test_name... "

    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((FAILED++))
    fi
}

# NIVEL 1: Dependencias y Rutas
echo "=== NIVEL 1: Dependencias y Rutas ==="
run_test "Dependencias core" "pytest tests/gui/test_dependencies.py::test_core_dependencies -v"
run_test "Torch CUDA" "pytest tests/gui/test_dependencies.py::test_torch_cuda_availability -v"
run_test "Gradio version" "pytest tests/gui/test_dependencies.py::test_gradio_version -v"
run_test "OpenCV backend" "pytest tests/gui/test_dependencies.py::test_opencv_backend -v"
run_test "Matplotlib backend" "pytest tests/gui/test_dependencies.py::test_matplotlib_non_interactive_backend -v"
run_test "Model paths (dev)" "pytest tests/gui/test_model_paths.py::test_development_mode_paths -v"
run_test "Model sizes" "pytest tests/gui/test_model_paths.py::test_model_file_sizes -v"
run_test "Config metrics" "pytest tests/gui/test_config.py::test_validated_metrics_structure -v"
run_test "Landmark errors" "pytest tests/gui/test_config.py::test_per_landmark_errors -v"
echo ""

# NIVEL 2: Tests Unitarios
echo "=== NIVEL 2: Tests Unitarios ==="
run_test "ModelManager singleton" "pytest tests/gui/test_model_manager.py::test_model_manager_singleton -v"
run_test "Predict landmarks shape" "pytest tests/gui/test_model_manager.py::test_predict_landmarks_shape -v"
run_test "Warp output size" "pytest tests/gui/test_model_manager.py::test_warp_image_output_size -v"
run_test "Validate image formats" "pytest tests/gui/test_inference_pipeline.py::test_validate_image_valid_formats -v"
run_test "Load and preprocess" "pytest tests/gui/test_inference_pipeline.py::test_load_and_preprocess -v"
run_test "Render original" "pytest tests/gui/test_visualizer.py::test_render_original -v"
run_test "Render landmarks" "pytest tests/gui/test_visualizer.py::test_render_landmarks_overlay -v"
run_test "GradCAM init" "pytest tests/gui/test_gradcam_utils.py::test_gradcam_initialization -v"
echo ""

# NIVEL 3: Tests de Integración
echo "=== NIVEL 3: Tests de Integración ==="
run_test "End-to-end pipeline" "pytest tests/gui/test_integration_pipeline.py::test_end_to_end_pipeline -v"
run_test "TTA flip correction" "pytest tests/gui/test_integration_pipeline.py::test_tta_flip_correction -v"
run_test "CLAHE preprocessing" "pytest tests/gui/test_integration_pipeline.py::test_clahe_preprocessing -v"
echo ""

# NIVEL 4: Tests de Sistema
echo "=== NIVEL 4: Tests de Sistema ==="
run_test "Create demo" "pytest tests/gui/test_gradio_interface.py::test_create_demo -v"
run_test "Export PDF" "pytest tests/gui/test_export_pdf.py::test_export_to_pdf -v"
echo ""

# NIVEL 5: Error Handling
echo "=== NIVEL 5: Manejo de Errores ==="
run_test "Invalid image format" "pytest tests/gui/test_error_handling.py::test_invalid_image_format -v"
run_test "Corrupted image" "pytest tests/gui/test_error_handling.py::test_corrupted_image -v"
echo ""

# Resumen
echo ""
echo "=========================================="
echo "RESUMEN DE VERIFICACIÓN"
echo "=========================================="
echo -e "${GREEN}Tests pasados: $PASSED${NC}"
echo -e "${RED}Tests fallidos: $FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ GUI VERIFICADA CORRECTAMENTE${NC}"
    exit 0
else
    echo -e "${RED}✗ GUI TIENE ERRORES - REVISAR${NC}"
    exit 1
fi
