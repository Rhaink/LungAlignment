"""
Inference pipeline orchestrator for the GUI.

Coordinates the complete workflow:
1. Load and preprocess image
2. Predict landmarks (ensemble + TTA)
3. Apply warping
4. Classify
5. Generate all visualizations
"""
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from .model_manager import get_model_manager
from .visualizer import (
    render_original,
    render_landmarks_overlay,
    render_delaunay_mesh,
    render_warped,
    render_warped_sahs,
    render_comparison_side_by_side,
    create_probability_chart,
    create_metrics_table,
    export_to_pdf,
)
from .config import (
    get_class_name_es,
    ERROR_INVALID_FORMAT,
    ERROR_TOO_SMALL,
    ERROR_MODEL_NOT_FOUND,
    ERROR_GPU_OOM,
    ERROR_WARPING_FAILED,
    SUCCESS_EXPORT,
    EXPORT_FORMAT,
    EXPORT_FILENAME_TEMPLATE,
)


def validate_image(image_path: str) -> tuple[bool, Optional[str]]:
    """
    Validate image format and size.

    Args:
        image_path: Path to image file

    Returns:
        (is_valid, error_message)
    """
    try:
        # Check file exists
        if not Path(image_path).exists():
            return False, "⚠️ Archivo no encontrado."

        # Check format
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        ext = Path(image_path).suffix.lower()
        if ext not in valid_extensions:
            return False, ERROR_INVALID_FORMAT

        # Try to load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return False, "⚠️ No se pudo cargar la imagen."

        # Check minimum size
        if image.shape[0] < 100 or image.shape[1] < 100:
            return False, ERROR_TOO_SMALL

        return True, None

    except Exception as e:
        return False, f"⚠️ Error al validar imagen: {str(e)}"


def load_and_preprocess(image_path: str) -> np.ndarray:
    """
    Load and preprocess image for inference.

    Args:
        image_path: Path to image file

    Returns:
        image: Preprocessed image (224, 224) grayscale
    """
    # Load as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize to 224x224
    if image.shape != (224, 224):
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

    return image


def process_image_full(image_path: str) -> Dict[str, Any]:
    """
    Full pipeline: landmarks → warping → classification → visualizations.

    Args:
        image_path: Path to input image

    Returns:
        Dictionary with all results:
        - 'success': bool
        - 'error': str (if failed)
        - 'original': PIL.Image
        - 'landmarks': PIL.Image (with overlay)
        - 'delaunay_mesh': PIL.Image (Delaunay triangulation overlay)
        - 'warped': PIL.Image
        - 'warped_sahs': PIL.Image
        - 'classification': Dict[str, float] (probabilities in Spanish)
        - 'metrics': pd.DataFrame
        - 'inference_time': float (seconds)
        - 'landmarks_coords': np.ndarray (15, 2)
    """
    start_time = time.time()

    try:
        # Validate image
        is_valid, error_msg = validate_image(image_path)
        if not is_valid:
            return {
                'success': False,
                'error': error_msg,
            }

        # Step 1: Load image
        image = load_and_preprocess(image_path)

        # Step 2: Predict landmarks
        manager = get_model_manager()
        try:
            manager.initialize(verbose=False)
        except FileNotFoundError as e:
            return {
                'success': False,
                'error': f"{ERROR_MODEL_NOT_FOUND}\n{str(e)}",
            }

        try:
            landmarks = manager.predict_landmarks(image, use_tta=True, use_clahe=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return {
                    'success': False,
                    'error': ERROR_GPU_OOM,
                }
            raise

        # Step 3: Apply warping
        try:
            warped = manager.warp_image(image, landmarks)
        except Exception as e:
            # Warping failed, but we can still show landmarks
            warped = image.copy()
            warping_failed = True
            print(f"Warning: Warping failed - {e}")
        else:
            warping_failed = False

        # Step 4: Classify
        probabilities, _, predicted_class_idx = manager.classify_with_gradcam(
            warped,
            target_class=None
        )

        # Get predicted class name in Spanish
        predicted_class_en = manager.class_names[predicted_class_idx]
        predicted_class_es = get_class_name_es(predicted_class_en)

        # Convert probabilities to Spanish labels
        probabilities_es = {
            get_class_name_es(k): v
            for k, v in probabilities.items()
        }

        # Generate visualizations
        img_original = render_original(image)
        img_landmarks = render_landmarks_overlay(image, landmarks, show_labels=True)

        # NEW: Generate Delaunay mesh visualization
        img_delaunay = render_delaunay_mesh(
            image,
            landmarks,
            show_labels=True,
            show_landmark_points=True,
            fill_triangles=False  # Solo bordes de triángulos
        )

        img_warped = render_warped(warped)
        img_sahs = render_warped_sahs(warped, threshold=10)

        # Create metrics table
        metrics_df = create_metrics_table(landmarks)

        # Calculate inference time
        inference_time = time.time() - start_time

        # Build result
        result = {
            'success': True,
            'original': img_original,
            'landmarks': img_landmarks,
            'delaunay_mesh': img_delaunay,
            'warped': img_warped,
            'warped_sahs': img_sahs,
            'classification': probabilities_es,
            'predicted_class': predicted_class_es,
            'metrics': metrics_df,
            'inference_time': inference_time,
            'landmarks_coords': landmarks,
            'warping_failed': warping_failed,
        }

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()

        return {
            'success': False,
            'error': f"❌ Error inesperado: {str(e)}",
        }


def process_image_quick(image_path: str) -> Dict[str, Any]:
    """
    Quick classification mode (skip intermediate visualizations).

    Args:
        image_path: Path to input image

    Returns:
        Dictionary with:
        - 'success': bool
        - 'error': str (if failed)
        - 'classification': Dict[str, float] (probabilities in Spanish)
        - 'predicted_class': str
        - 'inference_time': float
    """
    start_time = time.time()

    try:
        # Validate image
        is_valid, error_msg = validate_image(image_path)
        if not is_valid:
            return {
                'success': False,
                'error': error_msg,
            }

        # Load image
        image = load_and_preprocess(image_path)

        # Get manager
        manager = get_model_manager()
        try:
            manager.initialize(verbose=False)
        except FileNotFoundError as e:
            return {
                'success': False,
                'error': f"{ERROR_MODEL_NOT_FOUND}\n{str(e)}",
            }

        # Predict landmarks
        landmarks = manager.predict_landmarks(image, use_tta=True, use_clahe=True)

        # Warp
        warped = manager.warp_image(image, landmarks)

        # Classify (without GradCAM for speed)
        image_tensor = manager._prepare_image_for_classifier(warped)

        with torch.no_grad():
            logits = manager.classifier(image_tensor)
            probs = logits.softmax(dim=1).detach().cpu().numpy()[0]

        predicted_class_idx = probs.argmax()
        predicted_class_en = manager.class_names[predicted_class_idx]
        predicted_class_es = get_class_name_es(predicted_class_en)

        # Convert to Spanish labels
        probabilities_es = {
            get_class_name_es(manager.class_names[i]): float(probs[i])
            for i in range(len(probs))
        }

        inference_time = time.time() - start_time

        return {
            'success': True,
            'classification': probabilities_es,
            'predicted_class': predicted_class_es,
            'inference_time': inference_time,
        }

    except Exception as e:
        return {
            'success': False,
            'error': f"❌ Error inesperado: {str(e)}",
        }


def export_results(
    result: Dict[str, Any],
    output_dir: Optional[str] = None
) -> tuple[bool, str]:
    """
    Export results to PDF.

    Args:
        result: Result dictionary from process_image_full()
        output_dir: Output directory (default: current directory)

    Returns:
        (success, message or error)
    """
    try:
        if not result.get('success', False):
            return False, "⚠️ No hay resultados válidos para exportar."

        # Determine output path
        if output_dir is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = EXPORT_FILENAME_TEMPLATE.format(
            timestamp=timestamp,
            ext=EXPORT_FORMAT
        )
        output_path = output_dir / filename

        # Prepare images dict
        images = {
            'original': result['original'],
            'landmarks': result['landmarks'],
            'delaunay_mesh': result['delaunay_mesh'],
            'warped': result['warped'],
            'warped_sahs': result['warped_sahs'],
        }

        # Metadata
        metadata = {
            'Tiempo de Inferencia': f"{result['inference_time']:.2f} segundos",
            'Clase Predicha': result['predicted_class'],
            'Probabilidad': f"{result['classification'][result['predicted_class']] * 100:.2f}%",
        }

        # Export to PDF
        export_to_pdf(
            images=images,
            metrics_df=result['metrics'],
            output_path=str(output_path),
            metadata=metadata
        )

        return True, f"{SUCCESS_EXPORT}\nArchivo: {output_path}"

    except Exception as e:
        return False, f"❌ Error al exportar: {str(e)}"


def create_comparison_visualization(
    original: np.ndarray,
    warped: np.ndarray
) -> Image.Image:
    """
    Create side-by-side comparison.

    Args:
        original: Original image (224, 224)
        warped: Warped image (224, 224)

    Returns:
        PIL Image with comparison
    """
    return render_comparison_side_by_side(original, warped)
