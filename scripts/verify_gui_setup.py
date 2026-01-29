#!/usr/bin/env python3
"""
Script de verificación para la configuración de la GUI.

Verifica que todos los componentes necesarios estén disponibles sin cargar los modelos completos.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_python_version():
    """Check Python version."""
    print("\n1. Verificando versión de Python...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ⚠️  Se requiere Python 3.8 o superior")
        return False
    else:
        print("   ✓ Versión correcta")
        return True


def check_dependencies():
    """Check all required dependencies."""
    print("\n2. Verificando dependencias...")

    deps = {
        'torch': 'PyTorch',
        'gradio': 'Gradio',
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'matplotlib': 'Matplotlib',
        'pandas': 'Pandas',
        'PIL': 'Pillow',
    }

    all_ok = True
    for module, name in deps.items():
        try:
            if module == 'PIL':
                from PIL import Image
            elif module == 'cv2':
                import cv2
            else:
                __import__(module)

            # Get version if possible
            try:
                mod = sys.modules[module if module != 'PIL' else 'PIL']
                version = getattr(mod, '__version__', 'desconocida')
                print(f"   ✓ {name:15s} (v{version})")
            except:
                print(f"   ✓ {name:15s}")
        except ImportError:
            print(f"   ✗ {name:15s} - NO INSTALADO")
            all_ok = False

    return all_ok


def check_modules():
    """Check GUI modules can be imported."""
    print("\n3. Verificando módulos de la GUI...")

    modules = [
        'src_v2.gui.config',
        'src_v2.gui.gradcam_utils',
        'src_v2.gui.visualizer',
        'src_v2.gui.model_manager',
        'src_v2.gui.inference_pipeline',
        'src_v2.gui.app',
    ]

    all_ok = True
    for module_name in modules:
        try:
            __import__(module_name)
            short_name = module_name.split('.')[-1]
            print(f"   ✓ {short_name}")
        except Exception as e:
            short_name = module_name.split('.')[-1]
            print(f"   ✗ {short_name} - {e}")
            all_ok = False

    return all_ok


def check_model_files():
    """Check that required model files exist."""
    print("\n4. Verificando archivos de modelos...")

    from src_v2.gui.config import (
        LANDMARK_MODELS,
        CANONICAL_SHAPE,
        DELAUNAY_TRIANGLES,
        CLASSIFIER_CHECKPOINT
    )

    all_ok = True

    # Landmark models
    print("   Modelos de Landmarks:")
    for i, model_path in enumerate(LANDMARK_MODELS, 1):
        if model_path.exists():
            size_mb = model_path.stat().st_size / 1e6
            print(f"     ✓ Modelo {i}/4 ({size_mb:.1f} MB)")
        else:
            print(f"     ✗ Modelo {i}/4 - NO ENCONTRADO")
            print(f"       {model_path}")
            all_ok = False

    # Canonical shape
    print("   Forma Canónica:")
    if CANONICAL_SHAPE.exists():
        print(f"     ✓ canonical_shape_gpa.json")
    else:
        print(f"     ✗ canonical_shape_gpa.json - NO ENCONTRADO")
        all_ok = False

    # Triangulation
    print("   Triangulación:")
    if DELAUNAY_TRIANGLES.exists():
        print(f"     ✓ canonical_delaunay_triangles.json")
    else:
        print(f"     ✗ canonical_delaunay_triangles.json - NO ENCONTRADO")
        all_ok = False

    # Classifier
    print("   Clasificador:")
    if CLASSIFIER_CHECKPOINT.exists():
        size_mb = CLASSIFIER_CHECKPOINT.stat().st_size / 1e6
        print(f"     ✓ best_classifier.pt ({size_mb:.1f} MB)")
    else:
        print(f"     ✗ best_classifier.pt - NO ENCONTRADO")
        print(f"       {CLASSIFIER_CHECKPOINT}")
        all_ok = False

    return all_ok


def check_examples():
    """Check example images."""
    print("\n5. Verificando imágenes de ejemplo...")

    examples_dir = PROJECT_ROOT / 'examples'

    if not examples_dir.exists():
        print(f"   ✗ Directorio examples/ no encontrado")
        return False

    examples = [
        'covid_example.png',
        'normal_example.png',
        'viral_example.png',
    ]

    all_ok = True
    for example in examples:
        path = examples_dir / example
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"   ✓ {example:25s} ({size_kb:.1f} KB)")
        else:
            print(f"   ✗ {example:25s} - NO ENCONTRADO")
            all_ok = False

    return all_ok


def check_device():
    """Check GPU/CPU availability."""
    print("\n6. Verificando dispositivo de inferencia...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"   ✓ GPU disponible")
            print(f"     Nombre: {torch.cuda.get_device_name(0)}")
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"     Memoria: {total_mem:.1f} GB")
        else:
            print(f"   ⚠️  GPU no disponible - se usará CPU")
            print(f"     La inferencia será más lenta (~2-3x)")

        return True
    except Exception as e:
        print(f"   ✗ Error al verificar dispositivo: {e}")
        return False


def test_clahe():
    """Test CLAHE helper function."""
    print("\n7. Probando función CLAHE...")

    try:
        from src_v2.gui.model_manager import _apply_clahe_numpy
        import numpy as np

        # Create test image
        test_img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)

        # Apply CLAHE
        result = _apply_clahe_numpy(test_img, clip_limit=2.0, tile_size=4)

        if result.shape == (224, 224):
            print(f"   ✓ CLAHE funciona correctamente")
            return True
        else:
            print(f"   ✗ CLAHE retornó forma incorrecta: {result.shape}")
            return False
    except Exception as e:
        print(f"   ✗ Error en CLAHE: {e}")
        return False


def test_interface_creation():
    """Test Gradio interface creation."""
    print("\n8. Probando creación de interfaz Gradio...")

    try:
        from src_v2.gui.app import create_demo

        demo = create_demo()
        print(f"   ✓ Interfaz creada correctamente")
        print(f"     Tipo: {type(demo).__name__}")

        return True
    except Exception as e:
        print(f"   ✗ Error al crear interfaz: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks."""
    print_section("Verificación de Configuración de la GUI")

    checks = [
        ("Python", check_python_version),
        ("Dependencias", check_dependencies),
        ("Módulos GUI", check_modules),
        ("Archivos de Modelos", check_model_files),
        ("Imágenes de Ejemplo", check_examples),
        ("Dispositivo", check_device),
        ("CLAHE", test_clahe),
        ("Interfaz Gradio", test_interface_creation),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n   ✗ Error inesperado en {name}: {e}")
            results[name] = False

    # Summary
    print_section("Resumen")

    passed = sum(results.values())
    total = len(results)

    print(f"\nVerificaciones: {passed}/{total} pasadas\n")

    for name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")

    print()

    if all(results.values()):
        print("✅ Todas las verificaciones pasaron. El sistema está listo.")
        print("\nPara ejecutar la GUI:")
        print("  python scripts/run_demo.py")
        return 0
    else:
        print("⚠️  Algunas verificaciones fallaron.")

        # Specific recommendations
        if not results.get("Dependencias", True):
            print("\n📦 Instala las dependencias faltantes:")
            print("   pip install -r requirements.txt")
            print("   pip install gradio>=4.0.0")

        if not results.get("Archivos de Modelos", True):
            print("\n🤖 Entrena los modelos siguiendo:")
            print("   docs/REPRO_FULL_PIPELINE.md")

        if not results.get("Imágenes de Ejemplo", True):
            print("\n🖼️  Crea imágenes de ejemplo copiando del dataset:")
            print("   Ver: src_v2/gui/README.md")

        return 1


if __name__ == "__main__":
    sys.exit(main())
