#!/usr/bin/env python3
"""
Launcher for the COVID-19 detection GUI demonstration.

Usage:
    python scripts/run_demo.py [options]

Options:
    --share         Create public shareable link (via Gradio)
    --port PORT     Server port (default: 7860)
    --host HOST     Server host (default: localhost)

This script supports both development mode and PyInstaller standalone mode.
"""
import os
import sys
import argparse
from pathlib import Path


def get_base_path():
    """
    Get base path for resources.

    Returns the appropriate base directory depending on execution mode:
    - PyInstaller frozen mode: sys._MEIPASS (temporary extraction directory)
    - Development mode: Project root directory

    Returns:
        Path: Base path for accessing models and resources
    """
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller executable
        return Path(sys._MEIPASS)
    else:
        # Running as Python script
        return Path(__file__).parent.parent


# Get base path (works in both .exe and development mode)
BASE_PATH = get_base_path()
PROJECT_ROOT = BASE_PATH

# Add project root to path
sys.path.insert(0, str(PROJECT_ROOT))

# Configure models directory for deployment mode
if getattr(sys, 'frozen', False):
    # In PyInstaller mode, models are in 'models/' relative to _MEIPASS
    os.environ['COVID_DEMO_MODELS_DIR'] = str(BASE_PATH / 'models')
    os.environ['COVID_DEMO_FROZEN'] = '1'  # Flag to detect frozen mode


def check_dependencies():
    """Check that all required dependencies are installed."""
    missing = []

    try:
        import torch
    except ImportError:
        missing.append('torch')

    try:
        import gradio
    except ImportError:
        missing.append('gradio')

    try:
        import numpy
    except ImportError:
        missing.append('numpy')

    try:
        import cv2
    except ImportError:
        missing.append('opencv-python')

    try:
        import matplotlib
    except ImportError:
        missing.append('matplotlib')

    try:
        import pandas
    except ImportError:
        missing.append('pandas')

    try:
        from PIL import Image
    except ImportError:
        missing.append('pillow')

    if missing:
        print("❌ Error: Dependencias faltantes:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nEjecuta: pip install " + " ".join(missing))
        return False

    return True


def check_models():
    """Check that required model files exist."""
    from src_v2.gui.config import LANDMARK_MODELS, CANONICAL_SHAPE, DELAUNAY_TRIANGLES, CLASSIFIER_CHECKPOINT

    missing = []

    # Check landmark models
    for model_path in LANDMARK_MODELS:
        if not model_path.exists():
            missing.append(str(model_path))

    # Check canonical shape
    if not CANONICAL_SHAPE.exists():
        missing.append(str(CANONICAL_SHAPE))

    # Check triangulation
    if not DELAUNAY_TRIANGLES.exists():
        missing.append(str(DELAUNAY_TRIANGLES))

    # Check classifier
    if not CLASSIFIER_CHECKPOINT.exists():
        missing.append(str(CLASSIFIER_CHECKPOINT))

    if missing:
        print("⚠️  Advertencia: Archivos de modelo faltantes:")
        for path in missing:
            print(f"  - {path}")
        print("\nLa interfaz puede fallar al cargar. Asegúrate de haber entrenado los modelos.")
        print("Ver: docs/REPRO_FULL_PIPELINE.md")
        return False

    return True


def is_frozen():
    """Check if running as PyInstaller frozen executable."""
    return getattr(sys, 'frozen', False)


def print_header():
    """Print welcome header."""
    print("=" * 70)
    print("  Sistema de Detección de COVID-19 mediante Landmarks Anatómicos")
    if is_frozen():
        print("  COVID-19 Detection System - Standalone Demo")
    print("=" * 70)
    print()


def print_device_info():
    """Print device and GPU information."""
    try:
        import torch

        print("Información del Dispositivo:")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Dispositivo: {device.upper()}")

        if device == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  Memoria GPU: {total_memory:.1f} GB")
        else:
            print("  ⚠️  GPU no disponible. Usando CPU (puede ser más lento)")

        print()

    except Exception as e:
        print(f"⚠️  Error al detectar dispositivo: {e}\n")


def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Lanzador de interfaz GUI para detección de COVID-19")
    parser.add_argument('--share', action='store_true', help='Crear enlace público compartible')
    parser.add_argument('--port', type=int, default=7860, help='Puerto del servidor (default: 7860)')
    parser.add_argument('--host', type=str, default='localhost', help='Host del servidor (default: localhost)')
    args = parser.parse_args()

    # Print header
    print_header()

    # Check dependencies
    print("Verificando dependencias...")
    if not check_dependencies():
        sys.exit(1)
    print("✓ Todas las dependencias instaladas\n")

    # Check device
    print_device_info()

    # Check models
    print("Verificando archivos de modelos...")
    if is_frozen():
        print("(Esto puede tomar 10-30 segundos en la primera ejecución...)\n")

    models_ok = check_models()
    if models_ok:
        print("✓ Todos los modelos encontrados\n")
    else:
        print()
        if is_frozen():
            # In frozen mode, cannot continue without models
            print("❌ Error crítico: Modelos no encontrados.")
            print("Por favor contacte al desarrollador o reinstale la aplicación.")
            input("\nPresione Enter para salir...")
            sys.exit(1)
        else:
            # In development mode, ask user
            response = input("¿Deseas continuar de todos modos? (s/n): ")
            if response.lower() not in ['s', 'si', 'y', 'yes']:
                print("Abortando.")
                sys.exit(1)
        print()

    # Import and create demo
    print("Inicializando interfaz Gradio...")
    try:
        from src_v2.gui.app import create_demo
        demo = create_demo()
    except Exception as e:
        print(f"❌ Error al crear interfaz: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("✓ Interfaz creada\n")

    # Print launch info
    print("=" * 70)
    print("✓ Servidor listo. Abriendo navegador...")
    print("=" * 70)
    print()
    print(f"URL Local: http://{args.host}:{args.port}")

    if args.share:
        print("Generando enlace público (puede tardar unos segundos)...")

    print()
    print("Presiona Ctrl+C para detener el servidor")
    print()

    # Launch
    try:
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            inbrowser=True,  # Open browser automatically
            quiet=False,
        )
    except KeyboardInterrupt:
        print("\n\n✓ Servidor detenido.")
        if is_frozen():
            input("\nPresione Enter para salir...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error al lanzar servidor: {e}")
        import traceback
        traceback.print_exc()
        if is_frozen():
            input("\nPresione Enter para salir...")
        sys.exit(1)


if __name__ == "__main__":
    main()
