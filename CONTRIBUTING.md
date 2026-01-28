# Contributing to COVID-19 Landmark Detection

Thank you for your interest in contributing to this project. This document provides guidelines for contributing.

## Development Setup

### Prerequisites

- Python 3.9+
- PyTorch 2.0+ (with CUDA or ROCm support)
- Git

### Installation

1. Clone the repository:
```bash
# Reemplaza <usuario> con tu nombre de usuario de GitHub
git clone https://github.com/<usuario>/prediccion_warping_clasificacion.git
cd prediccion_warping_clasificacion
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

4. Install PyTorch according to your hardware:
```bash
# For NVIDIA GPUs (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For AMD GPUs (ROCm 6.0):
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# For CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Code Style

### Python Style

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Maximum line length: 100 characters
- Use meaningful variable names in English

### Imports

- Standard library imports first
- Third-party imports second
- Local imports third
- Alphabetical order within each group
- Use `from PIL import Image` (not `PILImage`)

### Documentation

- All public functions should have docstrings
- Use Google-style docstrings
- Document parameters, return values, and exceptions

Example:
```python
def predict_landmarks(image: np.ndarray, model: nn.Module) -> np.ndarray:
    """
    Predict anatomical landmarks from a chest X-ray image.

    Args:
        image: Input image as numpy array (H, W, 3)
        model: Trained landmark prediction model

    Returns:
        landmarks: Array of shape (15, 2) with predicted coordinates

    Raises:
        ValueError: If image dimensions are invalid
    """
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=src_v2 --cov-report=html

# Run specific test file
python -m pytest tests/test_processing.py -v

# Run tests matching pattern
python -m pytest tests/ -k "test_warp" -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use pytest fixtures for common setup
- Test edge cases and error conditions

## Pull Request Process

1. Create a feature branch from `main`:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit:
```bash
git add .
git commit -m "feat: description of your changes"
```

3. Ensure all tests pass:
```bash
python -m pytest tests/ -v
```

4. Push to your branch:
```bash
git push origin feature/your-feature-name
```

5. Open a Pull Request with:
   - Clear description of changes
   - Reference to related issues
   - Test results

## Commit Message Convention

Use conventional commit messages:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding or modifying tests
- `refactor:` Code refactoring
- `style:` Code style changes (formatting)
- `chore:` Maintenance tasks

Examples:
```
feat: add support for custom landmark configurations
fix: correct CLAHE tile_size default value
docs: update installation instructions
test: add unit tests for warp_mask function
```

## Project Structure

```
prediccion_warping_clasificacion/
├── src_v2/                    # Main source code
│   ├── cli.py                 # CLI commands (typer)
│   ├── constants.py           # Project constants
│   ├── data/                  # Data loading and transforms
│   ├── models/                # Neural network architectures
│   ├── training/              # Training logic
│   ├── evaluation/            # Evaluation metrics
│   ├── processing/            # Geometric processing (warping)
│   └── visualization/         # Visualization utilities
├── tests/                     # Test suite
├── scripts/                   # Utility scripts
├── docs/                      # Documentation
└── data/                      # Dataset (not in repo)
```

## Key Constants

When modifying code, be aware of these key constants in `src_v2/constants.py`:

- `NUM_LANDMARKS = 15` - Number of anatomical landmarks
- `DEFAULT_IMAGE_SIZE = 224` - Standard image size
- `DEFAULT_CLAHE_TILE_SIZE = 4` - CLAHE tile size (validated)
- `DEFAULT_MARGIN = 1.05` - Warping margin

## Questions?

If you have questions, please open an issue on GitHub.
