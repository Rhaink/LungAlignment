# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Documentation structure under `docs/` with reproduction and session notes.
- Seed sweep automation scripts for ensemble search.
- MIT license file.
- Config templates for landmark training and ensemble evaluation.
- `scripts/train.py --config` support and config-based ensemble evaluation helper.
- Config templates and `--config` support for classifier and hierarchical training.
- Landmark quickstart script and documentation (train -> ensemble -> predictions).
- `scripts/extract_predictions.py` config/models support for reproducible outputs.
- `train-classifier` CLI `--config` support (with `model` alias for `backbone`).
- ResNet18 classifier reproduction notes for warped_lung_best.
- Documented classifier smoke test artifacts for warped_lung_best.
- Documented ResNet18 full run metrics for warped_lung_best (session_2026-01-12).
- Updated best actual classifier metrics for warped_lung_best (sweep 2026-01-13).
- Added TTA option for classifier evaluation and inference.
- Added classifier accuracy sweep helper script.

### Changed
- Landmark ensemble best improves to 3.61 px with seed666 combo (see `GROUND_TRUTH.json`).
- Archived redundant ensemble scripts in `scripts/archive/` to keep a minimal repro set.
- `scripts/train_classifier.py` now wraps the CLI to avoid duplicated logic.
- Classifier docs/configs point to `outputs/warped_lung_best/session_warping`.

### Fixed
- Landmark evaluation scales per-sample image sizes correctly and serializes nested metrics safely.
- Optimize-margin handles empty val/test splits (quick mode) without numpy runtime warnings.
- Train/evaluate CSV loader now fails fast on headered/category CSVs instead of silently misparsing.
- Train split falls back to non-stratified when class counts are too small, avoiding crash on tiny datasets.
- Train saves both `final_model.pt` and `last.pt` so evaluate can run immediately after training.
- Evaluate split now falls back to non-stratified for tiny datasets.
- `predict` accepts `--output-json` alias in addition to `--json`.
- `extract_predictions.py` accepts `name` from config files (ex: `ensemble_best.json`).
- Original classifier loader now maps `Viral_Pneumonia` to `Viral Pneumonia` directory names.

## [2.0.0] - 2025-12-11

### Added
- Complete CLI with 20 commands via Typer
- Piecewise affine warping with Delaunay triangulation
- `warp_mask()` function for mask transformation
- Full coverage warping with boundary points
- Ensemble model evaluation (4 models + TTA)
- Cross-dataset validation pipeline
- Robustness testing (JPEG compression, Gaussian blur)
- Control experiment for mechanism validation (Session 39)
- Pulmonary Focus Score (PFS) analysis with warped masks
- CONTRIBUTING.md and CHANGELOG.md
- pyproject.toml for pip installation
- Complete seed management for reproducibility

### Changed
- Unified CLAHE tile_size to 4 (from inconsistent 4/8)
- Standardized PIL imports to `from PIL import Image`
- Deduplicated `triangle_area_2x()` in warp.py
- Updated requirements.txt (removed unused dependencies)
- Improved documentation with validated claims

### Fixed
- Bounding box clamping in `get_bounding_box()`
- Cross-evaluation now uses consistent 3 classes
- PFS calculation uses correctly warped masks
- Seeds now include `random.seed()` for stdlib

### Removed
- Unused `hydra-core` and `omegaconf` dependencies
- Legacy Hydra-based configuration system
- Obsolete tests depending on Hydra

### Documentation
- Updated README with validated results from Session 39
- Corrected generalization claim from 11x to 2.4x
- Added mechanism explanation (75% info + 25% geo)
- Session documentation (Sessions 35-42)

## [1.5.0] - 2025-12-08

### Added
- Session 35-38 improvements
- Cross-evaluation command
- Compare-architectures command
- Optimize-margin command

### Fixed
- Warping commands use correct `src_v2.processing.warp`
- CLI validation and error messages

## [1.4.0] - 2025-12-05

### Added
- Grad-CAM visualization
- PFS (Pulmonary Focus Score) analysis
- Error analysis by category

### Changed
- Improved model checkpoint handling
- Better logging throughout CLI

## [1.3.0] - 2025-12-01

### Added
- Image classifier with multiple architectures
- Support for ResNet18, ResNet50, EfficientNet-B0, DenseNet-121
- Train-classifier and evaluate-classifier commands
- Class weighting for imbalanced datasets

## [1.2.0] - 2025-11-25

### Added
- Coordinate Attention Module
- Two-phase training (frozen backbone + fine-tuning)
- Early stopping and model checkpointing
- WingLoss and CombinedLandmarkLoss

### Changed
- Improved landmark prediction accuracy to 3.71 px

## [1.1.0] - 2025-11-20

### Added
- GPA (Generalized Procrustes Analysis)
- Canonical shape computation
- Warp command in CLI
- Generate-dataset command

### Fixed
- Triangle degeneration handling in warping

## [1.0.0] - 2025-11-15

### Added
- Initial release
- ResNet-18 landmark prediction model
- Basic CLI with train, evaluate, predict commands
- LandmarkDataset with CLAHE preprocessing
- Basic test suite

## Validated Results (Session 39-41)

### Landmark Prediction
- **Error**: 3.71 px (ensemble 4 models + TTA)
- Per-category: Normal 3.42 px, COVID 3.77 px, Viral 4.40 px

### Classification
- Original 100%: 98.84% accuracy
- Warped 99%: 98.73% accuracy

### Robustness (vs Original 100%)
- JPEG Q50: **30x more robust** (0.53% vs 16.14% degradation)
- JPEG Q30: **23x more robust** (1.32% vs 29.97% degradation)
- Blur sigma=1: **2.4x more robust** (6.06% vs 14.43% degradation)

### Cross-Dataset Generalization
- Warped model: **2.4x better** (gap 3.17% vs 7.70%)

### Robustness Mechanism
- ~75%: Information reduction (implicit regularization via reduced fill rate)
- ~25%: Geometric normalization (additional warping benefit)
