"""
Landmark Comparison Visualization Module

This module provides functionality to generate comparative visualizations showing
both predicted landmarks and ground truth landmarks on the same image.

Usage:
    python -m src_v2 generate-landmark-comparison-dataset \\
        data/dataset/COVID-19_Radiography_Dataset \\
        outputs/landmark_comparisons/best_ensemble
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from src_v2.constants import (
    DEFAULT_IMAGE_SIZE,
    ORIGINAL_IMAGE_SIZE,
    NUM_LANDMARKS,
)
from src_v2.data.utils import load_coordinates_csv, get_landmarks_array
from src_v2.visualization.utils import draw_scientific_crosses_on_image

logger = logging.getLogger(__name__)


def load_ground_truth_mapping(csv_path: Path) -> Dict[str, np.ndarray]:
    """
    Load ground truth landmarks from CSV.

    Args:
        csv_path: Path to coordenadas_maestro.csv

    Returns:
        Dictionary mapping {image_name: landmarks_299} where landmarks are in 299×299 pixels

    Example:
        >>> gt_dict = load_ground_truth_mapping(Path('data/coordenadas/coordenadas_maestro.csv'))
        >>> print(len(gt_dict))  # 957
        >>> print(gt_dict['COVID-1'].shape)  # (15, 2)
    """
    logger.info(f"Loading ground truth from {csv_path}")

    df = load_coordinates_csv(str(csv_path))
    gt_dict = {}

    for _, row in df.iterrows():
        image_name = row['image_name']
        landmarks = get_landmarks_array(row)  # (15, 2) in 299×299 pixels
        gt_dict[image_name] = landmarks

    logger.info(f"Loaded {len(gt_dict)} ground truth samples")

    # Log category distribution
    categories = defaultdict(int)
    for name in gt_dict.keys():
        if name.startswith('COVID'):
            categories['COVID'] += 1
        elif name.startswith('Normal'):
            categories['Normal'] += 1
        elif name.startswith('Viral'):
            categories['Viral_Pneumonia'] += 1

    logger.info(f"Distribution: {dict(categories)}")

    return gt_dict


def match_predictions_with_gt(
    predictions_dict: Dict,
    gt_dict: Dict
) -> List[Tuple[str, str, np.ndarray, np.ndarray]]:
    """
    Match predictions with ground truth samples.

    Args:
        predictions_dict: Dictionary with {(image_name, category): landmarks_224}
        gt_dict: Dictionary with {image_name: landmarks_299}

    Returns:
        List of tuples: (image_name, category, pred_224, gt_299)

    Example:
        >>> matched = match_predictions_with_gt(pred_dict, gt_dict)
        >>> print(len(matched))  # Should be close to 957
    """
    matched = []
    missing = []

    for image_name, gt_landmarks in gt_dict.items():
        # Search for corresponding prediction
        found = False
        for (pred_name, category), pred_landmarks in predictions_dict.items():
            if pred_name == image_name:
                matched.append((image_name, category, pred_landmarks, gt_landmarks))
                found = True
                break

        if not found:
            missing.append(image_name)

    if missing:
        logger.warning(
            f"{len(missing)} images with GT have no predictions. "
            f"First 5: {missing[:5]}"
        )

    logger.info(f"Matched {len(matched)}/{len(gt_dict)} images with ground truth")
    return matched


def create_comparison_visualization(
    image: np.ndarray,
    pred_landmarks: np.ndarray,
    gt_landmarks: np.ndarray,
    pred_color: str = 'red',
    gt_color: str = 'green',
    cross_size: int = 5,
    thickness: int = 2,
    show_error_lines: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Create dual visualization with predictions and ground truth.

    Args:
        image: Grayscale image 299×299
        pred_landmarks: Predicted landmarks in 299×299 pixels (already scaled)
        gt_landmarks: Ground truth landmarks in 299×299 pixels
        pred_color: Color for predictions ('red', 'blue', 'green', etc.)
        gt_color: Color for ground truth
        cross_size: Size of cross arms in pixels
        thickness: Line thickness
        show_error_lines: If True, draw yellow lines between corresponding pairs

    Returns:
        Tuple of (rgb_image_with_landmarks, error_metrics)

    Example:
        >>> img = cv2.imread('covid.png', cv2.IMREAD_GRAYSCALE)
        >>> pred_299 = pred_224 * (299 / 224)
        >>> img_viz, metrics = create_comparison_visualization(img, pred_299, gt_299)
        >>> print(f"Mean error: {metrics['mean_error_px']:.2f} px")
    """
    # Convert to RGB
    if image.ndim == 2:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = image.copy()

    # Draw ground truth first (green, underneath)
    img_rgb = draw_scientific_crosses_on_image(
        img_rgb,
        gt_landmarks,
        cross_size=cross_size,
        thickness=thickness,
        color=gt_color,
        return_rgb=True
    )

    # Draw predictions on top (red)
    img_rgb = draw_scientific_crosses_on_image(
        img_rgb,
        pred_landmarks,
        cross_size=cross_size,
        thickness=thickness,
        color=pred_color,
        return_rgb=True
    )

    # Optional: draw error lines
    if show_error_lines:
        error_color = (0, 255, 255)  # Yellow in BGR
        for pred_pt, gt_pt in zip(pred_landmarks, gt_landmarks):
            cv2.line(
                img_rgb,
                tuple(pred_pt.astype(int)),
                tuple(gt_pt.astype(int)),
                error_color,
                1,
                cv2.LINE_AA
            )

    # Calculate errors
    errors_per_landmark = np.linalg.norm(pred_landmarks - gt_landmarks, axis=1)  # (15,)

    metrics = {
        'mean_error_px': float(errors_per_landmark.mean()),
        'std_error_px': float(errors_per_landmark.std()),
        'median_error_px': float(np.median(errors_per_landmark)),
        'max_error_px': float(errors_per_landmark.max()),
        'max_error_landmark_idx': int(errors_per_landmark.argmax()),
        'per_landmark_errors': errors_per_landmark.tolist()
    }

    return img_rgb, metrics


def generate_comparison_dataset(
    input_dir: Path,
    output_dir: Path,
    ground_truth_csv: Path,
    predictions_npz: Path,
    seed: int = 42,
    pred_color: str = 'red',
    gt_color: str = 'green',
    cross_size: int = 5,
    cross_thickness: int = 2,
    show_error_lines: bool = False,
    max_per_split: Optional[int] = None
) -> Dict:
    """
    Complete pipeline to generate comparison dataset.

    This function:
    1. Loads ground truth landmarks from CSV (957 images)
    2. Loads predictions from NPZ
    3. Matches both datasets
    4. Creates stratified splits (75/15/10)
    5. Generates dual visualizations:
       - Red crosses: predictions
       - Green crosses: ground truth
    6. Calculates errors per image and landmark
    7. Saves metadata with detailed statistics

    Args:
        input_dir: Path to COVID-19_Radiography_Dataset
        output_dir: Path for output comparison images
        ground_truth_csv: Path to coordenadas_maestro.csv
        predictions_npz: Path to predictions.npz
        seed: Random seed for reproducible splits
        pred_color: Color for predictions
        gt_color: Color for ground truth
        cross_size: Size of crosses in pixels
        cross_thickness: Line thickness
        show_error_lines: Whether to draw yellow lines between pairs
        max_per_split: Maximum images per split (for testing)

    Returns:
        Metadata dictionary with complete processing information

    Example:
        >>> metadata = generate_comparison_dataset(
        ...     input_dir=Path('data/dataset/COVID-19_Radiography_Dataset'),
        ...     output_dir=Path('outputs/landmark_comparisons/test'),
        ...     ground_truth_csv=Path('data/coordenadas/coordenadas_maestro.csv'),
        ...     predictions_npz=Path('outputs/landmark_predictions/.../predictions.npz'),
        ...     max_per_split=5  # For quick testing
        ... )
    """
    start_time = time.time()

    # 1. Load ground truth
    logger.info(f"Loading ground truth from {ground_truth_csv}")
    gt_dict = load_ground_truth_mapping(ground_truth_csv)
    logger.info(f"Ground truth loaded: {len(gt_dict)} images")

    # 2. Load predictions
    logger.info(f"Loading predictions from {predictions_npz}")
    data = np.load(predictions_npz, allow_pickle=True)

    # Handle different possible formats
    if 'landmarks' in data:
        landmarks_224 = data['landmarks']
        image_names = data['image_names']
        categories = data['categories']
    elif 'predictions' in data:
        landmarks_224 = data['predictions']
        image_names = data['image_paths']
        # Extract image names from paths
        image_names = [Path(p).stem for p in image_names]
        # Infer categories from image names
        categories = []
        for name in image_names:
            if name.startswith('COVID'):
                categories.append('COVID')
            elif name.startswith('Normal'):
                categories.append('Normal')
            elif name.startswith('Viral'):
                categories.append('Viral_Pneumonia')
            else:
                categories.append('Unknown')
        categories = np.array(categories)
    else:
        raise ValueError(f"Unknown NPZ format. Keys: {list(data.keys())}")

    predictions_dict = {}
    for i in range(len(landmarks_224)):
        name = str(image_names[i])
        cat = str(categories[i]).replace(" ", "_")
        predictions_dict[(name, cat)] = landmarks_224[i]

    logger.info(f"Predictions loaded: {len(predictions_dict)} images")

    # 3. Matching
    matched = match_predictions_with_gt(predictions_dict, gt_dict)
    logger.info(f"Matched: {len(matched)} images")

    if len(matched) == 0:
        raise ValueError("No matches found between predictions and ground truth!")

    # 4. Create DataFrame for splits
    matched_df = pd.DataFrame([
        {
            'image_name': name,
            'category': cat,
            'pred_landmarks_224': pred,
            'gt_landmarks_299': gt
        }
        for name, cat, pred, gt in matched
    ])

    # 5. Create stratified splits
    from sklearn.model_selection import train_test_split

    # First split: train (75%) vs temp (25%)
    train_df, temp_df = train_test_split(
        matched_df,
        test_size=0.25,
        random_state=seed,
        stratify=matched_df['category']
    )

    # Second split: val (60% of temp = 15% overall) vs test (40% of temp = 10% overall)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.4,
        random_state=seed,
        stratify=temp_df['category']
    )

    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

    logger.info("Splits created:")
    for split_name, split_df in splits.items():
        logger.info(f"  {split_name}: {len(split_df)} images")

    # 6. Processing
    SCALE_FACTOR = ORIGINAL_IMAGE_SIZE / DEFAULT_IMAGE_SIZE  # 299 / 224

    all_metadata = []
    stats = defaultdict(int)
    split_errors = defaultdict(list)

    for split_name, split_df in splits.items():
        logger.info(f"\n=== Processing {split_name.upper()} ===")

        if max_per_split:
            split_df = split_df.head(max_per_split)
            logger.info(f"Limited to {max_per_split} images for testing")

        for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"  {split_name}"):
            image_name = row['image_name']
            category = row['category']
            pred_landmarks_224 = row['pred_landmarks_224']
            gt_landmarks_299 = row['gt_landmarks_299']

            # Build path to original image
            # Try different possible structures
            possible_paths = [
                input_dir / category / "images" / f"{image_name}.png",
                input_dir / category / f"{image_name}.png",
                input_dir / category.replace("_", " ") / "images" / f"{image_name}.png",
            ]

            image_path = None
            for p in possible_paths:
                if p.exists():
                    image_path = p
                    break

            if image_path is None:
                logger.warning(f"Image not found: {image_name}")
                stats['missing_images'] += 1
                continue

            # Load image
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.warning(f"Failed to load: {image_path}")
                stats['failed_load'] += 1
                continue

            # Verify size
            if image.shape[:2] != (ORIGINAL_IMAGE_SIZE, ORIGINAL_IMAGE_SIZE):
                logger.warning(
                    f"Resizing {image_name} from {image.shape} to "
                    f"{ORIGINAL_IMAGE_SIZE}×{ORIGINAL_IMAGE_SIZE}"
                )
                image = cv2.resize(
                    image,
                    (ORIGINAL_IMAGE_SIZE, ORIGINAL_IMAGE_SIZE)
                )
                stats['resized'] += 1

            # Scale predictions to 299×299
            pred_landmarks_299 = pred_landmarks_224 * SCALE_FACTOR
            pred_landmarks_299 = np.clip(
                pred_landmarks_299,
                0,
                ORIGINAL_IMAGE_SIZE - 1
            )

            # Create visualization
            img_viz, metrics = create_comparison_visualization(
                image,
                pred_landmarks_299,
                gt_landmarks_299,
                pred_color=pred_color,
                gt_color=gt_color,
                cross_size=cross_size,
                thickness=cross_thickness,
                show_error_lines=show_error_lines
            )

            # Save
            output_filename = f"{image_name}_comparison.png"
            output_path = output_dir / split_name / category / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), img_viz)

            # Metadata
            all_metadata.append({
                'image_name': image_name,
                'category': category,
                'split': split_name,
                **metrics
            })

            split_errors[split_name].append(metrics['mean_error_px'])
            stats['processed'] += 1

    # 7. Global statistics
    all_errors = [m['mean_error_px'] for m in all_metadata]

    error_statistics = {
        'overall': {
            'mean_error_px': float(np.mean(all_errors)),
            'std_error_px': float(np.std(all_errors)),
            'median_error_px': float(np.median(all_errors)),
            'min_error_px': float(np.min(all_errors)),
            'max_error_px': float(np.max(all_errors)),
        },
        'per_split': {},
        'per_category': {},
        'per_landmark': {}
    }

    # Per split
    for split_name, errors in split_errors.items():
        if len(errors) > 0:
            error_statistics['per_split'][split_name] = {
                'mean': float(np.mean(errors)),
                'std': float(np.std(errors)),
                'median': float(np.median(errors)),
                'count': len(errors)
            }

    # Per category
    for category in set(m['category'] for m in all_metadata):
        cat_errors = [m['mean_error_px'] for m in all_metadata if m['category'] == category]
        if len(cat_errors) > 0:
            error_statistics['per_category'][category] = {
                'mean': float(np.mean(cat_errors)),
                'std': float(np.std(cat_errors)),
                'count': len(cat_errors)
            }

    # Per landmark
    for i in range(NUM_LANDMARKS):
        landmark_errors = [m['per_landmark_errors'][i] for m in all_metadata]
        if len(landmark_errors) > 0:
            error_statistics['per_landmark'][f'L{i+1}'] = {
                'mean': float(np.mean(landmark_errors)),
                'std': float(np.std(landmark_errors)),
                'median': float(np.median(landmark_errors))
            }

    # Worst cases
    sorted_metadata = sorted(all_metadata, key=lambda x: x['mean_error_px'], reverse=True)
    error_statistics['worst_cases'] = [
        {
            'image': m['image_name'],
            'category': m['category'],
            'split': m['split'],
            'error': m['mean_error_px']
        }
        for m in sorted_metadata[:10]
    ]

    # 8. Save outputs
    metadata = {
        'schema_version': '1.0.0',
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'input_dir': str(input_dir),
        'ground_truth_csv': str(ground_truth_csv),
        'predictions_npz': str(predictions_npz),
        'coordinate_spaces': {
            'ground_truth': '299x299',
            'predictions_original': '224x224',
            'visualization': '299x299',
            'scale_factor': float(SCALE_FACTOR)
        },
        'splits': {
            'seed': seed,
            'ratios': {'train': 0.75, 'val': 0.15, 'test': 0.10}
        },
        'visualization_config': {
            'pred_color': pred_color,
            'gt_color': gt_color,
            'cross_size': cross_size,
            'thickness': cross_thickness,
            'show_error_lines': show_error_lines
        },
        'processing_stats': dict(stats),
        'processing_time_seconds': time.time() - start_time,
        'image_metadata': all_metadata
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / 'error_statistics.json', 'w') as f:
        json.dump(error_statistics, f, indent=2)

    # CSV with errors
    pd.DataFrame(all_metadata).to_csv(output_dir / 'image_errors.csv', index=False)

    logger.info(f"\n{'='*60}")
    logger.info("Comparison dataset completed!")
    logger.info(f"{'='*60}")
    logger.info(f"Images processed: {stats['processed']}")
    logger.info(f"Mean error: {error_statistics['overall']['mean_error_px']:.2f} px")
    logger.info(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    logger.info(f"\nOutput directory: {output_dir}")

    return metadata


def find_original_image(base_dir: Path, image_name: str, category: str) -> Optional[Path]:
    """
    Find original image in different possible directory structures.

    Args:
        base_dir: Base directory for dataset (e.g., data/dataset/COVID-19_Radiography_Dataset)
        image_name: Image name without extension (e.g., 'COVID-1')
        category: Category name (e.g., 'COVID', 'Normal', 'Viral_Pneumonia')

    Returns:
        Path to original image if found, None otherwise

    Example:
        >>> path = find_original_image(
        ...     Path('data/dataset/COVID-19_Radiography_Dataset'),
        ...     'COVID-1',
        ...     'COVID'
        ... )
        >>> print(path)  # data/dataset/COVID-19_Radiography_Dataset/COVID/images/COVID-1.png
    """
    possible_paths = [
        base_dir / category / "images" / f"{image_name}.png",
        base_dir / category / f"{image_name}.png",
        base_dir / category.replace("_", " ") / "images" / f"{image_name}.png",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    return None


def load_predictions_mapping(npz_path: Path) -> Dict[str, np.ndarray]:
    """
    Load predictions from NPZ and return mapping dict.

    Args:
        npz_path: Path to predictions.npz file

    Returns:
        Dictionary mapping {image_name: landmarks_224} where landmarks are in 224×224 pixels

    Example:
        >>> pred_dict = load_predictions_mapping(Path('outputs/landmark_predictions/.../predictions.npz'))
        >>> print(len(pred_dict))  # 15153
        >>> print(pred_dict['COVID-1'].shape)  # (15, 2)
    """
    logger.info(f"Loading predictions from {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    # Handle different possible formats
    if 'landmarks' in data:
        landmarks_224 = data['landmarks']
        image_names = data['image_names']
    elif 'predictions' in data:
        landmarks_224 = data['predictions']
        image_paths = data['image_paths']
        image_names = [Path(p).stem for p in image_paths]
    else:
        raise ValueError(f"Unknown NPZ format. Keys: {list(data.keys())}")

    predictions_dict = {}
    for i in range(len(landmarks_224)):
        name = str(image_names[i])
        predictions_dict[name] = landmarks_224[i]

    logger.info(f"Loaded {len(predictions_dict)} predictions")
    return predictions_dict


def generate_comparison_dataset_aligned(
    warped_dataset_dir: Path,
    output_dir: Path,
    ground_truth_csv: Path,
    predictions_npz: Path,
    original_images_dir: Path,
    pred_color: str = 'red',
    gt_color: str = 'green',
    cross_size: int = 5,
    cross_thickness: int = 2,
    show_error_lines: bool = False,
) -> Dict:
    """
    Generate comparison dataset aligned with warped classification dataset.

    This function creates a comparison visualization dataset that mirrors the exact
    structure of the warped dataset used for classification, ensuring perfect alignment
    for analysis and debugging.

    Key differences from generate_comparison_dataset():
    - Uses warped dataset splits instead of creating new ones
    - Maintains exact folder structure and file naming
    - Provides 1:1 mapping with classification dataset
    - Only processes images that have ground truth (~957 of 15,153)

    Args:
        warped_dataset_dir: Path to warped dataset (e.g., outputs/warped_lung_best/session_warping)
        output_dir: Output directory for comparison images
        ground_truth_csv: Path to coordenadas_maestro.csv with GT landmarks
        predictions_npz: Path to predictions.npz with ensemble predictions
        original_images_dir: Path to original 299×299 images directory
        pred_color: Color for prediction landmarks ('red', 'blue', 'green', etc.)
        gt_color: Color for ground truth landmarks
        cross_size: Size of cross arms in pixels
        cross_thickness: Line thickness
        show_error_lines: If True, draw yellow lines between corresponding pairs

    Returns:
        Metadata dictionary with processing information and alignment stats

    Example:
        >>> metadata = generate_comparison_dataset_aligned(
        ...     warped_dataset_dir=Path('outputs/warped_lung_best/session_warping'),
        ...     output_dir=Path('outputs/landmark_comparisons/aligned_with_classifier'),
        ...     ground_truth_csv=Path('data/coordenadas/coordenadas_maestro.csv'),
        ...     predictions_npz=Path('outputs/landmark_predictions/session_warping/predictions.npz'),
        ...     original_images_dir=Path('data/dataset/COVID-19_Radiography_Dataset')
        ... )
        >>> print(f"Processed: {metadata['coverage']['processed']} images")
        >>> print(f"Coverage: {metadata['coverage']['coverage_percentage']:.2f}%")
    """
    start_time = time.time()

    logger.info("="*60)
    logger.info("Generating Comparison Dataset Aligned with Warped Dataset")
    logger.info("="*60)

    # 1. Load ground truth (957 images)
    logger.info(f"\n[1/6] Loading ground truth from {ground_truth_csv}")
    gt_dict = load_ground_truth_mapping(ground_truth_csv)

    # 2. Load predictions (15,153 images)
    logger.info(f"\n[2/6] Loading predictions from {predictions_npz}")
    predictions_dict = load_predictions_mapping(predictions_npz)

    # 3. Initialize statistics
    stats = {
        'total_in_warped': 0,
        'with_gt': 0,
        'processed': 0,
        'missing_original': 0,
        'missing_prediction': 0,
        'failed_load': 0,
        'per_split': {}
    }
    all_metadata = []
    SCALE_FACTOR = ORIGINAL_IMAGE_SIZE / DEFAULT_IMAGE_SIZE  # 299 / 224

    # 4. Process each split
    splits = ['train', 'val', 'test']
    logger.info(f"\n[3/6] Processing splits: {splits}")

    for split_name in splits:
        split_dir = warped_dataset_dir / split_name
        images_csv_path = split_dir / 'images.csv'

        if not images_csv_path.exists():
            logger.warning(f"Split CSV not found: {images_csv_path}")
            continue

        # Read images.csv from warped dataset
        # Format: image_name,category,warped_filename
        images_df = pd.read_csv(images_csv_path)

        logger.info(f"\n=== Processing {split_name.upper()} ===")
        logger.info(f"Total images in warped dataset: {len(images_df)}")

        stats['total_in_warped'] += len(images_df)
        split_stats = {
            'total': len(images_df),
            'with_gt': 0,
            'processed': 0,
            'missing_original': 0,
            'missing_prediction': 0,
            'failed_load': 0
        }

        # Filter only images that have ground truth
        images_with_gt = images_df[images_df['image_name'].isin(gt_dict.keys())]
        logger.info(f"Images with ground truth: {len(images_with_gt)}")

        stats['with_gt'] += len(images_with_gt)
        split_stats['with_gt'] = len(images_with_gt)

        # Process each image with GT
        for idx, row in tqdm(images_with_gt.iterrows(), total=len(images_with_gt), desc=f"  {split_name}"):
            image_name = row['image_name']
            category = row['category']
            warped_filename = row['warped_filename']  # e.g., "COVID-1_warped.png"

            # Get ground truth landmarks (299×299)
            gt_landmarks_299 = gt_dict[image_name]

            # Get predictions (224×224)
            if image_name not in predictions_dict:
                logger.warning(f"No prediction for {image_name}")
                stats['missing_prediction'] += 1
                split_stats['missing_prediction'] += 1
                continue

            pred_landmarks_224 = predictions_dict[image_name]

            # Find original image (299×299)
            original_image_path = find_original_image(original_images_dir, image_name, category)

            if original_image_path is None:
                logger.warning(f"Original image not found: {image_name}")
                stats['missing_original'] += 1
                split_stats['missing_original'] += 1
                continue

            # Load image
            image_orig = cv2.imread(str(original_image_path), cv2.IMREAD_GRAYSCALE)
            if image_orig is None:
                logger.warning(f"Failed to load: {original_image_path}")
                stats['failed_load'] += 1
                split_stats['failed_load'] += 1
                continue

            # Resize if necessary to 299×299
            if image_orig.shape[:2] != (ORIGINAL_IMAGE_SIZE, ORIGINAL_IMAGE_SIZE):
                image_orig = cv2.resize(
                    image_orig,
                    (ORIGINAL_IMAGE_SIZE, ORIGINAL_IMAGE_SIZE)
                )

            # Scale predictions from 224 → 299
            pred_landmarks_299 = pred_landmarks_224 * SCALE_FACTOR
            pred_landmarks_299 = np.clip(pred_landmarks_299, 0, ORIGINAL_IMAGE_SIZE - 1)

            # Create comparison visualization
            img_viz, metrics = create_comparison_visualization(
                image_orig,
                pred_landmarks_299,
                gt_landmarks_299,
                pred_color=pred_color,
                gt_color=gt_color,
                cross_size=cross_size,
                thickness=cross_thickness,
                show_error_lines=show_error_lines
            )

            # Save in SAME structure as warped dataset
            # Name: {image_name}_comparison.png (equivalent to {image_name}_warped.png)
            output_filename = f"{image_name}_comparison.png"
            output_path = output_dir / split_name / category / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(output_path), img_viz)

            # Save metadata
            all_metadata.append({
                'image_name': image_name,
                'category': category,
                'split': split_name,
                'warped_filename': warped_filename,
                'comparison_filename': output_filename,
                **metrics
            })

            stats['processed'] += 1
            split_stats['processed'] += 1

        stats['per_split'][split_name] = split_stats

        logger.info(f"  Processed: {split_stats['processed']}/{split_stats['with_gt']} images with GT")
        if split_stats['missing_original'] > 0:
            logger.warning(f"  Missing original: {split_stats['missing_original']}")
        if split_stats['missing_prediction'] > 0:
            logger.warning(f"  Missing prediction: {split_stats['missing_prediction']}")

    # 5. Calculate error statistics
    logger.info(f"\n[4/6] Calculating error statistics")

    all_errors = [m['mean_error_px'] for m in all_metadata]

    error_statistics = {
        'overall': {
            'mean_error_px': float(np.mean(all_errors)) if all_errors else 0.0,
            'std_error_px': float(np.std(all_errors)) if all_errors else 0.0,
            'median_error_px': float(np.median(all_errors)) if all_errors else 0.0,
            'min_error_px': float(np.min(all_errors)) if all_errors else 0.0,
            'max_error_px': float(np.max(all_errors)) if all_errors else 0.0,
        },
        'per_split': {},
        'per_category': {},
        'per_landmark': {}
    }

    # Per split
    for split_name in splits:
        split_errors = [m['mean_error_px'] for m in all_metadata if m['split'] == split_name]
        if len(split_errors) > 0:
            error_statistics['per_split'][split_name] = {
                'mean': float(np.mean(split_errors)),
                'std': float(np.std(split_errors)),
                'median': float(np.median(split_errors)),
                'count': len(split_errors)
            }

    # Per category
    for category in set(m['category'] for m in all_metadata):
        cat_errors = [m['mean_error_px'] for m in all_metadata if m['category'] == category]
        if len(cat_errors) > 0:
            error_statistics['per_category'][category] = {
                'mean': float(np.mean(cat_errors)),
                'std': float(np.std(cat_errors)),
                'count': len(cat_errors)
            }

    # Per landmark
    for i in range(NUM_LANDMARKS):
        landmark_errors = [m['per_landmark_errors'][i] for m in all_metadata]
        if len(landmark_errors) > 0:
            error_statistics['per_landmark'][f'L{i+1}'] = {
                'mean': float(np.mean(landmark_errors)),
                'std': float(np.std(landmark_errors)),
                'median': float(np.median(landmark_errors))
            }

    # Worst cases
    sorted_metadata = sorted(all_metadata, key=lambda x: x['mean_error_px'], reverse=True)
    error_statistics['worst_cases'] = [
        {
            'image': m['image_name'],
            'category': m['category'],
            'split': m['split'],
            'error': m['mean_error_px']
        }
        for m in sorted_metadata[:10]
    ]

    # 6. Save outputs
    logger.info(f"\n[5/6] Saving metadata and statistics")

    coverage_percentage = (stats['processed'] / stats['total_in_warped'] * 100) if stats['total_in_warped'] > 0 else 0.0

    metadata = {
        'schema_version': '2.0.0',
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'alignment': {
            'warped_dataset_dir': str(warped_dataset_dir),
            'aligned_with_classifier': True,
            'split_source': 'warped_dataset_images_csv'
        },
        'input_files': {
            'ground_truth_csv': str(ground_truth_csv),
            'predictions_npz': str(predictions_npz),
            'original_images_dir': str(original_images_dir)
        },
        'coordinate_spaces': {
            'ground_truth': '299x299',
            'predictions_original': '224x224',
            'visualization': '299x299',
            'scale_factor': float(SCALE_FACTOR)
        },
        'coverage': {
            'total_in_warped': stats['total_in_warped'],
            'with_ground_truth': stats['with_gt'],
            'processed': stats['processed'],
            'coverage_percentage': coverage_percentage
        },
        'visualization_config': {
            'pred_color': pred_color,
            'gt_color': gt_color,
            'cross_size': cross_size,
            'thickness': cross_thickness,
            'show_error_lines': show_error_lines
        },
        'processing_stats': stats,
        'processing_time_seconds': time.time() - start_time,
        'image_metadata': all_metadata
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / 'error_statistics.json', 'w') as f:
        json.dump(error_statistics, f, indent=2)

    # CSV with errors
    if all_metadata:
        pd.DataFrame(all_metadata).to_csv(output_dir / 'image_errors.csv', index=False)

    # 7. Final summary
    logger.info(f"\n[6/6] {'='*60}")
    logger.info("Aligned Comparison Dataset Completed!")
    logger.info(f"{'='*60}")
    logger.info(f"Total images in warped dataset: {stats['total_in_warped']}")
    logger.info(f"Images with ground truth: {stats['with_gt']}")
    logger.info(f"Images processed: {stats['processed']}")
    logger.info(f"Coverage: {coverage_percentage:.2f}%")
    if all_errors:
        logger.info(f"Mean error: {error_statistics['overall']['mean_error_px']:.2f} px")
    logger.info(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info(f"\nAlignment verification:")
    logger.info(f"  - Same splits as: {warped_dataset_dir}")
    logger.info(f"  - Structure: {split_name}/{{category}}/{{image_name}}_comparison.png")

    return metadata
