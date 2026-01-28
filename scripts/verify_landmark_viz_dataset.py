#!/usr/bin/env python3
"""
Verify that landmark visualization dataset matches warped dataset.

This script ensures that:
1. Same number of images in each split
2. Same image names in each split
3. Same class distribution

Usage:
    python scripts/verify_landmark_viz_dataset.py
    python scripts/verify_landmark_viz_dataset.py <warped_dir> <viz_dir>
"""
import json
from pathlib import Path
from collections import defaultdict


def verify_dataset_alignment(warped_dir: str, viz_dir: str):
    """Verify that visualization dataset has same images as warped dataset."""
    warped_path = Path(warped_dir)
    viz_path = Path(viz_dir)

    errors = []
    matches = []

    for split in ['train', 'val', 'test']:
        for class_name in ['COVID', 'Normal', 'Viral_Pneumonia']:
            # Get warped images
            warped_split_dir = warped_path / split / class_name
            if not warped_split_dir.exists():
                errors.append(f"{split}/{class_name}: warped directory doesn't exist")
                continue

            warped_files = {f.stem.replace('_warped', '')
                           for f in warped_split_dir.glob('*.png')}

            # Get viz images
            viz_split_dir = viz_path / split / class_name
            if not viz_split_dir.exists():
                errors.append(f"{split}/{class_name}: viz directory doesn't exist")
                continue

            viz_files = {f.stem.replace('_landmarks_viz', '')
                        for f in viz_split_dir.glob('*.png')}

            # Compare
            if warped_files != viz_files:
                missing_in_viz = warped_files - viz_files
                extra_in_viz = viz_files - warped_files

                if missing_in_viz:
                    errors.append(
                        f"{split}/{class_name}: {len(missing_in_viz)} images missing in viz"
                    )
                if extra_in_viz:
                    errors.append(
                        f"{split}/{class_name}: {len(extra_in_viz)} extra images in viz"
                    )
            else:
                matches.append(f"{split}/{class_name}: {len(warped_files)} images")
                print(f"✓ {split}/{class_name}: {len(warped_files)} images match")

    print(f"\n{'='*60}")
    print(f"Verification Results:")
    print(f"  Matching splits: {len(matches)}")
    print(f"  Errors: {len(errors)}")

    if errors:
        print(f"\n{'='*60}")
        print("ERRORS DETECTED:")
        for err in errors:
            print(f"  ✗ {err}")
        return False

    print(f"\n✓ All splits match perfectly!")
    return True


if __name__ == "__main__":
    import sys

    warped_dir = "outputs/warped_lung_best/session_warping"
    viz_dir = "outputs/landmark_visualizations/session_warping"

    if len(sys.argv) > 1:
        warped_dir = sys.argv[1]
    if len(sys.argv) > 2:
        viz_dir = sys.argv[2]

    print(f"Warped dataset: {warped_dir}")
    print(f"Viz dataset: {viz_dir}")
    print(f"{'='*60}\n")

    success = verify_dataset_alignment(warped_dir, viz_dir)
    sys.exit(0 if success else 1)
