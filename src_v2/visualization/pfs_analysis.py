"""Pulmonary Focus Score (PFS) Analysis Module.

This module provides comprehensive PFS analysis for evaluating whether
the model focuses its attention on pulmonary (lung) regions.

PFS = sum(heatmap * mask) / sum(heatmap)
- 1.0 = Model focuses entirely on lungs (ideal)
- 0.5 = Equal attention on lung and non-lung regions
- <0.5 = Model focuses more on non-lung areas (concerning)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import csv

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src_v2.visualization.gradcam import GradCAM, get_target_layer, calculate_pfs

logger = logging.getLogger(__name__)


@dataclass
class PFSResult:
    """Result of PFS calculation for a single image."""

    image_path: str
    true_class: str
    predicted_class: str
    confidence: float
    pfs: float
    correct: bool

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PFSSummary:
    """Summary statistics for PFS analysis."""

    total_samples: int
    mean_pfs: float
    std_pfs: float
    median_pfs: float
    min_pfs: float
    max_pfs: float
    pfs_by_class: Dict[str, Dict[str, float]]
    pfs_correct_vs_incorrect: Dict[str, Dict[str, float]]
    low_pfs_count: int
    low_pfs_rate: float
    threshold: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class PFSAnalyzer:
    """Analyzer for Pulmonary Focus Score calculations.

    Args:
        class_names: List of class names (e.g., ['COVID', 'Normal', 'Viral_Pneumonia'])
        threshold: PFS threshold below which attention is considered concerning

    Example:
        >>> analyzer = PFSAnalyzer(['COVID', 'Normal', 'Viral_Pneumonia'])
        >>> analyzer.add_result(PFSResult(...))
        >>> summary = analyzer.get_summary()
        >>> analyzer.save_reports(output_dir)
    """

    def __init__(self, class_names: List[str], threshold: float = 0.5):
        if not class_names:
            raise ValueError("class_names cannot be empty")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        self.class_names = class_names
        self.threshold = threshold
        self.results: List[PFSResult] = []

    def add_result(self, result: PFSResult) -> None:
        """Add a single PFS result."""
        self.results.append(result)

    def add_results(self, results: List[PFSResult]) -> None:
        """Add multiple PFS results."""
        self.results.extend(results)

    def get_summary(self) -> PFSSummary:
        """Calculate summary statistics from collected results.

        Returns:
            PFSSummary with statistics

        Raises:
            ValueError: If no results have been added
        """
        if not self.results:
            raise ValueError("No results to summarize. Add results first.")

        pfs_values = np.array([r.pfs for r in self.results])

        # Overall statistics
        mean_pfs = float(np.mean(pfs_values))
        std_pfs = float(np.std(pfs_values))
        median_pfs = float(np.median(pfs_values))
        min_pfs = float(np.min(pfs_values))
        max_pfs = float(np.max(pfs_values))

        # PFS by class
        pfs_by_class = {}
        for class_name in self.class_names:
            class_results = [r for r in self.results if r.true_class == class_name]
            if class_results:
                class_pfs = [r.pfs for r in class_results]
                pfs_by_class[class_name] = {
                    "mean": float(np.mean(class_pfs)),
                    "std": float(np.std(class_pfs)),
                    "count": len(class_pfs),
                }
            else:
                pfs_by_class[class_name] = {"mean": 0.0, "std": 0.0, "count": 0}

        # PFS for correct vs incorrect predictions
        correct_results = [r for r in self.results if r.correct]
        incorrect_results = [r for r in self.results if not r.correct]

        pfs_correct_vs_incorrect = {
            "correct": {
                "mean": float(np.mean([r.pfs for r in correct_results])) if correct_results else 0.0,
                "std": float(np.std([r.pfs for r in correct_results])) if correct_results else 0.0,
                "count": len(correct_results),
            },
            "incorrect": {
                "mean": float(np.mean([r.pfs for r in incorrect_results])) if incorrect_results else 0.0,
                "std": float(np.std([r.pfs for r in incorrect_results])) if incorrect_results else 0.0,
                "count": len(incorrect_results),
            },
        }

        # Low PFS count
        low_pfs_count = sum(1 for r in self.results if r.pfs < self.threshold)
        low_pfs_rate = low_pfs_count / len(self.results) if self.results else 0.0

        return PFSSummary(
            total_samples=len(self.results),
            mean_pfs=mean_pfs,
            std_pfs=std_pfs,
            median_pfs=median_pfs,
            min_pfs=min_pfs,
            max_pfs=max_pfs,
            pfs_by_class=pfs_by_class,
            pfs_correct_vs_incorrect=pfs_correct_vs_incorrect,
            low_pfs_count=low_pfs_count,
            low_pfs_rate=low_pfs_rate,
            threshold=self.threshold,
        )

    def get_low_pfs_results(self) -> List[PFSResult]:
        """Get results with PFS below threshold."""
        return [r for r in self.results if r.pfs < self.threshold]

    def save_reports(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """Save analysis reports to files.

        Args:
            output_dir: Directory to save reports

        Returns:
            Dictionary mapping report type to file path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save JSON summary
        summary = self.get_summary()
        summary_file = output_path / "pfs_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)
        saved_files["summary"] = summary_file
        logger.info("Saved PFS summary to %s", summary_file)

        # Save detailed CSV
        details_file = output_path / "pfs_details.csv"
        with open(details_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["image_path", "true_class", "predicted_class", "confidence", "pfs", "correct"],
            )
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.to_dict())
        saved_files["details"] = details_file
        logger.info("Saved PFS details to %s", details_file)

        # Save per-class CSV
        by_class_file = output_path / "pfs_by_class.csv"
        with open(by_class_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "mean_pfs", "std_pfs", "count"])
            for class_name, stats in summary.pfs_by_class.items():
                writer.writerow([class_name, stats["mean"], stats["std"], stats["count"]])
        saved_files["by_class"] = by_class_file
        logger.info("Saved PFS by class to %s", by_class_file)

        # Save low PFS samples
        low_pfs_results = self.get_low_pfs_results()
        if low_pfs_results:
            low_pfs_file = output_path / "low_pfs_samples.csv"
            with open(low_pfs_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["image_path", "true_class", "predicted_class", "confidence", "pfs", "correct"],
                )
                writer.writeheader()
                for result in sorted(low_pfs_results, key=lambda x: x.pfs):
                    writer.writerow(result.to_dict())
            saved_files["low_pfs"] = low_pfs_file
            logger.info("Saved %d low PFS samples to %s", len(low_pfs_results), low_pfs_file)

        return saved_files


def load_lung_mask(mask_path: Union[str, Path]) -> np.ndarray:
    """Load a lung mask from file.

    Args:
        mask_path: Path to mask image (PNG, grayscale or RGB)

    Returns:
        Binary mask array normalized to [0, 1]
    """
    mask = np.array(Image.open(mask_path))

    # Convert RGB to grayscale
    if len(mask.shape) == 3:
        mask = np.mean(mask, axis=2)

    # Normalize to [0, 1]
    if mask.max() > 1:
        mask = mask / 255.0

    return mask.astype(np.float32)


def find_mask_for_image(
    image_path: Union[str, Path],
    mask_dir: Union[str, Path],
    class_name: str,
) -> Optional[Path]:
    """Find the corresponding lung mask for an image.

    Handles both original and warped images by removing '_warped' suffix.

    Args:
        image_path: Path to the image
        mask_dir: Base directory containing masks
        class_name: Class name (e.g., 'COVID')

    Returns:
        Path to mask if found, None otherwise
    """
    image_path = Path(image_path)
    mask_dir = Path(mask_dir)

    # Get image name without extension
    image_name = image_path.stem

    # Remove '_warped' suffix if present
    if image_name.endswith("_warped"):
        image_name = image_name[:-7]

    # Try different mask directory structures and naming conventions
    # Includes _mask.png suffix used by generate-lung-masks command
    possible_paths = [
        mask_dir / class_name / "masks" / f"{image_name}.png",
        mask_dir / class_name / "masks" / f"{image_name}_mask.png",
        mask_dir / class_name / f"{image_name}.png",
        mask_dir / class_name / f"{image_name}_mask.png",
        mask_dir / "masks" / class_name / f"{image_name}.png",
        mask_dir / "masks" / class_name / f"{image_name}_mask.png",
        mask_dir / f"{image_name}.png",
        mask_dir / f"{image_name}_mask.png",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    return None


def generate_approximate_mask(
    image_shape: Tuple[int, int],
    margin: float = 0.15,
) -> np.ndarray:
    """Generate an approximate rectangular lung mask.

    Creates a centered rectangular region representing approximate lung area.

    Args:
        image_shape: (height, width) of the image
        margin: Fraction of image to exclude from edges (0-0.5)

    Returns:
        Binary mask with 1s in the approximate lung region
    """
    if not (0.0 <= margin < 0.5):
        raise ValueError(f"margin must be in [0, 0.5), got {margin}")

    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.float32)

    # Calculate margins
    h_margin = int(height * margin)
    w_margin = int(width * margin)

    # Fill central region
    mask[h_margin : height - h_margin, w_margin : width - w_margin] = 1.0

    return mask


def run_pfs_analysis(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    device: torch.device,
    mask_dir: Optional[Union[str, Path]] = None,
    use_approximate_masks: bool = False,
    approximate_margin: float = 0.15,
    num_samples: Optional[int] = None,
) -> Tuple[PFSAnalyzer, List[Dict]]:
    """Run PFS analysis on a dataset.

    Args:
        model: Classifier model
        dataloader: DataLoader with images
        class_names: List of class names
        device: Torch device
        mask_dir: Directory with lung masks (if available)
        use_approximate_masks: Use rectangular approximation if no masks
        approximate_margin: Margin for approximate masks
        num_samples: Limit number of samples (None = all)

    Returns:
        Tuple of (PFSAnalyzer with results, list of detailed info)
    """
    model.eval()
    backbone_name = getattr(model, "backbone_name", "resnet18")
    target_layer = get_target_layer(model, backbone_name)

    analyzer = PFSAnalyzer(class_names)
    detailed_results = []

    # Count samples per class
    samples_per_class = {c: 0 for c in class_names}
    max_per_class = num_samples // len(class_names) if num_samples else float("inf")

    with GradCAM(model, target_layer) as gradcam:
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Process one image at a time (GradCAM requirement)
            for i in range(images.shape[0]):
                image = images[i : i + 1].to(device)
                label = labels[i].item()
                true_class = class_names[label]

                # Check sample limit per class
                if samples_per_class[true_class] >= max_per_class:
                    continue

                # Get image path from dataset
                dataset = dataloader.dataset
                if hasattr(dataset, "samples"):
                    start_idx = batch_idx * dataloader.batch_size
                    img_path = dataset.samples[start_idx + i][0]
                else:
                    img_path = f"batch_{batch_idx}_sample_{i}"

                try:
                    # Generate GradCAM
                    heatmap, pred_class, confidence = gradcam(image)
                    predicted_class = class_names[pred_class]

                    # Get or generate mask
                    if mask_dir:
                        mask_path = find_mask_for_image(img_path, mask_dir, true_class)
                        if mask_path:
                            mask = load_lung_mask(mask_path)
                        elif use_approximate_masks:
                            mask = generate_approximate_mask(heatmap.shape, approximate_margin)
                        else:
                            logger.warning("No mask found for %s, skipping", img_path)
                            continue
                    elif use_approximate_masks:
                        mask = generate_approximate_mask(heatmap.shape, approximate_margin)
                    else:
                        raise ValueError("Either mask_dir or use_approximate_masks must be set")

                    # Calculate PFS
                    pfs = calculate_pfs(heatmap, mask)

                    # Store result
                    result = PFSResult(
                        image_path=str(img_path),
                        true_class=true_class,
                        predicted_class=predicted_class,
                        confidence=confidence,
                        pfs=pfs,
                        correct=(pred_class == label),
                    )
                    analyzer.add_result(result)

                    detailed_results.append({
                        "image_path": str(img_path),
                        "heatmap": heatmap,
                        "mask": mask,
                        "pfs": pfs,
                        "result": result,
                    })

                    samples_per_class[true_class] += 1

                except Exception as e:
                    logger.warning("Error processing %s: %s", img_path, str(e))
                    continue

            # Check if we've collected enough samples
            if num_samples and len(analyzer.results) >= num_samples:
                break

    return analyzer, detailed_results


def create_pfs_visualizations(
    detailed_results: List[Dict],
    output_dir: Union[str, Path],
    summary: PFSSummary,
) -> Dict[str, Path]:
    """Create PFS analysis visualizations.

    Args:
        detailed_results: List of detailed results with heatmaps
        output_dir: Directory to save figures
        summary: PFS summary statistics

    Returns:
        Dictionary mapping figure type to file path
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    figures_dir = output_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved_figures = {}

    # 1. PFS Distribution Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    pfs_values = [r["pfs"] for r in detailed_results]
    ax.hist(pfs_values, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(summary.mean_pfs, color="red", linestyle="--", label=f"Mean: {summary.mean_pfs:.3f}")
    ax.axvline(summary.threshold, color="orange", linestyle=":", label=f"Threshold: {summary.threshold}")
    ax.set_xlabel("Pulmonary Focus Score (PFS)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Pulmonary Focus Scores")
    ax.legend()
    ax.grid(True, alpha=0.3)

    dist_file = figures_dir / "pfs_distribution.png"
    fig.savefig(dist_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_figures["distribution"] = dist_file

    # 2. PFS by Class
    fig, ax = plt.subplots(figsize=(10, 6))
    classes = list(summary.pfs_by_class.keys())
    means = [summary.pfs_by_class[c]["mean"] for c in classes]
    stds = [summary.pfs_by_class[c]["std"] for c in classes]

    x = np.arange(len(classes))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, edgecolor="black")

    # Color bars based on threshold
    for bar, mean in zip(bars, means):
        bar.set_color("green" if mean >= summary.threshold else "red")

    ax.axhline(summary.threshold, color="orange", linestyle="--", label=f"Threshold: {summary.threshold}")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("Mean PFS (± std)")
    ax.set_title("Pulmonary Focus Score by Class")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    by_class_file = figures_dir / "pfs_by_class.png"
    fig.savefig(by_class_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_figures["by_class"] = by_class_file

    # 3. PFS vs Confidence scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    pfs_values = [r["result"].pfs for r in detailed_results]
    confidence_values = [r["result"].confidence for r in detailed_results]
    correct_values = [r["result"].correct for r in detailed_results]

    colors = ["green" if c else "red" for c in correct_values]
    ax.scatter(confidence_values, pfs_values, c=colors, alpha=0.5, edgecolors="none")

    ax.axhline(summary.threshold, color="orange", linestyle="--", label=f"PFS Threshold: {summary.threshold}")
    ax.set_xlabel("Prediction Confidence")
    ax.set_ylabel("Pulmonary Focus Score (PFS)")
    ax.set_title("PFS vs Prediction Confidence\n(Green=Correct, Red=Incorrect)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    scatter_file = figures_dir / "pfs_vs_confidence.png"
    fig.savefig(scatter_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_figures["vs_confidence"] = scatter_file

    # 4. Correct vs Incorrect comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = ["Correct", "Incorrect"]
    means = [
        summary.pfs_correct_vs_incorrect["correct"]["mean"],
        summary.pfs_correct_vs_incorrect["incorrect"]["mean"],
    ]
    stds = [
        summary.pfs_correct_vs_incorrect["correct"]["std"],
        summary.pfs_correct_vs_incorrect["incorrect"]["std"],
    ]
    counts = [
        summary.pfs_correct_vs_incorrect["correct"]["count"],
        summary.pfs_correct_vs_incorrect["incorrect"]["count"],
    ]

    x = np.arange(len(categories))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, edgecolor="black", color=["green", "red"])

    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[i] + 0.02,
                f"n={count}", ha="center", va="bottom")

    ax.axhline(summary.threshold, color="orange", linestyle="--", label=f"Threshold: {summary.threshold}")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Mean PFS (± std)")
    ax.set_title("PFS for Correct vs Incorrect Predictions")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.0)

    correct_file = figures_dir / "pfs_correct_vs_incorrect.png"
    fig.savefig(correct_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_figures["correct_vs_incorrect"] = correct_file

    logger.info("Saved %d PFS visualizations to %s", len(saved_figures), figures_dir)

    return saved_figures


def save_low_pfs_gradcam_samples(
    detailed_results: List[Dict],
    output_dir: Union[str, Path],
    threshold: float,
    max_samples: int = 20,
) -> int:
    """Save GradCAM visualizations for low PFS samples.

    Args:
        detailed_results: List of detailed results with heatmaps
        output_dir: Directory to save samples
        threshold: PFS threshold
        max_samples: Maximum number of samples to save

    Returns:
        Number of samples saved
    """
    from src_v2.visualization.gradcam import overlay_heatmap
    import cv2

    output_path = Path(output_dir) / "low_pfs_samples"
    output_path.mkdir(parents=True, exist_ok=True)

    # Filter and sort by PFS (ascending)
    low_pfs = [r for r in detailed_results if r["pfs"] < threshold]
    low_pfs.sort(key=lambda x: x["pfs"])

    saved_count = 0
    for item in low_pfs[:max_samples]:
        try:
            result = item["result"]
            heatmap = item["heatmap"]

            # Load original image
            img = np.array(Image.open(result.image_path).convert("RGB"))

            # Create overlay
            overlay = overlay_heatmap(img, heatmap, alpha=0.5)

            # Add annotations
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"PFS: {result.pfs:.3f} | {result.true_class} -> {result.predicted_class} ({result.confidence:.1%})"
            color = (0, 255, 0) if result.correct else (0, 0, 255)
            cv2.putText(overlay, text, (10, 25), font, 0.5, color, 1)

            # Save
            filename = f"pfs_{result.pfs:.3f}_{Path(result.image_path).stem}.png"
            cv2.imwrite(str(output_path / filename), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            saved_count += 1

        except Exception as e:
            logger.warning("Error saving sample: %s", str(e))

    logger.info("Saved %d low PFS GradCAM samples to %s", saved_count, output_path)
    return saved_count
