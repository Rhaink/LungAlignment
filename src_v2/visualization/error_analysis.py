"""Error analysis module for classification error investigation.

This module provides:
- ErrorAnalyzer: Class for analyzing classification errors
- Report generation in JSON and CSV formats
- Visualization of error patterns
- Integration with GradCAM for explainability
"""

import csv
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class ErrorDetail:
    """Details of a single classification error."""
    image_path: str
    true_class: str
    predicted_class: str
    true_class_idx: int
    predicted_class_idx: int
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)


@dataclass
class ErrorSummary:
    """Summary statistics of classification errors."""
    total_samples: int
    total_errors: int
    error_rate: float
    errors_by_true_class: Dict[str, int] = field(default_factory=dict)
    errors_by_predicted_class: Dict[str, int] = field(default_factory=dict)
    confusion_pairs: Dict[str, int] = field(default_factory=dict)
    avg_confidence_errors: float = 0.0
    avg_confidence_correct: float = 0.0
    confusion_matrix: List[List[int]] = field(default_factory=list)


class ErrorAnalyzer:
    """Analyzer for classification errors.

    Collects misclassified samples, generates statistics, and creates
    detailed reports for understanding model failure patterns.

    Args:
        class_names: List of class names

    Example:
        >>> analyzer = ErrorAnalyzer(class_names=['COVID', 'Normal', 'Viral_Pneumonia'])
        >>> for batch in dataloader:
        ...     outputs = model(batch['image'])
        ...     analyzer.add_batch(outputs, batch['labels'], batch['paths'])
        >>> summary = analyzer.get_summary()
        >>> analyzer.save_reports(output_dir)
    """

    def __init__(self, class_names: List[str]):
        if not class_names:
            raise ValueError("class_names cannot be empty")
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.errors: List[ErrorDetail] = []
        self.correct: List[Dict[str, Any]] = []
        self._confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def add_prediction(
        self,
        output: torch.Tensor,
        label: int,
        image_path: str,
    ) -> bool:
        """Add a single prediction to the analyzer.

        Args:
            output: Model output logits (num_classes,) or (1, num_classes)
            label: Ground truth label index
            image_path: Path to the image

        Returns:
            True if prediction was correct, False if error

        Raises:
            ValueError: If label is out of valid range.
        """
        # Validate label
        if not (0 <= label < self.num_classes):
            raise ValueError(f"Label {label} out of range [0, {self.num_classes})")

        # Handle batch dimension
        if output.dim() == 2:
            output = output.squeeze(0)

        probs = F.softmax(output, dim=0)
        predicted = output.argmax().item()
        confidence = probs[predicted].item()

        # Update confusion matrix
        self._confusion_matrix[label, predicted] += 1

        # Create probability dict
        prob_dict = {
            self.class_names[i]: float(probs[i].item())
            for i in range(self.num_classes)
        }

        if predicted != label:
            error = ErrorDetail(
                image_path=str(image_path),
                true_class=self.class_names[label],
                predicted_class=self.class_names[predicted],
                true_class_idx=label,
                predicted_class_idx=predicted,
                confidence=confidence,
                probabilities=prob_dict,
            )
            self.errors.append(error)
            return False
        else:
            self.correct.append({
                "image_path": str(image_path),
                "class": self.class_names[label],
                "confidence": confidence,
            })
            return True

    def add_batch(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        image_paths: List[str],
    ) -> int:
        """Add a batch of predictions.

        Args:
            outputs: Batch of model outputs (B, num_classes)
            labels: Batch of labels (B,)
            image_paths: List of image paths

        Returns:
            Number of errors in batch
        """
        errors_count = 0
        for i in range(outputs.size(0)):
            is_correct = self.add_prediction(
                outputs[i],
                labels[i].item(),
                image_paths[i],
            )
            if not is_correct:
                errors_count += 1
        return errors_count

    def get_summary(self) -> ErrorSummary:
        """Generate error summary statistics.

        Returns:
            ErrorSummary with detailed statistics
        """
        total_samples = len(self.errors) + len(self.correct)
        total_errors = len(self.errors)
        error_rate = total_errors / total_samples if total_samples > 0 else 0.0

        # Count errors by class
        errors_by_true = Counter(e.true_class for e in self.errors)
        errors_by_pred = Counter(e.predicted_class for e in self.errors)

        # Confusion pairs
        confusion_pairs = Counter(
            f"{e.true_class}->{e.predicted_class}"
            for e in self.errors
        )

        # Average confidences
        avg_conf_errors = (
            np.mean([e.confidence for e in self.errors])
            if self.errors else 0.0
        )
        avg_conf_correct = (
            np.mean([c["confidence"] for c in self.correct])
            if self.correct else 0.0
        )

        return ErrorSummary(
            total_samples=total_samples,
            total_errors=total_errors,
            error_rate=error_rate,
            errors_by_true_class=dict(errors_by_true),
            errors_by_predicted_class=dict(errors_by_pred),
            confusion_pairs=dict(confusion_pairs),
            avg_confidence_errors=float(avg_conf_errors),
            avg_confidence_correct=float(avg_conf_correct),
            confusion_matrix=self._confusion_matrix.tolist(),
        )

    def get_top_errors(
        self,
        k: int = 20,
        by: str = "confidence",
        descending: bool = True,
    ) -> List[ErrorDetail]:
        """Get top-K errors sorted by confidence.

        Args:
            k: Number of errors to return
            by: Sort key ('confidence')
            descending: Sort order (True = highest confidence first)

        Returns:
            List of top-K error details
        """
        sorted_errors = sorted(
            self.errors,
            key=lambda e: e.confidence,
            reverse=descending,
        )
        return sorted_errors[:k]

    def get_errors_by_pair(self, true_class: str, predicted_class: str) -> List[ErrorDetail]:
        """Get all errors for a specific confusion pair.

        Args:
            true_class: True class name
            predicted_class: Predicted class name

        Returns:
            List of errors matching the pair
        """
        return [
            e for e in self.errors
            if e.true_class == true_class and e.predicted_class == predicted_class
        ]

    def save_reports(
        self,
        output_dir: Path,
        save_json: bool = True,
        save_csv: bool = True,
    ) -> Dict[str, Path]:
        """Save error reports to files.

        Args:
            output_dir: Output directory
            save_json: Whether to save JSON summary
            save_csv: Whether to save CSV details

        Returns:
            Dict mapping report type to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}
        summary = self.get_summary()

        # Save JSON summary
        if save_json:
            json_path = output_dir / "error_summary.json"
            summary_dict = asdict(summary)
            summary_dict["class_names"] = self.class_names

            # Add per-class analysis
            class_analysis = {}
            for class_name in self.class_names:
                class_errors = [e for e in self.errors if e.true_class == class_name]
                class_total = self._confusion_matrix[self.class_names.index(class_name)].sum()
                class_analysis[class_name] = {
                    "total_samples": int(class_total),
                    "total_errors": len(class_errors),
                    "error_rate": len(class_errors) / class_total if class_total > 0 else 0.0,
                    "misclassified_as": dict(Counter(e.predicted_class for e in class_errors)),
                }
            summary_dict["per_class_analysis"] = class_analysis

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary_dict, f, indent=2, ensure_ascii=False)
            saved_files["json_summary"] = json_path
            logger.info("Saved error summary: %s", json_path)

        # Save CSV details
        if save_csv:
            csv_path = output_dir / "error_details.csv"
            fieldnames = [
                "image_path", "true_class", "predicted_class",
                "confidence", "true_class_idx", "predicted_class_idx",
            ] + [f"prob_{c}" for c in self.class_names]

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for error in self.errors:
                    row = {
                        "image_path": error.image_path,
                        "true_class": error.true_class,
                        "predicted_class": error.predicted_class,
                        "confidence": error.confidence,
                        "true_class_idx": error.true_class_idx,
                        "predicted_class_idx": error.predicted_class_idx,
                    }
                    for class_name in self.class_names:
                        row[f"prob_{class_name}"] = error.probabilities.get(class_name, 0.0)
                    writer.writerow(row)
            saved_files["csv_details"] = csv_path
            logger.info("Saved error details: %s", csv_path)

        # Save confusion analysis JSON
        confusion_path = output_dir / "confusion_analysis.json"
        confusion_analysis = {
            "confusion_matrix": self._confusion_matrix.tolist(),
            "class_names": self.class_names,
            "confusion_pairs": {},
        }
        for pair, count in summary.confusion_pairs.items():
            true_class, pred_class = pair.split("->")
            if true_class not in confusion_analysis["confusion_pairs"]:
                confusion_analysis["confusion_pairs"][true_class] = {}
            confusion_analysis["confusion_pairs"][true_class][pred_class] = count

        with open(confusion_path, "w", encoding="utf-8") as f:
            json.dump(confusion_analysis, f, indent=2, ensure_ascii=False)
        saved_files["confusion_analysis"] = confusion_path
        logger.info("Saved confusion analysis: %s", confusion_path)

        return saved_files


def analyze_classification_errors(
    model: torch.nn.Module,
    dataloader: DataLoader,
    class_names: List[str],
    device: torch.device,
    output_dir: Optional[Path] = None,
) -> Tuple[ErrorAnalyzer, ErrorSummary]:
    """Analyze classification errors for a model on a dataset.

    Args:
        model: Trained classifier model
        dataloader: DataLoader with test data
        class_names: List of class names
        device: Torch device
        output_dir: Optional directory to save reports

    Returns:
        Tuple of (ErrorAnalyzer, ErrorSummary)
    """
    analyzer = ErrorAnalyzer(class_names)
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            paths = batch.get("path", [f"image_{i}" for i in range(len(labels))])

            outputs = model(images)
            analyzer.add_batch(outputs, labels, paths)

    summary = analyzer.get_summary()

    if output_dir:
        analyzer.save_reports(output_dir)

    return analyzer, summary


def create_error_visualizations(
    analyzer: ErrorAnalyzer,
    output_dir: Path,
    copy_images: bool = False,
) -> Dict[str, Path]:
    """Create visualization figures for error analysis.

    Args:
        analyzer: ErrorAnalyzer with collected errors
        output_dir: Output directory for figures
        copy_images: Whether to copy misclassified images

    Returns:
        Dict mapping figure type to file path
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved_figures = {}
    summary = analyzer.get_summary()

    # 1. Error distribution by true class
    fig, ax = plt.subplots(figsize=(10, 6))
    classes = list(summary.errors_by_true_class.keys())
    counts = [summary.errors_by_true_class.get(c, 0) for c in classes]
    bars = ax.bar(classes, counts, color=["#E74C3C", "#27AE60", "#3498DB"][:len(classes)])
    ax.set_xlabel("True Class")
    ax.set_ylabel("Number of Errors")
    ax.set_title("Error Distribution by True Class")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    dist_path = figures_dir / "error_distribution.png"
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    saved_figures["error_distribution"] = dist_path

    # 2. Confidence histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    error_confs = [e.confidence for e in analyzer.errors]
    correct_confs = [c["confidence"] for c in analyzer.correct]
    if error_confs:
        ax.hist(error_confs, bins=20, alpha=0.7, label=f"Errors (n={len(error_confs)})",
                color="#E74C3C")
    if correct_confs:
        ax.hist(correct_confs, bins=20, alpha=0.7, label=f"Correct (n={len(correct_confs)})",
                color="#27AE60")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution: Errors vs Correct")
    ax.legend()
    plt.tight_layout()
    hist_path = figures_dir / "confidence_histogram.png"
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close()
    saved_figures["confidence_histogram"] = hist_path

    # 3. Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = np.array(summary.confusion_matrix)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(analyzer.class_names)))
    ax.set_yticks(range(len(analyzer.class_names)))
    ax.set_xticklabels(analyzer.class_names, rotation=45, ha="right")
    ax.set_yticklabels(analyzer.class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=text_color)

    plt.colorbar(im)
    plt.tight_layout()
    cm_path = figures_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    saved_figures["confusion_matrix"] = cm_path

    # Copy misclassified images if requested
    if copy_images:
        import shutil
        misclassified_dir = figures_dir / "misclassified"
        for error in analyzer.errors:
            class_dir = misclassified_dir / error.true_class
            class_dir.mkdir(parents=True, exist_ok=True)
            src_path = Path(error.image_path)
            if src_path.exists():
                dst_name = f"{src_path.stem}_pred_{error.predicted_class}_{error.confidence:.2f}{src_path.suffix}"
                shutil.copy(src_path, class_dir / dst_name)
        saved_figures["misclassified_dir"] = misclassified_dir

    return saved_figures
