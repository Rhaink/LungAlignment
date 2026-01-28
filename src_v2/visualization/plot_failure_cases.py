"""Generate failure case panels from misclassified test samples."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

from src_v2.constants import DEFAULT_IMAGE_SIZE
from src_v2.models import create_classifier, get_classifier_transforms

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("outputs/warped_lung_best/session_warping")
DEFAULT_DPI = 300
DEFAULT_PANEL_PX = 224
DEFAULT_BAR_HEIGHT_RATIO = 0.28


@dataclass(frozen=True)
class FailureCase:
    """Container for a single misclassified sample."""

    image_path: Path
    image_name: str
    true_class: str
    predicted_class: str
    probabilities: List[float]
    confidence: float
    margin: float

    @property
    def pair(self) -> str:
        return f"{self.true_class}->{self.predicted_class}"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Plot representative misclassified cases for the test split."
    )
    parser.add_argument(
        "--classifier",
        required=True,
        help="Path to classifier checkpoint (.pt) or directory containing it.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Dataset directory with train/val/test splits.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path for the failure cases figure.",
    )
    parser.add_argument(
        "--pairs",
        default=None,
        help="Comma-separated confusion pairs (e.g., COVID->Normal,...).",
    )
    parser.add_argument(
        "--cases-per-pair",
        type=int,
        default=2,
        help="Number of samples to show per confusion pair.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["confidence", "uncertainty", "random"],
        default="confidence",
        help="How to select cases inside each pair.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for selection.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cuda, cpu, mps.",
    )
    parser.add_argument(
        "--landmarks",
        type=Path,
        default=None,
        help="Path to landmarks.json. Defaults to <data-dir>/<split>/landmarks.json.",
    )
    parser.add_argument(
        "--no-landmarks",
        action="store_true",
        help="Disable landmark overlay.",
    )
    parser.add_argument(
        "--sahs",
        dest="sahs",
        action="store_true",
        help="Apply SAHS contrast enhancement.",
    )
    parser.add_argument(
        "--no-sahs",
        dest="sahs",
        action="store_false",
        help="Disable SAHS contrast enhancement.",
    )
    parser.set_defaults(sahs=True)
    parser.add_argument(
        "--sahs-threshold",
        type=int,
        default=10,
        help="Threshold for lung mask in SAHS.",
    )
    parser.add_argument(
        "--panel-px",
        type=int,
        default=DEFAULT_PANEL_PX,
        help="Target panel pixel size (controls figure size).",
    )
    parser.add_argument(
        "--bar-height-ratio",
        type=float,
        default=DEFAULT_BAR_HEIGHT_RATIO,
        help="Relative height of the probability bar row.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Output DPI for the PNG image.",
    )
    parser.add_argument(
        "--arrow-mode",
        choices=["auto", "none", "manual"],
        default="auto",
        help="Arrow placement mode.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="JSON file with manual arrow annotations.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional JSON output with selected case metadata.",
    )
    parser.add_argument(
        "--intensity-bar",
        action="store_true",
        help="Show grayscale intensity bar overlay on each panel.",
    )
    parser.add_argument(
        "--scale-bar",
        action="store_true",
        help="Show pixel scale bar overlay on each panel.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    """Resolve the torch device."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def resolve_checkpoint(checkpoint: Path) -> Path:
    """Resolve checkpoint path from a file or directory."""
    if checkpoint.is_file():
        return checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint}")

    candidates = [
        checkpoint / "best_classifier.pt",
        checkpoint / "best.pt",
        checkpoint / "model.pt",
        checkpoint / "checkpoint.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No checkpoint file found in {checkpoint}. "
        f"Tried: {', '.join(str(p.name) for p in candidates)}"
    )


def load_landmarks(landmarks_path: Path) -> Dict[str, np.ndarray]:
    """Load landmark coordinates keyed by image name."""
    if not landmarks_path.exists():
        logger.warning("Landmarks file not found: %s", landmarks_path)
        return {}
    with open(landmarks_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    landmarks = {}
    for entry in data:
        image_name = entry.get("image_name")
        coords = entry.get("landmarks")
        if image_name and coords:
            landmarks[image_name] = np.array(coords, dtype=np.float32)
    return landmarks


def load_annotations(path: Optional[Path]) -> Dict[str, Dict[str, List[float]]]:
    """Load manual arrow annotations keyed by image name."""
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        mapping: Dict[str, Dict[str, List[float]]] = {}
        for entry in data:
            image_name = entry.get("image_name")
            if image_name:
                mapping[image_name] = {
                    "arrow_tip": entry.get("arrow_tip", []),
                    "arrow_tail": entry.get("arrow_tail", []),
                }
        return mapping
    raise ValueError("Annotations JSON must be a dict or list.")


def extract_image_name(image_path: Path) -> str:
    """Extract base image name from warped filename."""
    stem = image_path.stem
    if stem.endswith("_warped"):
        return stem[: -len("_warped")]
    return stem


def apply_sahs_masked(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Apply SAHS contrast enhancement to lung region only."""
    if image.ndim != 2:
        raise ValueError("SAHS expects a grayscale image array.")

    mask = image > threshold
    lung_pixels = image[mask].astype(np.float64)
    if lung_pixels.size == 0:
        return image.copy()

    gray_mean = float(np.mean(lung_pixels))
    above_mean = lung_pixels[lung_pixels > gray_mean]
    below_or_equal = lung_pixels[lung_pixels <= gray_mean]

    max_value = gray_mean
    min_value = gray_mean

    if above_mean.size > 0:
        std_above = float(np.sqrt(np.mean((above_mean - gray_mean) ** 2)))
        max_value = gray_mean + 2.5 * std_above
    if below_or_equal.size > 0:
        std_below = float(np.sqrt(np.mean((below_or_equal - gray_mean) ** 2)))
        min_value = gray_mean - 2.0 * std_below

    enhanced = np.zeros_like(image)
    if max_value != min_value:
        transformed = (255.0 / (max_value - min_value)) * (image.astype(np.float64) - min_value)
        transformed = np.clip(transformed, 0, 255).astype(np.uint8)
        enhanced[mask] = transformed[mask]
    else:
        enhanced[mask] = image[mask]

    return enhanced


def remap_labels(
    labels: np.ndarray,
    dataset_classes: Sequence[str],
    class_names: Sequence[str],
) -> np.ndarray:
    """Align dataset label indices to checkpoint class name ordering."""
    if list(dataset_classes) == list(class_names):
        return labels

    mapping: Dict[int, int] = {}
    for dataset_idx, name in enumerate(dataset_classes):
        if name in class_names:
            mapping[dataset_idx] = class_names.index(name)

    if len(mapping) != len(class_names):
        missing = [name for name in class_names if name not in dataset_classes]
        raise ValueError(f"Dataset is missing classes: {missing}")

    return np.vectorize(mapping.get)(labels)


def collect_failure_cases(
    model: torch.nn.Module,
    dataloader: DataLoader,
    dataset: datasets.ImageFolder,
    class_names: Sequence[str],
    device: torch.device,
) -> List[FailureCase]:
    """Collect misclassified samples with probabilities."""
    model.eval()
    failure_cases: List[FailureCase] = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            labels_mapped = remap_labels(labels_np, dataset.classes, class_names)

            start_idx = batch_idx * dataloader.batch_size
            end_idx = start_idx + len(labels_np)
            batch_paths = [Path(dataset.samples[i][0]) for i in range(start_idx, end_idx)]

            for idx, image_path in enumerate(batch_paths):
                true_idx = int(labels_mapped[idx])
                pred_idx = int(np.argmax(probs[idx]))
                if true_idx == pred_idx:
                    continue
                sorted_probs = np.sort(probs[idx])
                margin = float(sorted_probs[-1] - sorted_probs[-2])
                failure_cases.append(
                    FailureCase(
                        image_path=image_path,
                        image_name=extract_image_name(image_path),
                        true_class=class_names[true_idx],
                        predicted_class=class_names[pred_idx],
                        probabilities=[float(p) for p in probs[idx]],
                        confidence=float(probs[idx][pred_idx]),
                        margin=margin,
                    )
                )

    return failure_cases


def select_cases_by_pair(
    failure_cases: Sequence[FailureCase],
    pairs: Sequence[str],
    cases_per_pair: int,
    mode: str,
    seed: int,
) -> List[FailureCase]:
    """Select representative cases per confusion pair."""
    rng = random.Random(seed)
    cases_by_pair: Dict[str, List[FailureCase]] = {}
    for case in failure_cases:
        cases_by_pair.setdefault(case.pair, []).append(case)

    selected: List[FailureCase] = []
    for pair in pairs:
        candidates = cases_by_pair.get(pair, [])
        if len(candidates) < cases_per_pair:
            raise ValueError(
                f"Not enough cases for pair {pair}: "
                f"needed {cases_per_pair}, found {len(candidates)}"
            )
        if mode == "confidence":
            candidates = sorted(candidates, key=lambda c: c.confidence, reverse=True)
        elif mode == "uncertainty":
            candidates = sorted(candidates, key=lambda c: c.margin)
        elif mode == "random":
            candidates = candidates[:]
            rng.shuffle(candidates)
        selected.extend(candidates[:cases_per_pair])

    return selected


def get_display_name(name: str) -> str:
    """Format class names for display."""
    if name.upper() == "COVID":
        return "COVID-19"
    if name == "Viral_Pneumonia":
        return "Viral Pneumonia"
    return name


def get_short_label(name: str) -> str:
    """Shorten class names for compact labels."""
    if name.upper() == "COVID":
        return "COVID"
    if name == "Viral_Pneumonia":
        return "Viral"
    return name


def compute_auto_arrow(
    image: np.ndarray,
    threshold: int = 10,
    border: int = 10,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute an arrow tip and tail based on gradient magnitude."""
    mask = image > threshold
    if not np.any(mask):
        center = (image.shape[1] / 2.0, image.shape[0] / 2.0)
        return center, (center[0] - 20, center[1] - 20)

    gy, gx = np.gradient(image.astype(np.float32))
    magnitude = np.hypot(gx, gy)
    magnitude[~mask] = 0.0
    if border > 0:
        magnitude[:border, :] = 0.0
        magnitude[-border:, :] = 0.0
        magnitude[:, :border] = 0.0
        magnitude[:, -border:] = 0.0

    y_idx, x_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    tail_x = float(max(x_idx - 25, 0))
    tail_y = float(max(y_idx - 25, 0))
    return (float(x_idx), float(y_idx)), (tail_x, tail_y)


def draw_probability_bars(
    ax: plt.Axes,
    class_names: Sequence[str],
    probabilities: Sequence[float],
    true_class: str,
    predicted_class: str,
) -> None:
    """Draw probability bars in a dedicated axis."""
    ax.set_facecolor("white")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.5, len(class_names) - 0.5)
    ax.invert_yaxis()
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"], fontsize=5)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2, pad=1)

    colors = []
    for name in class_names:
        if name == true_class:
            colors.append("#2ecc71")
        elif name == predicted_class:
            colors.append("#e74c3c")
        else:
            colors.append("#bdc3c7")

    bars = ax.barh(
        np.arange(len(class_names)),
        probabilities,
        color=colors,
        edgecolor="none",
        height=0.6,
    )
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels([get_short_label(n) for n in class_names], fontsize=6)
    ax.grid(axis="x", color="0.85", linewidth=0.4)
    ax.set_axisbelow(True)

    for idx, bar in enumerate(bars):
        prob = probabilities[idx]
        x_pos = min(bar.get_width() + 0.02, 0.98)
        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2.0,
            f"{prob:.2f}",
            va="center",
            ha="left",
            fontsize=5.5,
            color="black",
        )

    ax.text(
        0.0,
        1.05,
        f"True: {get_display_name(true_class)} | Pred: {get_display_name(predicted_class)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6,
        color="black",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)


def add_intensity_bar(ax: plt.Axes) -> None:
    """Add a grayscale intensity bar inset."""
    bar_ax = ax.inset_axes([0.58, 0.88, 0.38, 0.06])
    gradient = np.linspace(0, 255, 256, dtype=np.uint8)[None, :]
    bar_ax.imshow(gradient, cmap="gray", aspect="auto", vmin=0, vmax=255)
    bar_ax.set_yticks([])
    bar_ax.set_xticks([0, 255])
    bar_ax.set_xticklabels(["0", "255"], fontsize=4, color="white")
    bar_ax.tick_params(axis="x", colors="white", length=0)
    for spine in bar_ax.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(0.5)


def add_scale_bar(ax: plt.Axes, width: int, height: int, scale_px: Optional[int]) -> None:
    """Add a simple pixel scale bar."""
    if scale_px is None:
        scale_px = max(10, int(round(width * 0.2)))
    x_start = int(round(width * 0.05))
    y_start = int(round(height * 0.92))
    ax.plot(
        [x_start, x_start + scale_px],
        [y_start, y_start],
        color="white",
        linewidth=2,
    )
    ax.text(
        x_start,
        y_start - 5,
        f"{scale_px}px",
        color="white",
        fontsize=5,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=1),
    )


def apply_publication_style() -> None:
    """Apply a compact, publication-friendly matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
        }
    )


def create_failure_case_figure(
    cases: Sequence[FailureCase],
    class_names: Sequence[str],
    landmarks: Dict[str, np.ndarray],
    output_path: Path,
    panel_px: int,
    dpi: int,
    bar_height_ratio: float,
    sahs: bool,
    sahs_threshold: int,
    show_landmarks: bool,
    arrow_mode: str,
    annotations: Dict[str, Dict[str, List[float]]],
    show_intensity_bar: bool,
    show_scale_bar: bool,
) -> None:
    """Create the failure cases figure."""
    apply_publication_style()
    cols = 3
    rows = int(math.ceil(len(cases) / cols))
    panel_height_px = panel_px * (1.0 + bar_height_ratio)
    fig_width = cols * panel_px / dpi
    fig_height = rows * panel_height_px / dpi

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, layout="constrained")
    grid = fig.add_gridspec(rows, cols, wspace=0.08, hspace=0.15)

    panel_labels = [f"({chr(97 + i)})" for i in range(len(cases))]
    for idx, (case, panel_label) in enumerate(zip(cases, panel_labels)):
        row_idx = idx // cols
        col_idx = idx % cols
        subgrid = grid[row_idx, col_idx].subgridspec(
            2,
            1,
            height_ratios=[1.0, bar_height_ratio],
            hspace=0.05,
        )
        ax_img = fig.add_subplot(subgrid[0])
        ax_bar = fig.add_subplot(subgrid[1])
        image = np.array(Image.open(case.image_path).convert("L"))
        if sahs:
            image = apply_sahs_masked(image, threshold=sahs_threshold)

        ax_img.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax_img.axis("off")

        if show_landmarks and case.image_name in landmarks:
            coords = landmarks[case.image_name]
            ax_img.scatter(coords[:, 0], coords[:, 1], c="#ff2d2d", s=10, marker="o")
        elif show_landmarks:
            logger.warning("Landmarks missing for %s", case.image_name)

        if arrow_mode != "none":
            arrow_tip = None
            arrow_tail = None
            if arrow_mode == "manual":
                entry = annotations.get(case.image_name, {})
                tip = entry.get("arrow_tip", [])
                tail = entry.get("arrow_tail", [])
                if len(tip) == 2 and len(tail) == 2:
                    arrow_tip = (float(tip[0]), float(tip[1]))
                    arrow_tail = (float(tail[0]), float(tail[1]))
            if arrow_tip is None or arrow_tail is None:
                arrow_tip, arrow_tail = compute_auto_arrow(image, threshold=sahs_threshold)
            ax_img.annotate(
                "",
                xy=arrow_tip,
                xytext=arrow_tail,
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
            )

        if show_intensity_bar:
            add_intensity_bar(ax_img)
        if show_scale_bar:
            add_scale_bar(ax_img, image.shape[1], image.shape[0], scale_px=None)

        ax_img.text(
            0.02,
            0.98,
            panel_label,
            transform=ax_img.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="white",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1),
        )

        draw_probability_bars(
            ax_bar,
            class_names=class_names,
            probabilities=case.probabilities,
            true_class=case.true_class,
            predicted_class=case.predicted_class,
        )
        ax_bar.set_xlabel("Probability", fontsize=5, labelpad=1)

    empty_slots = (rows * cols) - len(cases)
    if empty_slots > 0:
        for idx in range(empty_slots):
            row_idx = (len(cases) + idx) // cols
            col_idx = (len(cases) + idx) % cols
            subgrid = grid[row_idx, col_idx].subgridspec(
                2,
                1,
                height_ratios=[1.0, bar_height_ratio],
                hspace=0.05,
            )
            ax_img = fig.add_subplot(subgrid[0])
            ax_bar = fig.add_subplot(subgrid[1])
            ax_img.axis("off")
            ax_bar.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    """Run the failure case plotting routine."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    checkpoint_path = resolve_checkpoint(Path(args.classifier))
    device = resolve_device(args.device)
    logger.info("Using checkpoint: %s", checkpoint_path)
    logger.info("Device: %s", device)

    model = create_classifier(checkpoint=str(checkpoint_path), device=device)
    ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = ckpt_data.get("class_names", None)
    if class_names is None:
        raise ValueError("Checkpoint missing class_names metadata.")

    data_dir = args.data_dir / args.split
    if not data_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {data_dir}")

    transform = get_classifier_transforms(train=False, img_size=DEFAULT_IMAGE_SIZE)
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    failure_cases = collect_failure_cases(model, dataloader, dataset, class_names, device)
    logger.info("Total misclassified samples: %d", len(failure_cases))

    if args.pairs:
        pairs = [pair.strip() for pair in args.pairs.split(",") if pair.strip()]
    else:
        pair_counts: Dict[str, int] = {}
        for case in failure_cases:
            pair_counts[case.pair] = pair_counts.get(case.pair, 0) + 1
        pairs = sorted(pair_counts, key=pair_counts.get, reverse=True)[:3]
        logger.info("Using top confusion pairs: %s", ", ".join(pairs))

    selected_cases = select_cases_by_pair(
        failure_cases,
        pairs=pairs,
        cases_per_pair=args.cases_per_pair,
        mode=args.selection_mode,
        seed=args.seed,
    )

    landmarks_path = args.landmarks or (args.data_dir / args.split / "landmarks.json")
    landmarks = load_landmarks(landmarks_path)
    annotations = load_annotations(args.annotations) if args.arrow_mode == "manual" else {}

    create_failure_case_figure(
        cases=selected_cases,
        class_names=class_names,
        landmarks=landmarks,
        output_path=args.output,
        panel_px=args.panel_px,
        dpi=args.dpi,
        bar_height_ratio=args.bar_height_ratio,
        sahs=args.sahs,
        sahs_threshold=args.sahs_threshold,
        show_landmarks=not args.no_landmarks,
        arrow_mode=args.arrow_mode,
        annotations=annotations,
        show_intensity_bar=args.intensity_bar,
        show_scale_bar=args.scale_bar,
    )

    if args.metadata:
        cases_data = []
        for case in selected_cases:
            entry = asdict(case)
            entry["image_path"] = str(case.image_path)
            cases_data.append(entry)
        metadata = {
            "pairs": pairs,
            "cases_per_pair": args.cases_per_pair,
            "selection_mode": args.selection_mode,
            "cases": cases_data,
        }
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        with open(args.metadata, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)

    logger.info("Saved failure cases figure to %s", args.output)


if __name__ == "__main__":
    main()
