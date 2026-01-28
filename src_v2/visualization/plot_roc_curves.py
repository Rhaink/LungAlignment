"""Generate ROC curves for a multiclass classifier using one-vs-rest strategy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets

from src_v2.constants import DEFAULT_IMAGE_SIZE
from src_v2.models import create_classifier, get_classifier_transforms


DEFAULT_DATA_DIR = Path("outputs/warped_lung_best/session_warping")
DEFAULT_DPI = 600
DEFAULT_WIDTH_CM = 12.0
DEFAULT_HEIGHT_CM = 4.0


@dataclass(frozen=True)
class RocPlotConfig:
    """Configuration for ROC plot styling."""

    titles: Sequence[str]
    colors: Sequence[str]
    auc_fontsize: float = 6.0
    title_fontsize: float = 7.2
    line_width: float = 1.6
    fill_alpha: float = 0.0
    grid_alpha: float = 0.0
    grid_color: str = "0.85"
    grid_linewidth: float = 0.4
    baseline_color: str = "0.5"
    baseline_style: str = "--"
    baseline_linewidth: float = 0.8


def apply_publication_style() -> None:
    """Apply a compact, publication-friendly matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7,
            "axes.titlesize": 7.2,
            "axes.labelsize": 7,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
        }
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Plot ROC curves for a multiclass classifier (one-vs-rest)."
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
        choices=["train", "val", "test", "all"],
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path for the ROC figure.",
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
        "--tta",
        action="store_true",
        help="Enable test-time augmentation (horizontal flip).",
    )
    parser.add_argument(
        "--width-cm",
        type=float,
        default=DEFAULT_WIDTH_CM,
        help="Figure width in centimeters.",
    )
    parser.add_argument(
        "--height-cm",
        type=float,
        default=DEFAULT_HEIGHT_CM,
        help="Figure height in centimeters.",
    )
    parser.add_argument(
        "--layout",
        choices=["row", "grid"],
        default="row",
        help="Panel layout: row (1x3) or grid (2x2 with summary panel).",
    )
    parser.add_argument(
        "--summary-panel",
        action="store_true",
        help="Include micro/macro AUC summary in the extra grid panel.",
    )
    parser.add_argument(
        "--inset-zoom",
        action="store_true",
        help="Add a zoomed inset (low FPR, high TPR) in each ROC panel.",
    )
    parser.add_argument(
        "--inset-xmax",
        type=float,
        default=0.05,
        help="Inset maximum FPR (x-axis) when inset-zoom is enabled.",
    )
    parser.add_argument(
        "--inset-ymin",
        type=float,
        default=0.95,
        help="Inset minimum TPR (y-axis) when inset-zoom is enabled.",
    )
    parser.add_argument(
        "--log-fpr",
        action="store_true",
        help="Use logarithmic scale on the FPR axis.",
    )
    parser.add_argument(
        "--log-fpr-min",
        type=float,
        default=1e-4,
        help="Minimum FPR shown when log-fpr is enabled.",
    )
    parser.add_argument(
        "--backend",
        choices=["manual", "sklearn", "plotly"],
        default="manual",
        help="Plotting backend: manual, scikit-learn, or plotly.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Output DPI for the PNG image.",
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


def load_dataset(data_dir: Path, split: str, transform) -> Tuple[Iterable, List[str]]:
    """Load dataset split for evaluation."""
    if split == "all":
        datasets_list = []
        for subset in ["train", "val", "test"]:
            subset_dir = data_dir / subset
            if subset_dir.exists():
                datasets_list.append(datasets.ImageFolder(subset_dir, transform=transform))
        if not datasets_list:
            raise FileNotFoundError(f"No valid split directories found in {data_dir}")
        dataset = ConcatDataset(datasets_list)
        class_names = datasets_list[0].classes
        return dataset, class_names

    split_dir = data_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split '{split}' does not exist in {data_dir}")
    dataset = datasets.ImageFolder(split_dir, transform=transform)
    return dataset, dataset.classes


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
        if name not in class_names:
            continue
        mapping[dataset_idx] = class_names.index(name)

    if len(mapping) != len(class_names):
        missing = [name for name in class_names if name not in dataset_classes]
        raise ValueError(f"Dataset is missing classes: {missing}")

    return np.vectorize(mapping.get)(labels)


def collect_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    tta: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect labels and probabilities for the dataset."""
    model.eval()
    all_labels: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if tta:
                flipped = torch.flip(inputs, dims=[3])
                outputs = (outputs + model(flipped)) / 2.0
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_probs)


def compute_roc_curves(
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    """Compute ROC curves and AUC values for each class and micro/macro averages."""
    n_classes = len(class_names)
    labels_bin = label_binarize(labels, classes=list(range(n_classes)))

    fpr: Dict[str, np.ndarray] = {}
    tpr: Dict[str, np.ndarray] = {}
    roc_auc: Dict[str, float] = {}

    for idx, name in enumerate(class_names):
        if labels_bin[:, idx].sum() == 0:
            raise ValueError(f"No positive samples for class '{name}' in labels.")
        fpr[name], tpr[name], _ = roc_curve(labels_bin[:, idx], probs[:, idx])
        roc_auc[name] = float(auc(fpr[name], tpr[name]))

    fpr["micro"], tpr["micro"], _ = roc_curve(labels_bin.ravel(), probs.ravel())
    roc_auc["micro"] = float(auc(fpr["micro"], tpr["micro"]))
    roc_auc["macro"] = float(np.mean([roc_auc[name] for name in class_names]))

    return fpr, tpr, roc_auc


def cm_to_inch(value_cm: float) -> float:
    """Convert centimeters to inches."""
    return value_cm / 2.54


def get_log_fpr_values(values: np.ndarray, fpr_min: float) -> np.ndarray:
    """Floor FPR values for log-scale plotting."""
    return np.maximum(values, fpr_min)


def apply_log_fpr_axis(ax: plt.Axes, fpr_min: float) -> None:
    """Apply log scaling and ticks to the FPR axis."""
    ticks = [fpr_min, 1e-3, 1e-2, 1e-1, 1.0]
    tick_labels = [f"{tick:.0e}" if tick < 1 else "1" for tick in ticks]
    ax.set_xscale("log")
    ax.set_xlim(fpr_min, 1.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)


def add_zoom_inset(
    ax: plt.Axes,
    fpr: np.ndarray,
    tpr: np.ndarray,
    color: str,
    fpr_max: float,
    tpr_min: float,
    line_width: float,
) -> None:
    """Add a zoomed inset focusing on low FPR / high TPR region."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    inset_ax = inset_axes(
        ax,
        width="40%",
        height="40%",
        loc="center right",
        borderpad=0.7,
    )
    inset_ax.plot(fpr, tpr, color=color, linewidth=line_width)
    inset_ax.set_xlim(0.0, fpr_max)
    inset_ax.set_ylim(tpr_min, 1.0)
    inset_ax.set_xticks([0.0, fpr_max])
    inset_ax.set_yticks([tpr_min, 1.0])
    inset_ax.tick_params(labelsize=6, pad=1)
    inset_ax.set_facecolor("white")
    for spine in inset_ax.spines.values():
        spine.set_linewidth(0.6)
    mark_inset(
        ax,
        inset_ax,
        loc1=2,
        loc2=3,
        fc="none",
        ec="0.6",
        lw=0.5,
        linestyle=":",
        alpha=0.8,
    )


def plot_roc_curves(
    fpr: Dict[str, np.ndarray],
    tpr: Dict[str, np.ndarray],
    roc_auc: Dict[str, float],
    class_names: Sequence[str],
    output_path: Path,
    width_cm: float,
    height_cm: float,
    dpi: int,
    layout: str,
    summary_panel: bool,
    inset_zoom: bool,
    inset_xmax: float,
    inset_ymin: float,
    log_fpr: bool,
    log_fpr_min: float,
) -> None:
    """Plot ROC curves in row or grid layout and save to disk."""
    apply_publication_style()

    config = RocPlotConfig(
        titles=["COVID-19", "Normal", "Viral Pneumonia"],
        colors=["#1f77b4", "#2ca02c", "#ff7f0e"],
    )

    if layout == "grid":
        fig, axes = plt.subplots(
            2,
            2,
            sharex=True,
            sharey=True,
            figsize=(cm_to_inch(width_cm), cm_to_inch(height_cm)),
            dpi=dpi,
        )
        plot_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
        summary_ax = axes[1, 1]
    else:
        fig, axes = plt.subplots(
            1,
            3,
            sharex=True,
            sharey=True,
            figsize=(cm_to_inch(width_cm), cm_to_inch(height_cm)),
            dpi=dpi,
        )
        plot_axes = list(axes)
        summary_ax = None

    ticks = np.linspace(0.0, 1.0, 6)
    panel_labels = ["(a)", "(b)", "(c)"]
    for idx, (ax, name, title, color, panel) in enumerate(
        zip(plot_axes, class_names, config.titles, config.colors, panel_labels)
    ):
        plot_fpr = (
            get_log_fpr_values(fpr[name], log_fpr_min) if log_fpr else fpr[name]
        )
        ax.plot(
            plot_fpr,
            tpr[name],
            color=color,
            linewidth=config.line_width,
            zorder=3,
            solid_capstyle="round",
        )
        if config.fill_alpha > 0:
            ax.fill_between(
                fpr[name],
                tpr[name],
                color=color,
                alpha=config.fill_alpha,
            )
        ax.set_title(title, fontsize=config.title_fontsize, pad=2)
        if log_fpr:
            apply_log_fpr_axis(ax, log_fpr_min)
        else:
            ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        if not log_fpr:
            ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlabel("")
        ax.set_ylabel("")
        auc_x, auc_y, auc_ha = (0.05, 0.08, "left") if inset_zoom else (0.97, 0.07, "right")
        ax.text(
            auc_x,
            auc_y,
            f"AUC = {roc_auc[name]:.3f}",
            transform=ax.transAxes,
            ha=auc_ha,
            va="bottom",
            fontsize=config.auc_fontsize,
            fontweight="normal",
            color="0.2",
        )
        ax.text(
            0.02,
            0.98,
            panel,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            fontweight="bold",
        )
        ax.tick_params(pad=2)
        ax.set_facecolor("white")
        ax.set_aspect("equal", adjustable="box")
        if idx > 0:
            ax.tick_params(labelleft=False)
        for spine in ax.spines.values():
            spine.set_zorder(0)
        if inset_zoom and not log_fpr:
            add_zoom_inset(
                ax=ax,
                fpr=fpr[name],
                tpr=tpr[name],
                color=color,
                fpr_max=inset_xmax,
                tpr_min=inset_ymin,
                line_width=config.line_width,
            )

    fig.patch.set_facecolor("white")
    if layout == "grid":
        fig.subplots_adjust(left=0.09, right=0.995, top=0.92, bottom=0.12, wspace=0.3, hspace=0.32)
        fig.text(0.5, 0.06, "False Positive Rate", ha="center", va="center", fontsize=7)
        fig.text(
            0.015,
            0.5,
            "True Positive Rate",
            ha="center",
            va="center",
            rotation=90,
            fontsize=7,
        )
        if summary_ax is not None:
            summary_ax.axis("off")
            if summary_panel:
                summary_ax.text(
                    0.5,
                    0.62,
                    f"Micro-AUC = {roc_auc['micro']:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="0.2",
                )
                summary_ax.text(
                    0.5,
                    0.42,
                    f"Macro-AUC = {roc_auc['macro']:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="0.2",
                )
    else:
        fig.subplots_adjust(left=0.09, right=0.995, top=0.84, bottom=0.27, wspace=0.32)
        fig.text(0.5, 0.12, "False Positive Rate", ha="center", va="center", fontsize=7)
        fig.text(
            0.015,
            0.5,
            "True Positive Rate",
            ha="center",
            va="center",
            rotation=90,
            fontsize=7,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def plot_roc_curves_sklearn(
    fpr: Dict[str, np.ndarray],
    tpr: Dict[str, np.ndarray],
    roc_auc: Dict[str, float],
    class_names: Sequence[str],
    output_path: Path,
    width_cm: float,
    height_cm: float,
    dpi: int,
    layout: str,
    summary_panel: bool,
    inset_zoom: bool,
    inset_xmax: float,
    inset_ymin: float,
    log_fpr: bool,
    log_fpr_min: float,
) -> None:
    """Plot ROC curves using scikit-learn's RocCurveDisplay."""
    apply_publication_style()

    config = RocPlotConfig(
        titles=["COVID-19", "Normal", "Viral Pneumonia"],
        colors=["#1f77b4", "#2ca02c", "#ff7f0e"],
    )

    if layout == "grid":
        fig, axes = plt.subplots(
            2,
            2,
            sharex=True,
            sharey=True,
            figsize=(cm_to_inch(width_cm), cm_to_inch(height_cm)),
            dpi=dpi,
        )
        plot_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
        summary_ax = axes[1, 1]
    else:
        fig, axes = plt.subplots(
            1,
            3,
            sharex=True,
            sharey=True,
            figsize=(cm_to_inch(width_cm), cm_to_inch(height_cm)),
            dpi=dpi,
        )
        plot_axes = list(axes)
        summary_ax = None

    ticks = np.linspace(0.0, 1.0, 6)
    panel_labels = ["(a)", "(b)", "(c)"]

    for idx, (ax, name, title, color, panel) in enumerate(
        zip(plot_axes, class_names, config.titles, config.colors, panel_labels)
    ):
        plot_fpr = (
            get_log_fpr_values(fpr[name], log_fpr_min) if log_fpr else fpr[name]
        )
        display = RocCurveDisplay(
            fpr=plot_fpr,
            tpr=tpr[name],
            roc_auc=roc_auc[name],
        )
        display.plot(
            ax=ax,
            name="",
            plot_chance_level=False,
            curve_kwargs={
                "color": color,
                "linewidth": config.line_width,
                "zorder": 3,
            },
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.set_title(title, fontsize=config.title_fontsize, pad=2)
        if log_fpr:
            apply_log_fpr_axis(ax, log_fpr_min)
        else:
            ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        if not log_fpr:
            ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlabel("")
        ax.set_ylabel("")
        auc_x, auc_y, auc_ha = (0.05, 0.08, "left") if inset_zoom else (0.97, 0.07, "right")
        ax.text(
            auc_x,
            auc_y,
            f"AUC = {roc_auc[name]:.3f}",
            transform=ax.transAxes,
            ha=auc_ha,
            va="bottom",
            fontsize=config.auc_fontsize,
            fontweight="normal",
            color="0.2",
        )
        ax.text(
            0.02,
            0.98,
            panel,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            fontweight="bold",
        )
        ax.tick_params(pad=2)
        ax.set_facecolor("white")
        ax.set_aspect("equal", adjustable="box")
        if idx > 0:
            ax.tick_params(labelleft=False)
        for spine in ax.spines.values():
            spine.set_zorder(0)
        if inset_zoom and not log_fpr:
            add_zoom_inset(
                ax=ax,
                fpr=fpr[name],
                tpr=tpr[name],
                color=color,
                fpr_max=inset_xmax,
                tpr_min=inset_ymin,
                line_width=config.line_width,
            )

    fig.patch.set_facecolor("white")
    if layout == "grid":
        fig.subplots_adjust(
            left=0.09, right=0.995, top=0.92, bottom=0.12, wspace=0.3, hspace=0.32
        )
        fig.text(0.5, 0.06, "False Positive Rate", ha="center", va="center", fontsize=7)
        fig.text(
            0.015,
            0.5,
            "True Positive Rate",
            ha="center",
            va="center",
            rotation=90,
            fontsize=7,
        )
        if summary_ax is not None:
            summary_ax.axis("off")
            if summary_panel:
                summary_ax.text(
                    0.5,
                    0.62,
                    f"Micro-AUC = {roc_auc['micro']:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="0.2",
                )
                summary_ax.text(
                    0.5,
                    0.42,
                    f"Macro-AUC = {roc_auc['macro']:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="0.2",
                )
    else:
        fig.subplots_adjust(left=0.09, right=0.995, top=0.84, bottom=0.27, wspace=0.32)
        fig.text(0.5, 0.12, "False Positive Rate", ha="center", va="center", fontsize=7)
        fig.text(
            0.015,
            0.5,
            "True Positive Rate",
            ha="center",
            va="center",
            rotation=90,
            fontsize=7,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def plot_roc_curves_plotly(
    fpr: Dict[str, np.ndarray],
    tpr: Dict[str, np.ndarray],
    roc_auc: Dict[str, float],
    class_names: Sequence[str],
    output_path: Path,
    width_cm: float,
    height_cm: float,
    dpi: int,
    layout: str,
    summary_panel: bool,
    inset_zoom: bool,
    inset_xmax: float,
    inset_ymin: float,
    log_fpr: bool,
    log_fpr_min: float,
) -> None:
    """Plot ROC curves using Plotly for a distinct visual style."""
    del inset_zoom, inset_xmax, inset_ymin, log_fpr, log_fpr_min
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    titles = ["COVID-19", "Normal", "Viral Pneumonia"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    panel_labels = ["(a)", "(b)", "(c)"]

    if layout == "grid":
        rows, cols = 2, 2
        plot_positions = [(1, 1), (1, 2), (2, 1)]
    else:
        rows, cols = 1, 3
        plot_positions = [(1, 1), (1, 2), (1, 3)]

    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.12 if layout == "row" else 0.1,
        vertical_spacing=0.12,
    )

    tickvals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for idx, (row, col) in enumerate(plot_positions):
        name = class_names[idx]
        fig.add_trace(
            go.Scatter(
                x=fpr[name],
                y=tpr[name],
                mode="lines",
                line={"color": colors[idx], "width": 2},
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line={"color": "rgba(120,120,120,0.8)", "width": 1, "dash": "dash"},
            row=row,
            col=col,
        )
        fig.add_annotation(
            x=0.5,
            y=1.08,
            xref="x domain",
            yref="y domain",
            text=titles[idx],
            showarrow=False,
            font={"size": 12},
            row=row,
            col=col,
        )
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="x domain",
            yref="y domain",
            text=panel_labels[idx],
            showarrow=False,
            font={"size": 11, "color": "black"},
            row=row,
            col=col,
        )
        fig.add_annotation(
            x=0.96,
            y=0.06,
            xref="x domain",
            yref="y domain",
            text=f"AUC = {roc_auc[name]:.3f}",
            showarrow=False,
            font={"size": 10, "color": "rgba(0,0,0,0.7)"},
            row=row,
            col=col,
        )
        fig.update_xaxes(
            row=row,
            col=col,
            range=[0, 1],
            tickvals=tickvals,
            tickformat=".1f",
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=1,
            ticks="outside",
            ticklen=4,
        )
        fig.update_yaxes(
            row=row,
            col=col,
            range=[0, 1],
            tickvals=tickvals,
            tickformat=".1f",
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=1,
            ticks="outside",
            ticklen=4,
        )

    if layout == "grid":
        fig.update_xaxes(
            row=2,
            col=2,
            showgrid=False,
            showline=False,
            zeroline=False,
            showticklabels=False,
        )
        fig.update_yaxes(
            row=2,
            col=2,
            showgrid=False,
            showline=False,
            zeroline=False,
            showticklabels=False,
        )
        if summary_panel:
            fig.add_annotation(
                x=0.5,
                y=0.62,
                xref="x domain",
                yref="y domain",
                text=f"Micro-AUC = {roc_auc['micro']:.3f}",
                showarrow=False,
                font={"size": 11, "color": "rgba(0,0,0,0.7)"},
                row=2,
                col=2,
            )
            fig.add_annotation(
                x=0.5,
                y=0.42,
                xref="x domain",
                yref="y domain",
                text=f"Macro-AUC = {roc_auc['macro']:.3f}",
                showarrow=False,
                font={"size": 11, "color": "rgba(0,0,0,0.7)"},
                row=2,
                col=2,
            )

    fig.add_annotation(
        x=0.5,
        y=-0.12,
        xref="paper",
        yref="paper",
        text="False Positive Rate",
        showarrow=False,
        font={"size": 12},
    )
    fig.add_annotation(
        x=-0.06,
        y=0.5,
        xref="paper",
        yref="paper",
        text="True Positive Rate",
        showarrow=False,
        font={"size": 12},
        textangle=-90,
    )

    width_px = int(round(cm_to_inch(width_cm) * dpi))
    height_px = int(round(cm_to_inch(height_cm) * dpi))
    fig.update_layout(
        width=width_px,
        height=height_px,
        margin={"l": 70, "r": 20, "t": 30, "b": 60},
        font={"family": "Times New Roman, Times, serif", "size": 12},
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".html":
        fig.write_html(output_path, include_plotlyjs="cdn")
    else:
        fig.write_image(output_path)


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    checkpoint_path = resolve_checkpoint(Path(args.classifier))
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    device = resolve_device(args.device)
    model = create_classifier(checkpoint=str(checkpoint_path), device=device)
    model.eval()

    ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names = ckpt_data.get("class_names")
    if not class_names:
        raise ValueError("Checkpoint missing class_names; cannot compute ROC curves.")

    eval_transform = get_classifier_transforms(train=False, img_size=DEFAULT_IMAGE_SIZE)
    dataset, dataset_classes = load_dataset(data_dir, args.split, eval_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    labels, probs = collect_predictions(model, dataloader, device, args.tta)
    labels = remap_labels(labels, dataset_classes, class_names)

    fpr, tpr, roc_auc = compute_roc_curves(labels, probs, class_names)
    if args.backend == "sklearn":
        plot_roc_curves_sklearn(
            fpr,
            tpr,
            roc_auc,
            class_names,
            Path(args.output),
            width_cm=args.width_cm,
            height_cm=args.height_cm,
            dpi=args.dpi,
            layout=args.layout,
            summary_panel=args.summary_panel,
            inset_zoom=args.inset_zoom,
            inset_xmax=args.inset_xmax,
            inset_ymin=args.inset_ymin,
            log_fpr=args.log_fpr,
            log_fpr_min=args.log_fpr_min,
        )
    elif args.backend == "plotly":
        plot_roc_curves_plotly(
            fpr,
            tpr,
            roc_auc,
            class_names,
            Path(args.output),
            width_cm=args.width_cm,
            height_cm=args.height_cm,
            dpi=args.dpi,
            layout=args.layout,
            summary_panel=args.summary_panel,
            inset_zoom=args.inset_zoom,
            inset_xmax=args.inset_xmax,
            inset_ymin=args.inset_ymin,
            log_fpr=args.log_fpr,
            log_fpr_min=args.log_fpr_min,
        )
    else:
        plot_roc_curves(
            fpr,
            tpr,
            roc_auc,
            class_names,
            Path(args.output),
            width_cm=args.width_cm,
            height_cm=args.height_cm,
            dpi=args.dpi,
            layout=args.layout,
            summary_panel=args.summary_panel,
            inset_zoom=args.inset_zoom,
            inset_xmax=args.inset_xmax,
            inset_ymin=args.inset_ymin,
            log_fpr=args.log_fpr,
            log_fpr_min=args.log_fpr_min,
        )

    micro_auc = roc_auc.get("micro")
    macro_auc = roc_auc.get("macro")
    print(
        f"ROC curves saved to {args.output}\n"
        f"Micro-average AUC: {micro_auc:.3f}\n"
        f"Macro-average AUC: {macro_auc:.3f}"
    )


if __name__ == "__main__":
    main()
