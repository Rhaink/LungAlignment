"""
Visualization functions for the GUI.

Provides rendering functions for landmarks, warped images, GradCAM heatmaps,
and comparison views.
"""
import io
from typing import Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from PIL import Image
from scipy.spatial import Delaunay

# Use non-interactive backend for thread safety
matplotlib.use('Agg')

from .config import (
    LANDMARK_COLORS,
    LANDMARK_GROUPS,
    LANDMARK_LABELS_ES,
    CLASS_COLORS,
    CLASS_NAMES_ES,
)
from .gradcam_utils import overlay_heatmap_on_image
from ..constants import SYMMETRIC_PAIRS, CENTRAL_LANDMARKS


def render_original(image: np.ndarray) -> Image.Image:
    """
    Render original image with minimal processing.

    Args:
        image: Input image (H, W) grayscale or (H, W, 3) RGB

    Returns:
        PIL Image
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    if len(image.shape) == 2:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)

    ax.axis('off')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Convert to PIL
    pil_image = _fig_to_pil(fig)
    plt.close(fig)

    return pil_image


def render_landmarks_overlay(
    image: np.ndarray,
    landmarks: np.ndarray,
    show_labels: bool = True,
    show_connections: bool = True,
    image_size: int = 224
) -> Image.Image:
    """
    Render landmarks overlaid on image with color-coded groups.

    Args:
        image: Input image (H, W) grayscale or (H, W, 3) RGB
        landmarks: Landmark coordinates (15, 2) in pixel space
        show_labels: Whether to show L1-L15 labels
        show_connections: Whether to show connecting lines
        image_size: Expected image size for validation

    Returns:
        PIL Image with landmarks overlay
    """
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

    # Display image
    if len(image.shape) == 2:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)

    # Ensure landmarks are in pixel coordinates
    if landmarks.max() <= 1.0:
        landmarks_px = landmarks * image_size
    else:
        landmarks_px = landmarks

    # ============================================================
    # NUEVO: Dibujar líneas conectoras ANTES de los puntos
    # ============================================================
    if show_connections:
        _draw_connection_lines(
            ax,
            landmarks_px,
            show_symmetric=True,
            show_central_axis=True
        )
    # ============================================================

    # Draw each landmark with color by group
    for i in range(15):
        group = LANDMARK_GROUPS[i]
        color = LANDMARK_COLORS[group]

        # Draw landmark point (circle with white border)
        ax.scatter(
            landmarks_px[i, 0],
            landmarks_px[i, 1],
            c=color,
            s=150,
            marker='o',
            edgecolors='white',
            linewidths=2,
            alpha=0.9,
            zorder=10
        )

        # Add label if requested
        if show_labels:
            ax.annotate(
                f'L{i+1}',
                (landmarks_px[i, 0] + 3, landmarks_px[i, 1] - 3),
                fontsize=10,
                color='white',
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6),
                zorder=11
            )

    # ============================================================
    # NUEVO: Leyenda mejorada con nombres anatómicos
    # ============================================================
    # COMMENTED OUT FOR v1.0.9 - Leyenda removida a solicitud del usuario
    # legend_elements = [
    #     plt.Line2D([0], [0], marker='o', color='w',
    #                markerfacecolor=LANDMARK_COLORS['axis'],
    #                markersize=12,
    #                label=LANDMARK_LABELS_ES['axis']),
    #     plt.Line2D([0], [0], marker='o', color='w',
    #                markerfacecolor=LANDMARK_COLORS['central'],
    #                markersize=12,
    #                label=LANDMARK_LABELS_ES['central']),
    #     plt.Line2D([0], [0], marker='o', color='w',
    #                markerfacecolor=LANDMARK_COLORS['lateral'],
    #                markersize=12,
    #                label=LANDMARK_LABELS_ES['lateral']),
    #     plt.Line2D([0], [0], marker='o', color='w',
    #                markerfacecolor=LANDMARK_COLORS['border'],
    #                markersize=12,
    #                label=LANDMARK_LABELS_ES['border']),
    #     plt.Line2D([0], [0], marker='o', color='w',
    #                markerfacecolor=LANDMARK_COLORS['costal'],
    #                markersize=12,
    #                label=LANDMARK_LABELS_ES['costal']),
    #     # ============================================================
    #     # NUEVO: Añadir elementos de leyenda para líneas
    #     # ============================================================
    #     plt.Line2D([0], [0], color='white', linewidth=1.5,
    #                linestyle='--', alpha=0.4,
    #                label='Simetría Izq-Der'),
    #     plt.Line2D([0], [0], color='#00FFFF', linewidth=2.0,
    #                linestyle='-', alpha=0.6,
    #                label='Eje Central Vertical'),
    # ]
    #
    # ax.legend(
    #     handles=legend_elements,
    #     loc='upper right',
    #     fontsize=9,  # Reducido de 10 para acomodar más items
    #     facecolor='black',
    #     edgecolor='white',
    #     labelcolor='white',
    #     framealpha=0.8
    # )
    # ============================================================

    ax.axis('off')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Convert to PIL
    pil_image = _fig_to_pil(fig)
    plt.close(fig)

    return pil_image


def _draw_connection_lines(
    ax: plt.Axes,
    landmarks_px: np.ndarray,
    show_symmetric: bool = True,
    show_central_axis: bool = True
):
    """
    Dibuja líneas conectoras entre puntos de referencia relacionados.

    Args:
        ax: Matplotlib axes
        landmarks_px: Coordenadas de landmarks en píxeles (15, 2)
        show_symmetric: Si True, dibuja líneas entre pares simétricos izq-der
        show_central_axis: Si True, dibuja línea del eje central L1-L9-L10-L11-L2
    """
    # Líneas de simetría (horizontal izq-der)
    if show_symmetric:
        for left_idx, right_idx in SYMMETRIC_PAIRS:
            ax.plot(
                [landmarks_px[left_idx, 0], landmarks_px[right_idx, 0]],
                [landmarks_px[left_idx, 1], landmarks_px[right_idx, 1]],
                color='white',
                linewidth=1.5,
                alpha=0.4,
                linestyle='--',
                zorder=5  # Detrás de los puntos (zorder=10)
            )

    # Eje central (vertical L1-L9-L10-L11-L2)
    if show_central_axis:
        # Secuencia: L1(0) → L9(8) → L10(9) → L11(10) → L2(1)
        central_sequence = [0] + CENTRAL_LANDMARKS + [1]

        for i in range(len(central_sequence) - 1):
            idx1, idx2 = central_sequence[i], central_sequence[i + 1]
            ax.plot(
                [landmarks_px[idx1, 0], landmarks_px[idx2, 0]],
                [landmarks_px[idx1, 1], landmarks_px[idx2, 1]],
                color='#00FFFF',  # Cyan (color del grupo central)
                linewidth=2.0,
                alpha=0.6,
                linestyle='-',
                zorder=5
            )


def render_delaunay_mesh(
    image: np.ndarray,
    landmarks: np.ndarray,
    show_labels: bool = True,
    show_landmark_points: bool = True,
    mesh_color: str = '#00FFFF',
    mesh_alpha: float = 0.6,
    mesh_linewidth: float = 1.5,
    fill_triangles: bool = False,
    image_size: int = 224
) -> Image.Image:
    """
    Renderiza malla de Delaunay sobre imagen original.

    La malla muestra cómo se divide la imagen en regiones triangulares
    para el proceso de warping piecewise affine.

    Args:
        image: Input image (H, W) grayscale
        landmarks: Landmark coordinates (15, 2) in pixel space
        show_labels: Whether to show L1-L15 labels
        show_landmark_points: Whether to show landmark dots
        mesh_color: Color for triangle edges
        mesh_alpha: Transparency for triangle edges
        mesh_linewidth: Width of triangle edges
        fill_triangles: Whether to fill triangles with semi-transparent color
        image_size: Expected image size for validation

    Returns:
        PIL Image with Delaunay mesh overlay
    """
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

    # Display image
    if len(image.shape) == 2:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)

    # Ensure landmarks are in pixel coordinates
    if landmarks.max() <= 1.0:
        landmarks_px = landmarks * image_size
    else:
        landmarks_px = landmarks

    # Compute Delaunay triangulation on PREDICTED landmarks
    tri = Delaunay(landmarks_px)
    triangles = tri.simplices  # (n_triangles, 3) array of indices

    # Draw each triangle
    for tri_indices in triangles:
        # Get triangle vertices
        tri_points = landmarks_px[tri_indices]

        # Create polygon for this triangle
        poly = Polygon(
            tri_points,
            closed=True,
            edgecolor=mesh_color,
            facecolor=mesh_color if fill_triangles else 'none',
            linewidth=mesh_linewidth,
            alpha=mesh_alpha if fill_triangles else 1.0,
            linestyle='-',
            zorder=5
        )
        ax.add_patch(poly)

    # Optionally draw landmark points
    if show_landmark_points:
        for i in range(15):
            group = LANDMARK_GROUPS[i]
            color = LANDMARK_COLORS[group]

            ax.scatter(
                landmarks_px[i, 0],
                landmarks_px[i, 1],
                c=color,
                s=100,
                marker='o',
                edgecolors='white',
                linewidths=1.5,
                alpha=0.9,
                zorder=10
            )

    # Optionally add labels
    if show_labels:
        for i in range(15):
            ax.annotate(
                f'L{i+1}',
                (landmarks_px[i, 0] + 3, landmarks_px[i, 1] - 3),
                fontsize=8,
                color='white',
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6),
                zorder=11
            )

    ax.axis('off')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Convert to PIL
    pil_image = _fig_to_pil(fig)
    plt.close(fig)

    return pil_image


def render_warped(warped_image: np.ndarray) -> Image.Image:
    """
    Render warped (normalized) image.

    Args:
        warped_image: Warped image (H, W) grayscale

    Returns:
        PIL Image
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    ax.imshow(warped_image, cmap='gray')
    ax.axis('off')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Convert to PIL
    pil_image = _fig_to_pil(fig)
    plt.close(fig)

    return pil_image


def enhance_contrast_sahs_masked(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Aplica el algoritmo SAHS solo a la región pulmonar (píxeles > threshold).

    El fondo negro (píxeles <= threshold) se mantiene intacto.

    Statistical Asymmetrical Histogram Stretching (SAHS):
    - Calcula límites de estiramiento asimétricos basados en la media
    - Factor 2.5 para el límite superior
    - Factor 2.0 para el límite inferior

    Args:
        image: Input image (H, W) grayscale uint8
        threshold: Threshold to separate background from lung region

    Returns:
        Enhanced image (H, W) grayscale uint8
    """
    if image is None:
        raise ValueError("La imagen de entrada es None")

    # Convertir a escala de grises si es necesario
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    # Crear máscara de la región pulmonar (excluyendo fondo negro)
    mask = gray_image > threshold
    lung_pixels = gray_image[mask].astype(np.float64)

    if lung_pixels.size == 0:
        return gray_image

    # Calcular la media solo de los píxeles pulmonares
    gray_mean = np.mean(lung_pixels)

    # Separar píxeles por encima y debajo de la media
    above_mean = lung_pixels[lung_pixels > gray_mean]
    below_or_equal_mean = lung_pixels[lung_pixels <= gray_mean]

    # Calcular límites usando desviación estándar asimétrica
    max_value = gray_mean
    min_value = gray_mean

    if above_mean.size > 0:
        # Factor 2.5 para el límite superior
        std_above = np.sqrt(np.mean((above_mean - gray_mean) ** 2))
        max_value = gray_mean + 2.5 * std_above

    if below_or_equal_mean.size > 0:
        # Factor 2.0 para el límite inferior
        std_below = np.sqrt(np.mean((below_or_equal_mean - gray_mean) ** 2))
        min_value = gray_mean - 2.0 * std_below

    # Crear imagen de salida (iniciar con ceros para mantener fondo negro)
    enhanced_image = np.zeros_like(gray_image)

    # Aplicar transformación solo a la región pulmonar
    if max_value != min_value:
        transformed = (255 / (max_value - min_value)) * (gray_image.astype(np.float64) - min_value)
        transformed = np.clip(transformed, 0, 255).astype(np.uint8)
        # Aplicar solo donde hay región pulmonar
        enhanced_image[mask] = transformed[mask]
    else:
        enhanced_image[mask] = gray_image[mask]

    return enhanced_image


def render_warped_sahs(warped_image: np.ndarray, threshold: int = 10) -> Image.Image:
    """
    Renderiza imagen warped con mejora de contraste SAHS.

    Args:
        warped_image: Imagen normalizada (H, W) grayscale uint8
        threshold: Umbral para separar fondo de región pulmonar

    Returns:
        PIL.Image con SAHS aplicado
    """
    # Aplicar SAHS
    enhanced = enhance_contrast_sahs_masked(warped_image, threshold)

    # Crear figura matplotlib
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.imshow(enhanced, cmap='gray', vmin=0, vmax=255)
    ax.axis('off')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    plt.tight_layout()

    # Convertir a PIL
    pil_image = _fig_to_pil(fig)
    plt.close(fig)

    return pil_image


def render_gradcam(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    predicted_class: Optional[str] = None
) -> Image.Image:
    """
    Render GradCAM heatmap overlay on image.

    Args:
        image: Original image (H, W) grayscale or (H, W, 3) RGB
        heatmap: Normalized heatmap (H, W) with values [0, 1]
        alpha: Heatmap transparency (0=invisible, 1=opaque)
        predicted_class: Optional predicted class name for title

    Returns:
        PIL Image with GradCAM overlay
    """
    # Create overlay
    overlay = overlay_heatmap_on_image(image, heatmap, alpha=alpha)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    ax.imshow(overlay)
    ax.axis('off')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Add colorbar to show heatmap scale
    # Create a dummy mappable for colorbar
    from matplotlib import cm
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Activación', rotation=270, labelpad=20, color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Convert to PIL
    pil_image = _fig_to_pil(fig)
    plt.close(fig)

    return pil_image


def render_comparison_side_by_side(
    original: np.ndarray,
    warped: np.ndarray
) -> Image.Image:
    """
    Render side-by-side comparison of original and warped images.

    Args:
        original: Original image (H, W)
        warped: Warped image (H, W)

    Returns:
        PIL Image with comparison
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=150)

    # Original
    ax1.imshow(original, cmap='gray')
    ax1.axis('off')

    # Warped
    ax2.imshow(warped, cmap='gray')
    ax2.axis('off')

    fig.patch.set_facecolor('black')
    plt.tight_layout()

    # Convert to PIL
    pil_image = _fig_to_pil(fig)
    plt.close(fig)

    return pil_image


def create_probability_chart(
    probabilities: dict,
    title: str = "Probabilidades de Clasificación"
) -> Image.Image:
    """
    Create horizontal bar chart of class probabilities.

    Args:
        probabilities: Dict mapping class names to probabilities
        title: Chart title

    Returns:
        PIL Image with bar chart
    """
    # Convert class names to Spanish
    classes_es = []
    probs = []
    colors = []

    for cls, prob in probabilities.items():
        if cls in ['COVID', 'Normal', 'Viral_Pneumonia']:
            idx = ['COVID', 'Normal', 'Viral_Pneumonia'].index(cls)
            classes_es.append(CLASS_NAMES_ES[idx])
            probs.append(prob * 100)  # Convert to percentage
            colors.append(CLASS_COLORS[cls])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)

    # Horizontal bars
    y_pos = np.arange(len(classes_es))
    bars = ax.barh(y_pos, probs, color=colors, alpha=0.8, edgecolor='white', linewidth=2)

    # Add percentage labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        width = bar.get_width()
        ax.text(
            width + 1,
            bar.get_y() + bar.get_height() / 2,
            f'{prob:.2f}%',
            ha='left',
            va='center',
            fontsize=12,
            color='white',
            weight='bold'
        )

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes_es, fontsize=12, color='white')
    ax.set_xlabel('Probabilidad (%)', fontsize=12, color='white')
    ax.set_xlim([0, 105])

    # Grid
    ax.grid(axis='x', alpha=0.3, color='white', linestyle='--')
    ax.set_axisbelow(True)

    # Background
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('black')

    # Tick colors
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Spine colors
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    plt.tight_layout()

    # Convert to PIL
    pil_image = _fig_to_pil(fig)
    plt.close(fig)

    return pil_image


def create_metrics_table(
    landmarks: np.ndarray,
    per_landmark_errors: Optional[dict] = None
) -> pd.DataFrame:
    """
    Create metrics table with landmark coordinates and errors.

    Args:
        landmarks: Predicted landmarks (15, 2) in pixel space
        per_landmark_errors: Optional dict with reference errors per landmark

    Returns:
        DataFrame with metrics
    """
    from .config import PER_LANDMARK_ERRORS

    if per_landmark_errors is None:
        per_landmark_errors = PER_LANDMARK_ERRORS

    data = []
    for i in range(15):
        landmark_name = f'L{i+1}'
        x, y = landmarks[i]

        data.append({
            'Landmark': landmark_name,
            'X (px)': f'{x:.1f}',
            'Y (px)': f'{y:.1f}'
        })

    df = pd.DataFrame(data)
    return df


def export_to_pdf(
    images: dict,
    metrics_df: pd.DataFrame,
    output_path: str,
    metadata: Optional[dict] = None
):
    """
    Export all visualizations to a multi-page PDF.

    Args:
        images: Dict with keys: 'original', 'landmarks', 'delaunay_mesh', 'warped', 'warped_sahs'
        metrics_df: DataFrame with metrics
        output_path: Output PDF path
        metadata: Optional metadata (inference time, etc.)
    """
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output_path) as pdf:
        # Page 1: Original + Landmarks
        if 'landmarks' in images:
            fig = plt.figure(figsize=(11, 8.5))
            plt.imshow(images['landmarks'])
            plt.axis('off')
            plt.title('Puntos de Referencia Detectados', fontsize=16, color='black', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        # Page 2: Delaunay Mesh (NUEVA)
        if 'delaunay_mesh' in images:
            fig = plt.figure(figsize=(11, 8.5))
            plt.imshow(images['delaunay_mesh'])
            plt.axis('off')
            plt.title('Malla de Delaunay para Warping', fontsize=16, color='black', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        # Page 3: Warped (antes era Page 2)
        if 'warped' in images:
            fig = plt.figure(figsize=(11, 8.5))
            plt.imshow(images['warped'], cmap='gray')
            plt.axis('off')
            plt.title('Imagen Normalizada', fontsize=16, color='black', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        # Page 4: SAHS (antes era Page 3)
        if 'warped_sahs' in images:
            fig = plt.figure(figsize=(11, 8.5))
            plt.imshow(images['warped_sahs'])
            plt.axis('off')
            plt.title('Imagen Normalizada con SAHS', fontsize=16, color='black', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        # Page 5: Metrics table (antes era Page 4)
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(
            cellText=metrics_df.values,
            colLabels=metrics_df.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.30, 0.30, 0.30]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(len(metrics_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Metadata
        if metadata:
            metadata_text = '\n'.join([f'{k}: {v}' for k, v in metadata.items()])
            ax.text(
                0.5, 0.02, metadata_text,
                transform=fig.transFigure,
                ha='center', va='bottom',
                fontsize=9, family='monospace',
                color='gray'
            )

        plt.title('Métricas Detalladas', fontsize=16, pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Metadata
        d = pdf.infodict()
        d['Title'] = 'Resultados de Detección COVID-19'
        d['Author'] = 'Sistema de Puntos de Referencia Anatómicos'
        d['Subject'] = 'Análisis de radiografía de tórax'
        d['Keywords'] = 'COVID-19, Puntos de Referencia, Deep Learning, SAHS'
        d['CreationDate'] = None  # Will use current time


def _fig_to_pil(fig: plt.Figure) -> Image.Image:
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=150)
    buf.seek(0)
    pil_image = Image.open(buf).copy()  # Copy to avoid issues when closing buffer
    buf.close()
    return pil_image
