"""Diagram and architecture visualization helpers.

This module provides thesis-ready diagrams using optional dependencies:
- NetworkX for graph-style pipeline diagrams
- Graphviz (pydot) for flow diagrams
- Keras/TensorFlow plot_model or ann_visualizer for NN architectures
- BertViz for transformer attention views
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import networkx as nx


PIPELINE_NODES: Sequence[Tuple[str, str]] = [
    ("CXR Dataset", "data"),
    ("Landmark Detector", "model"),
    ("Canonical Shape (GPA)", "process"),
    ("Warping", "process"),
    ("Warped Dataset", "data"),
    ("CNN Classifier", "model"),
    ("Evaluation", "process"),
    ("Results", "output"),
]

PIPELINE_EDGES: Sequence[Tuple[str, str]] = [
    ("CXR Dataset", "Landmark Detector"),
    ("Landmark Detector", "Canonical Shape (GPA)"),
    ("Canonical Shape (GPA)", "Warping"),
    ("Warping", "Warped Dataset"),
    ("Warped Dataset", "CNN Classifier"),
    ("CNN Classifier", "Evaluation"),
    ("Evaluation", "Results"),
]

PIPELINE_COLORS = {
    "data": "#b3cde3",
    "model": "#ccebc5",
    "process": "#fbb4ae",
    "output": "#decbe4",
}


def _coerce_output_path(output_path: Union[str, Path], default_suffix: str) -> Path:
    path = Path(output_path)
    if not path.suffix:
        path = path.with_suffix(default_suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_pipeline_graph() -> "nx.DiGraph":
    """Build a default pipeline graph for thesis figures.

    Returns:
        networkx.DiGraph: Directed graph with node metadata.
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("networkx is required to build the pipeline graph") from exc

    graph = nx.DiGraph()
    for node, group in PIPELINE_NODES:
        graph.add_node(node, group=group)
    graph.add_edges_from(PIPELINE_EDGES)
    return graph


def save_pipeline_networkx_diagram(
    output_path: Union[str, Path],
    layout: str = "dot",
    title: Optional[str] = "Pipeline Overview",
) -> Path:
    """Save a pipeline diagram using NetworkX.

    Args:
        output_path: Output image path.
        layout: Graphviz layout program (dot, neato, fdp, sfdp).
        title: Optional title for the figure.

    Returns:
        Path to the saved diagram.
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("networkx is required for NetworkX diagrams") from exc

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to render NetworkX diagrams") from exc

    graph = build_pipeline_graph()
    node_colors = [
        PIPELINE_COLORS.get(graph.nodes[node].get("group"), "#dddddd")
        for node in graph.nodes
    ]

    try:
        from networkx.drawing.nx_pydot import graphviz_layout

        pos = graphviz_layout(graph, prog=layout)
    except Exception:
        # Fall back to spring layout when Graphviz is not available.
        pos = nx.spring_layout(graph, seed=7)

    output_path = _coerce_output_path(output_path, ".png")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    nx.draw_networkx(
        graph,
        pos=pos,
        ax=ax,
        with_labels=True,
        node_color=node_colors,
        node_size=2800,
        font_size=9,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=14,
    )
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("NetworkX diagram saved to: %s", output_path)
    return output_path


def save_pipeline_graphviz_diagram(
    output_path: Union[str, Path],
    rankdir: str = "LR",
    title: Optional[str] = None,
) -> Path:
    """Save a pipeline diagram using Graphviz via pydot.

    Args:
        output_path: Output file path (png/svg/pdf).
        rankdir: Graphviz rank direction (LR, TB).
        title: Optional title for the graph.

    Returns:
        Path to the saved diagram.
    """
    try:
        import pydot
    except ImportError as exc:
        raise ImportError("pydot is required for Graphviz diagrams") from exc

    output_path = _coerce_output_path(output_path, ".png")
    graph = pydot.Dot(graph_type="digraph", rankdir=rankdir)
    graph.set_node_defaults(shape="box", style="rounded,filled", fontname="Helvetica")
    graph.set_edge_defaults(color="#555555")

    if title:
        graph.set_label(title)
        graph.set_labelloc("t")

    for node, group in PIPELINE_NODES:
        graph.add_node(pydot.Node(node, fillcolor=PIPELINE_COLORS.get(group, "#dddddd")))
    for src, dst in PIPELINE_EDGES:
        graph.add_edge(pydot.Edge(src, dst))

    fmt = output_path.suffix.lstrip(".")
    try:
        graph.write(str(output_path), format=fmt)
    except Exception as exc:
        raise RuntimeError(
            "Graphviz rendering failed. Ensure Graphviz is installed and on PATH."
        ) from exc

    logger.info("Graphviz diagram saved to: %s", output_path)
    return output_path


def save_keras_model_diagram(
    model: object,
    output_path: Union[str, Path],
    prefer: str = "plot_model",
    show_shapes: bool = True,
    show_layer_names: bool = True,
    expand_nested: bool = True,
) -> Path:
    """Save a Keras model architecture diagram.

    Args:
        model: Keras model instance.
        output_path: Output image path.
        prefer: "plot_model" or "ann_visualizer".
        show_shapes: Include tensor shapes (plot_model only).
        show_layer_names: Include layer names (plot_model only).
        expand_nested: Expand nested models (plot_model only).

    Returns:
        Path to the saved diagram.
    """
    if prefer == "plot_model":
        output_path = _coerce_output_path(output_path, ".png")
        try:
            from tensorflow.keras.utils import plot_model
        except ImportError as exc:
            raise ImportError("tensorflow is required for plot_model diagrams") from exc

        plot_model(
            model,
            to_file=str(output_path),
            show_shapes=show_shapes,
            show_layer_names=show_layer_names,
            expand_nested=expand_nested,
        )
        logger.info("Keras plot_model diagram saved to: %s", output_path)
        return output_path

    if prefer == "ann_visualizer":
        output_path = _coerce_output_path(output_path, ".gv")
        try:
            from ann_visualizer.visualize import ann_viz
        except ImportError as exc:
            raise ImportError("ann_visualizer is required for ann_viz diagrams") from exc

        base_path = output_path.with_suffix("")
        ann_viz(model, view=False, filename=str(base_path))
        logger.info("ANN Visualizer diagram saved to: %s", output_path)
        return output_path

    raise ValueError("prefer must be 'plot_model' or 'ann_visualizer'")


def save_bertviz_attention_html(
    attentions: Sequence,
    tokens: Sequence[str],
    output_path: Union[str, Path],
    sentence_b: Optional[Sequence[str]] = None,
    layer: Optional[int] = None,
    heads: Optional[Sequence[int]] = None,
) -> Path:
    """Save a BertViz attention view as HTML.

    Args:
        attentions: Attention tensors from a transformer model.
        tokens: Tokens for sentence A.
        output_path: Output HTML path.
        sentence_b: Optional tokens for sentence B.
        layer: Optional layer index to focus on.
        heads: Optional list of head indices to focus on.

    Returns:
        Path to the saved HTML file.
    """
    try:
        from bertviz import head_view
    except ImportError as exc:
        raise ImportError("bertviz is required to render attention views") from exc

    output_path = _coerce_output_path(output_path, ".html")
    import inspect

    signature = inspect.signature(head_view)
    kwargs = {
        "attention": attentions,
        "tokens": list(tokens),
        "layer": layer,
        "heads": heads,
        "html_action": "return",
    }
    if sentence_b is not None:
        if "sentence_b" in signature.parameters:
            kwargs["sentence_b"] = list(sentence_b)
        elif "sentence_b_start" in signature.parameters:
            merged_tokens = list(tokens) + ["[SEP]"] + list(sentence_b)
            kwargs["tokens"] = merged_tokens
            kwargs["sentence_b_start"] = len(tokens) + 1

    try:
        html = head_view(**kwargs)
    except TypeError:
        kwargs.pop("html_action", None)
        view = head_view(**kwargs)
        html = getattr(view, "html", None)

    if not isinstance(html, str):
        if hasattr(html, "data"):
            html = html.data
        elif hasattr(html, "_repr_html_"):
            html = html._repr_html_()

    if not html or not isinstance(html, str):
        raise RuntimeError(
            "Unable to extract HTML from bertviz. Try running head_view in a notebook."
        )

    output_path.write_text(html, encoding="utf-8")
    logger.info("BertViz attention view saved to: %s", output_path)
    return output_path
