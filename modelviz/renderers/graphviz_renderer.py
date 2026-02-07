"""
Graphviz-based rendering engine for modelviz.

Renders neural network graphs using the graphviz library for
display in Jupyter/Colab notebooks.
"""

from typing import Optional, Sequence

import graphviz

from modelviz.graph.builder import Edge
from modelviz.graph.layer_node import LayerNode

# Color scheme for different layer types
LAYER_COLORS = {
    "conv": "#6366f1",  # Indigo for convolutions
    "linear": "#8b5cf6",  # Purple for linear/dense
    "pool": "#06b6d4",  # Cyan for pooling
    "norm": "#10b981",  # Emerald for normalization
    "activation": "#f59e0b",  # Amber for activations
    "dropout": "#ef4444",  # Red for dropout
    "flatten": "#ec4899",  # Pink for flatten
    "embed": "#84cc16",  # Lime for embeddings
    "recurrent": "#14b8a6",  # Teal for RNN/LSTM/GRU
    "attention": "#f97316",  # Orange for attention
    "default": "#64748b",  # Slate for unknown
}

# Font colors for dark backgrounds
FONT_COLOR_LIGHT = "#ffffff"
FONT_COLOR_DARK = "#1e293b"


def render_graph(
    nodes: Sequence[LayerNode],
    edges: Sequence[Edge],
    show_shapes: bool = True,
    show_params: bool = True,
    title: Optional[str] = None,
) -> graphviz.Digraph:
    """
    Render a neural network graph using Graphviz.

    Creates a directed graph visualization with styled nodes based on
    layer types and optional shape/parameter annotations.

    Args:
        nodes: Sequence of LayerNode objects to render.
        edges: Sequence of Edge objects connecting nodes.
        show_shapes: Whether to show output shapes in node labels.
        show_params: Whether to show parameter counts in node labels.
        title: Optional title for the graph.

    Returns:
        graphviz.Digraph object that can be displayed in notebooks.

    Example:
        >>> from modelviz.graph import LayerNode, GraphBuilder
        >>> nodes = [LayerNode(0, "conv1", "Conv2d", output_shape=(1, 32, 28, 28), params=320)]
        >>> builder = GraphBuilder(nodes)
        >>> graph = builder.build()
        >>> digraph = render_graph(graph.nodes, graph.edges)
        >>> digraph  # Displays in Jupyter

    # TODO: Add HTML renderer backend option
    # TODO: Add 3D renderer backend option
    """
    # Create the digraph with styling
    dot = graphviz.Digraph(
        name="neural_network",
        format="svg",
        graph_attr={
            "rankdir": "TB",  # Top to bottom layout
            "splines": "ortho",
            "nodesep": "0.5",
            "ranksep": "0.8",
            "bgcolor": "transparent",
            "fontname": "Helvetica Neue, Helvetica, Arial, sans-serif",
            "fontsize": "14",
            "pad": "0.5",
        },
        node_attr={
            "fontname": "Helvetica Neue, Helvetica, Arial, sans-serif",
            "fontsize": "11",
            "margin": "0.2,0.1",
        },
        edge_attr={
            "color": "#94a3b8",
            "arrowsize": "0.8",
            "penwidth": "1.5",
        },
    )

    # Add title if provided
    if title:
        dot.attr(label=title, labelloc="t", fontsize="16", fontweight="bold")

    # Add nodes
    for node in nodes:
        label = _create_node_label(node, show_shapes, show_params)
        style = _get_node_style(node)
        dot.node(str(node.id), label=label, **style)

    # Add edges with styling based on edge type
    for edge in edges:
        edge_attrs = {}

        # Style based on edge type
        edge_type = getattr(edge, "edge_type", "sequential")

        if edge_type == "residual":
            # Residual/skip connections - curved, dashed, different color
            edge_attrs["style"] = "dashed"
            edge_attrs["color"] = "#f59e0b"  # Amber
            edge_attrs["penwidth"] = "2.0"
            edge_attrs["constraint"] = "false"  # Don't affect layout
        elif edge_type == "skip":
            # Skip connections - curved, dotted
            edge_attrs["style"] = "dotted"
            edge_attrs["color"] = "#10b981"  # Emerald
            edge_attrs["penwidth"] = "2.0"
            edge_attrs["constraint"] = "false"
        elif edge_type == "concat":
            # Concatenation - bold, different color
            edge_attrs["style"] = "bold"
            edge_attrs["color"] = "#8b5cf6"  # Purple
            edge_attrs["penwidth"] = "2.5"
        # else: sequential - use defaults

        if edge.label:
            edge_attrs["label"] = edge.label

        dot.edge(str(edge.source_id), str(edge.target_id), **edge_attrs)

    return dot


def _create_node_label(
    node: LayerNode,
    show_shapes: bool,
    show_params: bool,
) -> str:
    """
    Create a multiline label for a node.

    Format:
        LayerType
        Output: (shape)
        Params: N
    """
    lines = [node.display_type]

    if show_shapes and node.output_shape is not None:
        shape_str = node.formatted_output_shape
        lines.append(f"Output: {shape_str}")

    if show_params and node.params > 0:
        params_str = node.formatted_params
        lines.append(f"Params: {params_str}")

    # Use HTML-like labels for multiline
    return "\\n".join(lines)


def _get_node_style(node: LayerNode) -> dict[str, str]:
    """
    Get Graphviz styling attributes for a node based on its type.

    Styling rules:
    - Conv → box
    - Linear/Dense → ellipse
    - Pooling → box (smaller height)
    - Flatten → trapezium
    - Activation → rounded box
    - BatchNorm → box with dashed border
    """
    layer_type = node.type.lower()

    # Determine shape and style based on layer type
    shape = "box"
    style = "filled,rounded"
    color = LAYER_COLORS["default"]
    fontcolor = FONT_COLOR_LIGHT
    peripheries = "1"
    height = "0.5"
    width = "1.5"

    # Convolution layers
    if "conv" in layer_type:
        shape = "box"
        color = LAYER_COLORS["conv"]
        height = "0.6"
        width = "2.0"

    # Linear / Dense layers
    elif "linear" in layer_type or "dense" in layer_type:
        shape = "ellipse"
        color = LAYER_COLORS["linear"]
        height = "0.5"
        width = "1.8"

    # Pooling layers
    elif "pool" in layer_type:
        shape = "box"
        color = LAYER_COLORS["pool"]
        height = "0.4"
        width = "1.5"

    # Flatten layers
    elif "flatten" in layer_type:
        shape = "trapezium"
        color = LAYER_COLORS["flatten"]
        height = "0.4"
        width = "1.5"

    # Activation functions
    elif any(
        act in layer_type
        for act in ["relu", "sigmoid", "tanh", "gelu", "softmax", "activation"]
    ):
        shape = "box"
        style = "filled,rounded"
        color = LAYER_COLORS["activation"]
        height = "0.4"
        width = "1.2"

    # Batch normalization
    elif "norm" in layer_type or "bn" in layer_type:
        shape = "box"
        style = "filled,dashed"
        color = LAYER_COLORS["norm"]
        height = "0.4"
        width = "1.5"

    # Dropout
    elif "dropout" in layer_type:
        shape = "box"
        style = "filled,dashed"
        color = LAYER_COLORS["dropout"]
        height = "0.4"
        width = "1.2"

    # Embedding layers
    elif "embed" in layer_type:
        shape = "box3d"
        color = LAYER_COLORS["embed"]
        height = "0.5"
        width = "1.5"

    # Recurrent layers
    elif any(rnn in layer_type for rnn in ["lstm", "gru", "rnn"]):
        shape = "box"
        style = "filled,rounded"
        color = LAYER_COLORS["recurrent"]
        height = "0.6"
        width = "2.0"

    # Attention layers
    elif "attention" in layer_type or "transformer" in layer_type:
        shape = "box"
        style = "filled,rounded"
        color = LAYER_COLORS["attention"]
        height = "0.6"
        width = "2.0"

    # Handle grouped nodes (multiple layers merged)
    if node.is_grouped:
        peripheries = "2"  # Double border for grouped nodes
        width = "2.2"

    return {
        "shape": shape,
        "style": style,
        "fillcolor": color,
        "fontcolor": fontcolor,
        "height": height,
        "width": width,
        "peripheries": peripheries,
    }


def render_to_file(
    nodes: Sequence[LayerNode],
    edges: Sequence[Edge],
    filepath: str,
    format: str = "png",
    show_shapes: bool = True,
    show_params: bool = True,
    title: Optional[str] = None,
) -> str:
    """
    Render the graph and save to a file.

    Args:
        nodes: Sequence of LayerNode objects.
        edges: Sequence of Edge objects.
        filepath: Output file path (without extension).
        format: Output format ("png", "svg", "pdf").
        show_shapes: Whether to show output shapes.
        show_params: Whether to show parameter counts.
        title: Optional title for the graph.

    Returns:
        Path to the rendered file.
    """
    dot = render_graph(nodes, edges, show_shapes, show_params, title)
    dot.format = format
    return dot.render(filepath, cleanup=True)
