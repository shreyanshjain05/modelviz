"""
Plotly-based 3D rendering engine for modelviz.

Renders neural network graphs as interactive 3D visualizations
that work in Jupyter/Colab notebooks.
"""

import math
from typing import Literal, Optional, Sequence

from modelviz.graph.builder import Edge
from modelviz.graph.layer_node import LayerNode

# Color scheme for different layer types (same as graphviz but in hex)
LAYER_COLORS_3D = {
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


def render_graph_3d(
    nodes: Sequence[LayerNode],
    edges: Sequence[Edge],
    show_shapes: bool = True,
    show_params: bool = True,
    title: Optional[str] = None,
    layout: Literal["tower", "spiral", "grid"] = "tower",
):
    """
    Render a neural network graph as an interactive 3D visualization.

    Creates a 3D Plotly figure with layers represented as 3D boxes,
    with size proportional to parameter count or output dimensions.

    Args:
        nodes: Sequence of LayerNode objects to render.
        edges: Sequence of Edge objects connecting nodes.
        show_shapes: Whether to show output shapes in hover labels.
        show_params: Whether to show parameter counts in hover labels.
        title: Optional title for the visualization.
        layout: Layout style - "tower" (vertical stack), "spiral", or "grid".

    Returns:
        plotly.graph_objects.Figure that can be displayed in notebooks.

    Example:
        >>> from modelviz.graph import LayerNode, GraphBuilder
        >>> nodes = [LayerNode(0, "conv1", "Conv2d", output_shape=(1, 32, 28, 28), params=320)]
        >>> builder = GraphBuilder(nodes)
        >>> graph = builder.build()
        >>> fig = render_graph_3d(graph.nodes, graph.edges)
        >>> fig.show()  # Interactive 3D visualization
    """
    import plotly.graph_objects as go

    # Calculate positions for each node based on layout
    positions = _calculate_positions(nodes, layout)

    # Create figure
    fig = go.Figure()

    # Add edges as 3D lines
    for edge in edges:
        source_pos = positions.get(edge.source_id)
        target_pos = positions.get(edge.target_id)
        if source_pos and target_pos:
            fig.add_trace(
                go.Scatter3d(
                    x=[source_pos[0], target_pos[0]],
                    y=[source_pos[1], target_pos[1]],
                    z=[source_pos[2], target_pos[2]],
                    mode="lines",
                    line=dict(color="#94a3b8", width=3),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Add nodes as 3D scatter points with cubes
    for node in nodes:
        pos = positions.get(node.id, (0, 0, 0))
        color = _get_layer_color(node.type)
        size = _calculate_node_size(node)

        # Create hover text
        hover_text = _create_hover_text(node, show_shapes, show_params)

        # Add node as a 3D marker
        fig.add_trace(
            go.Scatter3d(
                x=[pos[0]],
                y=[pos[1]],
                z=[pos[2]],
                mode="markers+text",
                marker=dict(
                    size=size,
                    color=color,
                    opacity=0.9,
                    symbol="square",
                    line=dict(color="white", width=2),
                ),
                text=[node.display_type],
                textposition="top center",
                textfont=dict(size=10, color="white"),
                hovertext=hover_text,
                hoverinfo="text",
                name=node.name,
                showlegend=False,
            )
        )

    # Add layer type as annotations at each node
    annotations = []
    for node in nodes:
        pos = positions.get(node.id, (0, 0, 0))
        annotations.append(
            dict(
                x=pos[0],
                y=pos[1],
                z=pos[2],
                text=node.display_type,
                showarrow=False,
                font=dict(size=10, color="white"),
            )
        )

    # Update layout for 3D view
    fig.update_layout(
        title=dict(
            text=title or "Neural Network Architecture (3D)",
            font=dict(size=20, color="#1e293b"),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(
                title="Width",
                showgrid=True,
                gridcolor="#e2e8f0",
                showbackground=True,
                backgroundcolor="#f8fafc",
            ),
            yaxis=dict(
                title="Depth",
                showgrid=True,
                gridcolor="#e2e8f0",
                showbackground=True,
                backgroundcolor="#f8fafc",
            ),
            zaxis=dict(
                title="Layer",
                showgrid=True,
                gridcolor="#e2e8f0",
                showbackground=True,
                backgroundcolor="#f8fafc",
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=2),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False,
    )

    return fig


def _calculate_positions(
    nodes: Sequence[LayerNode],
    layout: str,
) -> dict[int, tuple[float, float, float]]:
    """Calculate 3D positions for each node based on layout style."""
    positions = {}
    n = len(nodes)

    if layout == "tower":
        # Vertical stack - layers go up in Z
        for i, node in enumerate(nodes):
            x = 0
            y = 0
            z = i * 2  # Spacing between layers
            positions[node.id] = (x, y, z)

    elif layout == "spiral":
        # Spiral upward
        for i, node in enumerate(nodes):
            angle = i * 0.5
            radius = 2
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = i * 1.5
            positions[node.id] = (x, y, z)

    elif layout == "grid":
        # Grid layout with wrapping
        cols = max(1, int(math.sqrt(n)))
        for i, node in enumerate(nodes):
            row = i // cols
            col = i % cols
            x = col * 3
            y = 0
            z = row * 2
            positions[node.id] = (x, y, z)

    return positions


def _get_layer_color(layer_type: str) -> str:
    """Get color for a layer type."""
    layer_type_lower = layer_type.lower()

    if "conv" in layer_type_lower:
        return LAYER_COLORS_3D["conv"]
    elif "linear" in layer_type_lower or "dense" in layer_type_lower:
        return LAYER_COLORS_3D["linear"]
    elif "pool" in layer_type_lower:
        return LAYER_COLORS_3D["pool"]
    elif "flatten" in layer_type_lower:
        return LAYER_COLORS_3D["flatten"]
    elif any(
        act in layer_type_lower
        for act in ["relu", "sigmoid", "tanh", "gelu", "softmax", "activation"]
    ):
        return LAYER_COLORS_3D["activation"]
    elif "norm" in layer_type_lower or "bn" in layer_type_lower:
        return LAYER_COLORS_3D["norm"]
    elif "dropout" in layer_type_lower:
        return LAYER_COLORS_3D["dropout"]
    elif "embed" in layer_type_lower:
        return LAYER_COLORS_3D["embed"]
    elif any(rnn in layer_type_lower for rnn in ["lstm", "gru", "rnn"]):
        return LAYER_COLORS_3D["recurrent"]
    elif "attention" in layer_type_lower or "transformer" in layer_type_lower:
        return LAYER_COLORS_3D["attention"]

    return LAYER_COLORS_3D["default"]


def _calculate_node_size(node: LayerNode) -> float:
    """Calculate node size based on parameters or output shape."""
    # Base size
    base_size = 15

    # Scale by parameter count
    if node.params > 0:
        # Log scale for params
        param_scale = (
            math.log10(node.params + 1) / 6
        )  # Normalize to ~0-1 for up to 1M params
        return base_size + param_scale * 20

    # Scale by output dimensions if no params
    if node.output_shape:
        total_elements = 1
        for dim in node.output_shape:
            if dim > 0:
                total_elements *= dim
        dim_scale = math.log10(total_elements + 1) / 8
        return base_size + dim_scale * 15

    return base_size


def _create_hover_text(
    node: LayerNode,
    show_shapes: bool,
    show_params: bool,
) -> str:
    """Create hover text for a node."""
    lines = [
        f"<b>{node.name}</b>",
        f"Type: {node.display_type}",
    ]

    if show_shapes:
        if node.input_shape:
            lines.append(f"Input: {node.input_shape}")
        if node.output_shape:
            lines.append(f"Output: {node.output_shape}")

    if show_params:
        lines.append(f"Params: {node.formatted_params}")

    return "<br>".join(lines)


def render_3d_to_file(
    nodes: Sequence[LayerNode],
    edges: Sequence[Edge],
    filepath: str,
    format: str = "png",
    width: int = 1200,
    height: int = 800,
    show_shapes: bool = True,
    show_params: bool = True,
    title: Optional[str] = None,
    layout: Literal["tower", "spiral", "grid"] = "tower",
) -> str:
    """
    Render the 3D graph and save to a file.

    Args:
        nodes: Sequence of LayerNode objects.
        edges: Sequence of Edge objects.
        filepath: Output file path (with or without extension).
        format: Output format ("png", "svg", "pdf", "html").
        width: Image width in pixels.
        height: Image height in pixels.
        show_shapes: Whether to show output shapes.
        show_params: Whether to show parameter counts.
        title: Optional title for the graph.
        layout: Layout style.

    Returns:
        Path to the rendered file.
    """
    fig = render_graph_3d(nodes, edges, show_shapes, show_params, title, layout)

    # Ensure proper extension
    if not filepath.endswith(f".{format}"):
        filepath = f"{filepath}.{format}"

    if format == "html":
        fig.write_html(filepath)
    else:
        fig.write_image(filepath, width=width, height=height, scale=2)

    return filepath
