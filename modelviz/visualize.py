"""
Main visualization API for modelviz.

Provides the primary `visualize()` function that users interact with to
generate neural network architecture diagrams.
"""

from typing import Any, Literal, Optional, Union

import graphviz

from modelviz.graph.builder import GraphBuilder
from modelviz.graph.layer_node import LayerNode
from modelviz.renderers.graphviz_renderer import render_graph, render_to_file
from modelviz.utils.framework_detect import detect_framework
from modelviz.utils.grouping import group_layers

FrameworkLiteral = Literal["auto", "pytorch", "tensorflow"]


class VisualizationError(Exception):
    """Base exception for visualization errors."""

    pass


class InputShapeRequiredError(VisualizationError):
    """Raised when input_shape is required but not provided."""

    pass


class UnsupportedFrameworkError(VisualizationError):
    """Raised when the model framework is not supported."""

    pass


def visualize(
    model: Any,
    input_shape: Optional[tuple[int, ...]] = None,
    framework: FrameworkLiteral = "auto",
    show_shapes: bool = True,
    show_params: bool = True,
    group_blocks: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> graphviz.Digraph:
    """
    Visualize a neural network model architecture.

    Creates a clean architecture diagram showing layer flow, layer types,
    tensor shapes, and parameter counts. Works with both PyTorch and
    TensorFlow/Keras models.

    Args:
        model: A neural network model (PyTorch nn.Module or tf.keras.Model).
        input_shape: Shape of input tensor (required for PyTorch models).
                    Include batch dimension, e.g., (1, 3, 224, 224).
        framework: Framework to use ("auto", "pytorch", "tensorflow").
                  Default is "auto" which auto-detects the framework.
        show_shapes: Whether to display output shapes in the diagram.
        show_params: Whether to display parameter counts in the diagram.
        group_blocks: Whether to merge common patterns like Conv+ReLU.
        save_path: Optional file path to save the diagram (without extension).
                  Saves as PNG by default. Use ".svg" extension for SVG.
        title: Optional title for the diagram.

    Returns:
        graphviz.Digraph object that renders inline in Jupyter/Colab.

    Raises:
        InputShapeRequiredError: If PyTorch model is provided without input_shape.
        UnsupportedFrameworkError: If the model framework is not supported.
        VisualizationError: If visualization fails for any other reason.

    Example:
        >>> import torch.nn as nn
        >>> from modelviz import visualize
        >>>
        >>> model = nn.Sequential(
        ...     nn.Conv2d(1, 32, 3),
        ...     nn.ReLU(),
        ...     nn.MaxPool2d(2),
        ...     nn.Flatten(),
        ...     nn.Linear(32 * 13 * 13, 10)
        ... )
        >>> visualize(model, input_shape=(1, 1, 28, 28))  # Displays in notebook

    Example (TensorFlow):
        >>> import tensorflow as tf
        >>> from modelviz import visualize
        >>>
        >>> model = tf.keras.Sequential([
        ...     tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        ...     tf.keras.layers.Dense(10, activation='softmax')
        ... ])
        >>> visualize(model)  # Displays in notebook
    """
    # Detect or validate framework
    detected_framework = _resolve_framework(model, framework)

    # Parse model based on framework
    nodes = _parse_model(model, detected_framework, input_shape)

    # Apply block grouping if enabled
    if group_blocks:
        nodes = group_layers(nodes)

    # Build graph structure
    builder = GraphBuilder(nodes)
    graph = builder.build()

    # Render the graph
    digraph = render_graph(
        nodes=graph.nodes,
        edges=graph.edges,
        show_shapes=show_shapes,
        show_params=show_params,
        title=title,
    )

    # Save to file if path provided
    if save_path:
        _save_graph(digraph, save_path)

    return digraph


def _resolve_framework(
    model: Any,
    framework: FrameworkLiteral,
) -> Literal["pytorch", "tensorflow"]:
    """
    Resolve the framework to use for parsing.

    Args:
        model: The model object.
        framework: User-specified framework or "auto".

    Returns:
        Resolved framework identifier.
    """
    if framework == "auto":
        try:
            return detect_framework(model)
        except ValueError as e:
            raise UnsupportedFrameworkError(str(e))

    if framework in ("pytorch", "tensorflow"):
        return framework

    raise UnsupportedFrameworkError(
        f"Unsupported framework: {framework}. "
        f"Use 'auto', 'pytorch', or 'tensorflow'."
    )


def _parse_model(
    model: Any,
    framework: Literal["pytorch", "tensorflow"],
    input_shape: Optional[tuple[int, ...]],
) -> list[LayerNode]:
    """
    Parse the model using the appropriate framework parser.

    Args:
        model: The model object.
        framework: Framework identifier.
        input_shape: Input tensor shape (required for PyTorch).

    Returns:
        List of LayerNode objects.
    """
    if framework == "pytorch":
        if input_shape is None:
            raise InputShapeRequiredError(
                "input_shape is required for PyTorch models. "
                "Provide the shape including batch dimension, "
                "e.g., input_shape=(1, 3, 224, 224)"
            )

        from modelviz.parsers.torch_parser import (
            DynamicControlFlowError,
            ForwardPassError,
            parse_pytorch_model,
        )

        try:
            return parse_pytorch_model(model, input_shape)
        except DynamicControlFlowError as e:
            raise VisualizationError(str(e))
        except ForwardPassError as e:
            raise VisualizationError(str(e))

    elif framework == "tensorflow":
        from modelviz.parsers.tf_parser import parse_keras_model

        try:
            return parse_keras_model(model)
        except (TypeError, ValueError) as e:
            raise VisualizationError(f"Failed to parse Keras model: {e}")

    raise UnsupportedFrameworkError(f"Unknown framework: {framework}")


def _save_graph(digraph: graphviz.Digraph, save_path: str) -> None:
    """
    Save the graph to a file.

    Determines format from file extension. Defaults to PNG.
    """
    # Determine format from path
    if save_path.endswith(".svg"):
        format = "svg"
        filepath = save_path[:-4]
    elif save_path.endswith(".pdf"):
        format = "pdf"
        filepath = save_path[:-4]
    elif save_path.endswith(".png"):
        format = "png"
        filepath = save_path[:-4]
    else:
        format = "png"
        filepath = save_path

    digraph.format = format
    digraph.render(filepath, cleanup=True)


def visualize_3d(
    model: Any,
    input_shape: Optional[tuple[int, ...]] = None,
    framework: FrameworkLiteral = "auto",
    show_shapes: bool = True,
    show_params: bool = True,
    group_blocks: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    layout: Literal["tower", "spiral", "grid"] = "tower",
):
    """
    Visualize a neural network model in interactive 3D.

    Creates an interactive 3D visualization using Plotly that can be
    rotated, zoomed, and explored. Works in Jupyter/Colab notebooks.

    Args:
        model: A neural network model (PyTorch nn.Module or tf.keras.Model).
        input_shape: Shape of input tensor (required for PyTorch models).
        framework: Framework to use ("auto", "pytorch", "tensorflow").
        show_shapes: Whether to display output shapes in hover labels.
        show_params: Whether to display parameter counts in hover labels.
        group_blocks: Whether to merge common patterns like Conv+ReLU.
        save_path: Optional path to save (supports .png, .svg, .html).
        title: Optional title for the visualization.
        layout: 3D layout style - "tower", "spiral", or "grid".

    Returns:
        plotly.graph_objects.Figure that renders inline in notebooks.

    Example:
        >>> import torch.nn as nn
        >>> from modelviz import visualize_3d
        >>>
        >>> model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        >>> visualize_3d(model, input_shape=(1, 10))  # Interactive 3D view
    """
    from modelviz.renderers.plotly_renderer import render_3d_to_file, render_graph_3d

    # Detect or validate framework
    detected_framework = _resolve_framework(model, framework)

    # Parse model based on framework
    nodes = _parse_model(model, detected_framework, input_shape)

    # Apply block grouping if enabled
    if group_blocks:
        nodes = group_layers(nodes)

    # Build graph structure
    builder = GraphBuilder(nodes)
    graph = builder.build()

    # Render the 3D graph
    fig = render_graph_3d(
        nodes=graph.nodes,
        edges=graph.edges,
        show_shapes=show_shapes,
        show_params=show_params,
        title=title,
        layout=layout,
    )

    # Save to file if path provided
    if save_path:
        if save_path.endswith(".html"):
            fig.write_html(save_path)
        else:
            # Determine format
            if save_path.endswith(".svg"):
                format = "svg"
            elif save_path.endswith(".pdf"):
                format = "pdf"
            else:
                format = "png"
            fig.write_image(save_path, width=1200, height=800, scale=2)

    return fig


def visualize_threejs(
    model: Any,
    input_shape: Optional[tuple[int, ...]] = None,
    framework: FrameworkLiteral = "auto",
    show_shapes: bool = True,
    show_params: bool = True,
    group_blocks: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """
    Visualize a neural network model with Three.js 3D graphics.

    Creates a stunning interactive 3D visualization where each layer type
    has a distinct, meaningful 3D representation:

    - Conv: 3D boxes (depth = channels)
    - Linear: Flat planes
    - Activation: Glowing spheres
    - Pooling: Small cubes
    - BatchNorm: Thin slabs
    - Flatten: Cones
    - Dropout: Wireframe cubes
    - RNN/LSTM: Cylinders
    - Attention: Octahedrons

    Features:
    - Interactive rotation, zoom, and pan
    - Hover tooltips with layer details
    - Animated data flow particles
    - Beautiful dark theme with gradients

    Args:
        model: A neural network model (PyTorch nn.Module or tf.keras.Model).
        input_shape: Shape of input tensor (required for PyTorch models).
        framework: Framework to use ("auto", "pytorch", "tensorflow").
        show_shapes: Whether to show output shapes in tooltips.
        show_params: Whether to show parameter counts in tooltips.
        group_blocks: Whether to merge common patterns like Conv+ReLU.
        save_path: Path to save HTML file (recommended: .html extension).
        title: Optional title for the visualization.

    Returns:
        HTML string of the visualization. Open in a browser to view.

    Example:
        >>> import torch.nn as nn
        >>> from modelviz import visualize_threejs
        >>>
        >>> model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        >>> html = visualize_threejs(model, input_shape=(1, 10), save_path="model.html")
        >>> # Open model.html in browser for interactive 3D view
    """
    from modelviz.renderers.threejs_renderer import render_threejs

    # Detect or validate framework
    detected_framework = _resolve_framework(model, framework)

    # Parse model based on framework
    nodes = _parse_model(model, detected_framework, input_shape)

    # Apply block grouping if enabled
    if group_blocks:
        nodes = group_layers(nodes)

    # Build graph structure
    builder = GraphBuilder(nodes)
    graph = builder.build()

    # Render with Three.js
    html = render_threejs(
        nodes=graph.nodes,
        edges=graph.edges,
        show_shapes=show_shapes,
        show_params=show_params,
        title=title,
    )

    # Save to file if path provided
    if save_path:
        if not save_path.endswith(".html"):
            save_path = f"{save_path}.html"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)

    return html
