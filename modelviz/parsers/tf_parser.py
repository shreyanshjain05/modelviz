"""
TensorFlow/Keras model parser for modelviz.

Extracts layer information from tf.keras.Model objects.
"""

from typing import Any

from modelviz.graph.layer_node import LayerNode


def parse_keras_model(model: Any) -> list[LayerNode]:
    """
    Parse a TensorFlow/Keras model and extract layer information.

    Extracts information from model.layers including layer names,
    types, output shapes, and parameter counts.

    Args:
        model: A tf.keras.Model or keras.Model object.

    Returns:
        List of LayerNode objects representing the model's layers.

    Raises:
        TypeError: If the model is not a Keras model.
        ValueError: If the model has not been built.

    Example:
        >>> import tensorflow as tf
        >>> model = tf.keras.Sequential([
        ...     tf.keras.layers.Dense(10, input_shape=(5,)),
        ...     tf.keras.layers.ReLU()
        ... ])
        >>> nodes = parse_keras_model(model)
        >>> len(nodes)
        2
    """
    # Validate input - check for both tf.keras and standalone keras
    if not _is_keras_model(model):
        raise TypeError(f"Expected tf.keras.Model or keras.Model, got {type(model)}")

    # Check if model is built
    if not hasattr(model, "layers") or not model.layers:
        raise ValueError(
            "Model has no layers. Make sure the model is built by calling "
            "model.build() or by passing input_shape to the first layer."
        )

    nodes: list[LayerNode] = []

    for idx, layer in enumerate(model.layers):
        # Get layer name and type
        name = layer.name
        layer_type = layer.__class__.__name__

        # Get output shape
        output_shape = _get_output_shape(layer)

        # Get input shape (from the layer's input if available)
        input_shape = _get_input_shape(layer)

        # Get parameter count
        try:
            params = layer.count_params()
        except (ValueError, RuntimeError):
            # Layer may not be built yet
            params = 0

        node = LayerNode(
            id=idx,
            name=name,
            type=layer_type,
            input_shape=input_shape,
            output_shape=output_shape,
            params=params,
        )
        nodes.append(node)

    return nodes


def _is_keras_model(model: Any) -> bool:
    """Check if model is a Keras Model (tf.keras or standalone keras)."""
    try:
        import tensorflow as tf

        if isinstance(model, tf.keras.Model):
            return True
    except ImportError:
        pass

    try:
        import keras

        if isinstance(model, keras.Model):
            return True
    except ImportError:
        pass

    return False


def _get_output_shape(layer: Any) -> tuple[int, ...] | None:
    """
    Extract output shape from a Keras layer.

    Handles various output shape formats that Keras may return.
    """
    try:
        output_shape = layer.output_shape

        if output_shape is None:
            return None

        # Handle single output shape
        if isinstance(output_shape, tuple):
            # Convert TensorShape or list to tuple of ints
            return _normalize_shape(output_shape)

        # Handle multiple outputs (list of shapes)
        if isinstance(output_shape, list) and len(output_shape) > 0:
            return _normalize_shape(output_shape[0])

    except (AttributeError, RuntimeError):
        # Layer may not be built
        pass

    return None


def _get_input_shape(layer: Any) -> tuple[int, ...] | None:
    """
    Extract input shape from a Keras layer.

    Handles various input shape formats that Keras may return.
    """
    try:
        # Try to get input_shape attribute
        if hasattr(layer, "input_shape"):
            input_shape = layer.input_shape

            if input_shape is None:
                return None

            if isinstance(input_shape, tuple):
                return _normalize_shape(input_shape)

            if isinstance(input_shape, list) and len(input_shape) > 0:
                return _normalize_shape(input_shape[0])

    except (AttributeError, RuntimeError):
        pass

    return None


def _normalize_shape(shape: Any) -> tuple[int, ...]:
    """
    Normalize a shape to a tuple of integers.

    Handles TensorShape, lists, and tuples with None values.
    """
    if hasattr(shape, "as_list"):
        # TensorFlow TensorShape
        shape = shape.as_list()

    # Convert to tuple, replacing None with -1 for display
    result = []
    for dim in shape:
        if dim is None:
            result.append(-1)  # Use -1 to represent dynamic dimensions
        else:
            result.append(int(dim))

    return tuple(result)


def get_keras_summary(model: Any) -> dict[str, Any]:
    """
    Get a summary of the Keras model.

    Args:
        model: A tf.keras.Model or keras.Model object.

    Returns:
        Dictionary with model summary information.
    """
    nodes = parse_keras_model(model)
    total_params = sum(node.params for node in nodes)

    return {
        "num_layers": len(nodes),
        "total_params": total_params,
        "layers": [
            {
                "name": node.name,
                "type": node.type,
                "output_shape": node.output_shape,
                "params": node.params,
            }
            for node in nodes
        ],
    }
