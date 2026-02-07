"""
Framework detection utilities for modelviz.

Uses lazy imports to avoid requiring both PyTorch and TensorFlow.
"""

from typing import Any, Literal

FrameworkType = Literal["pytorch", "tensorflow"]


def detect_framework(model: Any) -> FrameworkType:
    """
    Auto-detect the deep learning framework of a model.

    Uses lazy imports to check framework types without requiring
    both frameworks to be installed.

    Args:
        model: A neural network model object.

    Returns:
        Framework identifier: "pytorch" or "tensorflow".

    Raises:
        ValueError: If the model type is not recognized or
                   neither framework is available.

    Example:
        >>> import torch.nn as nn
        >>> model = nn.Linear(10, 5)
        >>> detect_framework(model)
        'pytorch'
    """
    # Try PyTorch first (lazy import)
    if _is_pytorch_model(model):
        return "pytorch"

    # Try TensorFlow/Keras (lazy import)
    if _is_keras_model(model):
        return "tensorflow"

    # Model type not recognized
    model_type = type(model).__module__ + "." + type(model).__name__
    raise ValueError(
        f"Unsupported model type: {model_type}. "
        f"modelviz supports PyTorch (torch.nn.Module) and "
        f"TensorFlow/Keras (tf.keras.Model) models."
    )


def _is_pytorch_model(model: Any) -> bool:
    """
    Check if model is a PyTorch nn.Module.

    Uses lazy import to avoid requiring PyTorch installation.
    """
    try:
        import torch.nn as nn

        return isinstance(model, nn.Module)
    except ImportError:
        return False


def _is_keras_model(model: Any) -> bool:
    """
    Check if model is a TensorFlow/Keras Model.

    Uses lazy import to avoid requiring TensorFlow installation.
    Handles both tf.keras.Model and standalone keras.Model.
    """
    # Try tf.keras first
    try:
        import tensorflow as tf

        if isinstance(model, tf.keras.Model):
            return True
    except ImportError:
        pass

    # Try standalone keras
    try:
        import keras

        if isinstance(model, keras.Model):
            return True
    except ImportError:
        pass

    return False


def is_framework_available(framework: FrameworkType) -> bool:
    """
    Check if a framework is available for import.

    Args:
        framework: Framework to check ("pytorch" or "tensorflow").

    Returns:
        True if the framework can be imported, False otherwise.
    """
    if framework == "pytorch":
        try:
            import torch  # noqa: F401

            return True
        except ImportError:
            return False
    elif framework == "tensorflow":
        try:
            import tensorflow  # noqa: F401

            return True
        except ImportError:
            try:
                import keras  # noqa: F401

                return True
            except ImportError:
                return False
    return False
