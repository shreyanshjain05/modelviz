"""Tests for framework detection."""

import pytest

from modelviz.utils.framework_detect import detect_framework, is_framework_available


class TestFrameworkDetection:
    """Test cases for framework detection."""

    def test_unsupported_type_raises_error(self):
        """Test that unsupported types raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported model type"):
            detect_framework("not a model")

    def test_unsupported_type_error_message(self):
        """Test error message includes type information."""
        with pytest.raises(ValueError) as exc_info:
            detect_framework([1, 2, 3])

        assert "list" in str(exc_info.value).lower()


class TestPyTorchDetection:
    """Test cases for PyTorch model detection."""

    @pytest.fixture
    def pytorch_available(self):
        """Check if PyTorch is available."""
        if not is_framework_available("pytorch"):
            pytest.skip("PyTorch not installed")

    def test_detect_pytorch_linear(self, pytorch_available):
        """Test detection of PyTorch Linear layer."""
        import torch.nn as nn

        model = nn.Linear(10, 5)
        assert detect_framework(model) == "pytorch"

    def test_detect_pytorch_sequential(self, pytorch_available):
        """Test detection of PyTorch Sequential."""
        import torch.nn as nn

        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        assert detect_framework(model) == "pytorch"


class TestKerasDetection:
    """Test cases for TensorFlow/Keras model detection."""

    @pytest.fixture
    def tensorflow_available(self):
        """Check if TensorFlow is available."""
        if not is_framework_available("tensorflow"):
            pytest.skip("TensorFlow not installed")

    def test_detect_keras_sequential(self, tensorflow_available):
        """Test detection of Keras Sequential model."""
        import tensorflow as tf

        model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(5,))])
        assert detect_framework(model) == "tensorflow"

    def test_detect_keras_functional(self, tensorflow_available):
        """Test detection of Keras Functional model."""
        import tensorflow as tf

        inputs = tf.keras.Input(shape=(10,))
        outputs = tf.keras.layers.Dense(5)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        assert detect_framework(model) == "tensorflow"


class TestIsFrameworkAvailable:
    """Test cases for framework availability checking."""

    def test_invalid_framework(self):
        """Test that invalid framework returns False."""
        assert is_framework_available("invalid") is False
