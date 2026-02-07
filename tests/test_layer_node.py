"""Tests for LayerNode dataclass."""

import pytest

from modelviz.graph.layer_node import LayerNode


class TestLayerNode:
    """Test cases for LayerNode dataclass."""

    def test_create_basic_node(self):
        """Test creating a basic layer node."""
        node = LayerNode(
            id=0,
            name="conv1",
            type="Conv2d",
            output_shape=(1, 32, 28, 28),
            params=320,
        )

        assert node.id == 0
        assert node.name == "conv1"
        assert node.type == "Conv2d"
        assert node.output_shape == (1, 32, 28, 28)
        assert node.params == 320
        assert node.is_grouped is False
        assert node.grouped_types == []

    def test_invalid_id_raises_error(self):
        """Test that negative id raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            LayerNode(id=-1, name="conv1", type="Conv2d")

    def test_empty_name_raises_error(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            LayerNode(id=0, name="", type="Conv2d")

    def test_empty_type_raises_error(self):
        """Test that empty type raises ValueError."""
        with pytest.raises(ValueError, match="type cannot be empty"):
            LayerNode(id=0, name="conv1", type="")

    def test_display_type_regular(self):
        """Test display_type for regular nodes."""
        node = LayerNode(id=0, name="conv1", type="Conv2d")
        assert node.display_type == "Conv2d"

    def test_display_type_grouped(self):
        """Test display_type for grouped nodes."""
        node = LayerNode(
            id=0,
            name="block1",
            type="Conv2d",
            is_grouped=True,
            grouped_types=["Conv2d", "ReLU"],
        )
        assert node.display_type == "Conv2d + ReLU"

    def test_formatted_output_shape(self):
        """Test formatted output shape."""
        node = LayerNode(
            id=0, name="conv1", type="Conv2d", output_shape=(1, 32, 28, 28)
        )
        assert node.formatted_output_shape == "(1, 32, 28, 28)"

    def test_formatted_output_shape_none(self):
        """Test formatted output shape when None."""
        node = LayerNode(id=0, name="conv1", type="Conv2d")
        assert node.formatted_output_shape == "?"

    def test_formatted_params_small(self):
        """Test formatted params for small counts."""
        node = LayerNode(id=0, name="conv1", type="Conv2d", params=320)
        assert node.formatted_params == "320"

    def test_formatted_params_thousands(self):
        """Test formatted params for thousands."""
        node = LayerNode(id=0, name="conv1", type="Conv2d", params=5_120)
        assert node.formatted_params == "5.1K"

    def test_formatted_params_millions(self):
        """Test formatted params for millions."""
        node = LayerNode(id=0, name="fc1", type="Linear", params=1_234_567)
        assert node.formatted_params == "1.23M"

    def test_repr(self):
        """Test string representation."""
        node = LayerNode(
            id=0, name="conv1", type="Conv2d", output_shape=(1, 32, 28, 28), params=320
        )
        repr_str = repr(node)

        assert "LayerNode" in repr_str
        assert "conv1" in repr_str
        assert "Conv2d" in repr_str
