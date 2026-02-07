"""Tests for layer grouping utilities."""

import pytest

from modelviz.graph.layer_node import LayerNode
from modelviz.utils.grouping import group_layers


class TestGroupLayers:
    """Test cases for layer grouping functionality."""

    def test_empty_list(self):
        """Test grouping with empty list."""
        result = group_layers([])
        assert result == []

    def test_single_node(self):
        """Test grouping with single node."""
        nodes = [LayerNode(0, "conv1", "Conv2d", params=320)]
        result = group_layers(nodes)

        assert len(result) == 1
        assert result[0].type == "Conv2d"

    def test_no_grouping_needed(self):
        """Test when no grouping patterns match."""
        nodes = [
            LayerNode(0, "conv1", "Conv2d", params=320),
            LayerNode(1, "pool1", "MaxPool2d", params=0),
        ]
        result = group_layers(nodes)

        assert len(result) == 2
        assert result[0].type == "Conv2d"
        assert result[1].type == "MaxPool2d"

    def test_group_conv_relu(self):
        """Test grouping Conv + ReLU pattern."""
        nodes = [
            LayerNode(0, "conv1", "Conv2d", output_shape=(1, 32, 28, 28), params=320),
            LayerNode(1, "relu1", "ReLU", output_shape=(1, 32, 28, 28), params=0),
        ]
        result = group_layers(nodes)

        assert len(result) == 1
        assert result[0].is_grouped is True
        assert result[0].grouped_types == ["Conv2d", "ReLU"]
        assert result[0].display_type == "Conv2d + ReLU"
        assert result[0].params == 320

    def test_group_linear_relu(self):
        """Test grouping Linear + ReLU pattern."""
        nodes = [
            LayerNode(0, "fc1", "Linear", output_shape=(1, 128), params=1024),
            LayerNode(1, "relu1", "ReLU", output_shape=(1, 128), params=0),
        ]
        result = group_layers(nodes)

        assert len(result) == 1
        assert result[0].is_grouped is True
        assert result[0].grouped_types == ["Linear", "ReLU"]

    def test_group_conv_bn_relu(self):
        """Test grouping Conv + BatchNorm + ReLU pattern."""
        nodes = [
            LayerNode(0, "conv1", "Conv2d", output_shape=(1, 32, 28, 28), params=320),
            LayerNode(1, "bn1", "BatchNorm2d", output_shape=(1, 32, 28, 28), params=64),
            LayerNode(2, "relu1", "ReLU", output_shape=(1, 32, 28, 28), params=0),
        ]
        result = group_layers(nodes)

        assert len(result) == 1
        assert result[0].is_grouped is True
        assert result[0].grouped_types == ["Conv2d", "BatchNorm2d", "ReLU"]
        assert result[0].params == 320 + 64  # Sum of all params

    def test_multiple_groups(self):
        """Test multiple consecutive grouping patterns."""
        nodes = [
            LayerNode(0, "conv1", "Conv2d", params=320),
            LayerNode(1, "relu1", "ReLU", params=0),
            LayerNode(2, "conv2", "Conv2d", params=640),
            LayerNode(3, "relu2", "ReLU", params=0),
        ]
        result = group_layers(nodes)

        assert len(result) == 2
        assert result[0].is_grouped is True
        assert result[1].is_grouped is True

    def test_mixed_grouping(self):
        """Test mix of grouped and ungrouped layers."""
        nodes = [
            LayerNode(0, "conv1", "Conv2d", params=320),
            LayerNode(1, "relu1", "ReLU", params=0),
            LayerNode(2, "pool1", "MaxPool2d", params=0),
            LayerNode(3, "flatten", "Flatten", params=0),
        ]
        result = group_layers(nodes)

        assert len(result) == 3
        assert result[0].is_grouped is True  # Conv + ReLU
        assert result[1].is_grouped is False  # MaxPool
        assert result[2].is_grouped is False  # Flatten

    def test_ids_reassigned(self):
        """Test that IDs are properly reassigned after grouping."""
        nodes = [
            LayerNode(0, "conv1", "Conv2d", params=320),
            LayerNode(1, "relu1", "ReLU", params=0),
            LayerNode(2, "pool1", "MaxPool2d", params=0),
        ]
        result = group_layers(nodes)

        assert result[0].id == 0
        assert result[1].id == 1

    def test_output_shape_preserved(self):
        """Test that output shape comes from last node in group."""
        nodes = [
            LayerNode(
                0,
                "conv1",
                "Conv2d",
                input_shape=(1, 1, 28, 28),
                output_shape=(1, 32, 26, 26),
                params=320,
            ),
            LayerNode(
                1,
                "relu1",
                "ReLU",
                input_shape=(1, 32, 26, 26),
                output_shape=(1, 32, 26, 26),
                params=0,
            ),
        ]
        result = group_layers(nodes)

        assert result[0].input_shape == (1, 1, 28, 28)
        assert result[0].output_shape == (1, 32, 26, 26)
