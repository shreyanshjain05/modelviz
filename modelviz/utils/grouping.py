"""
Layer grouping utilities for modelviz.

Implements sliding window pattern matching to merge common layer patterns
into single visual nodes for cleaner diagrams.
"""

from typing import Sequence

from modelviz.graph.layer_node import LayerNode

# Common patterns to group together (as tuples of layer type keywords)
GROUPING_PATTERNS: list[tuple[str, ...]] = [
    # Conv + BatchNorm + ReLU (most specific first)
    ("Conv", "BatchNorm", "ReLU"),
    ("Conv", "BatchNorm", "LeakyReLU"),
    ("Conv", "BatchNorm", "GELU"),
    # Conv + ReLU
    ("Conv", "ReLU"),
    ("Conv", "LeakyReLU"),
    ("Conv", "GELU"),
    ("Conv", "Sigmoid"),
    ("Conv", "Tanh"),
    # Linear + ReLU
    ("Linear", "ReLU"),
    ("Linear", "LeakyReLU"),
    ("Linear", "GELU"),
    ("Linear", "Sigmoid"),
    ("Linear", "Tanh"),
    # Dense + ReLU (Keras)
    ("Dense", "ReLU"),
    ("Dense", "LeakyReLU"),
    ("Dense", "Activation"),
    # BatchNorm + ReLU
    ("BatchNorm", "ReLU"),
    ("BatchNorm", "LeakyReLU"),
]


def group_layers(nodes: Sequence[LayerNode]) -> list[LayerNode]:
    """
    Group consecutive layers that match common patterns.

    Uses sliding window pattern matching to merge patterns like:
    - Conv + ReLU
    - Conv + BatchNorm + ReLU
    - Linear + ReLU

    Args:
        nodes: Sequence of LayerNode objects to process.

    Returns:
        New list of LayerNode objects with grouped patterns merged.

    Example:
        >>> nodes = [
        ...     LayerNode(0, "conv1", "Conv2d", output_shape=(1, 32, 28, 28), params=320),
        ...     LayerNode(1, "relu1", "ReLU", output_shape=(1, 32, 28, 28), params=0),
        ... ]
        >>> grouped = group_layers(nodes)
        >>> len(grouped)
        1
        >>> grouped[0].display_type
        'Conv2d + ReLU'
    """
    if len(nodes) <= 1:
        return list(nodes)

    result: list[LayerNode] = []
    i = 0

    while i < len(nodes):
        # Try to match patterns starting from current position
        matched = False

        # Try longer patterns first
        for pattern in GROUPING_PATTERNS:
            pattern_len = len(pattern)
            if i + pattern_len <= len(nodes):
                if _matches_pattern(nodes[i : i + pattern_len], pattern):
                    # Create grouped node
                    grouped_node = _create_grouped_node(
                        nodes[i : i + pattern_len],
                        new_id=len(result),
                    )
                    result.append(grouped_node)
                    i += pattern_len
                    matched = True
                    break

        if not matched:
            # No pattern matched, keep the node as-is with new ID
            node = nodes[i]
            result.append(
                LayerNode(
                    id=len(result),
                    name=node.name,
                    type=node.type,
                    input_shape=node.input_shape,
                    output_shape=node.output_shape,
                    params=node.params,
                    is_grouped=node.is_grouped,
                    grouped_types=node.grouped_types,
                )
            )
            i += 1

    return result


def _matches_pattern(nodes: Sequence[LayerNode], pattern: tuple[str, ...]) -> bool:
    """
    Check if a sequence of nodes matches a pattern.

    Pattern matching is case-insensitive and uses substring matching
    to handle variations like Conv1d, Conv2d, Conv3d.
    """
    if len(nodes) != len(pattern):
        return False

    for node, pattern_type in zip(nodes, pattern):
        if not _type_matches(node.type, pattern_type):
            return False

    return True


def _type_matches(layer_type: str, pattern_type: str) -> bool:
    """
    Check if a layer type matches a pattern type.

    Uses case-insensitive substring matching.
    """
    return pattern_type.lower() in layer_type.lower()


def _create_grouped_node(
    nodes: Sequence[LayerNode],
    new_id: int,
) -> LayerNode:
    """
    Create a single grouped node from multiple nodes.

    The grouped node uses:
    - Input shape from the first node
    - Output shape from the last node
    - Sum of all parameters
    - Combined name from all nodes
    """
    first_node = nodes[0]
    last_node = nodes[-1]

    # Create combined name
    names = [n.name for n in nodes]
    combined_name = " → ".join(names)

    # Collect all types
    types = [n.type for n in nodes]

    # Sum parameters
    total_params = sum(n.params for n in nodes)

    return LayerNode(
        id=new_id,
        name=combined_name,
        type=first_node.type,  # Primary type is the first layer
        input_shape=first_node.input_shape,
        output_shape=last_node.output_shape,
        params=total_params,
        is_grouped=True,
        grouped_types=types,
    )
