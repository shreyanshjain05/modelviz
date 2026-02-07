"""
LayerNode dataclass representing a single layer in the neural network graph.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LayerNode:
    """
    Represents a single layer node in the neural network architecture graph.

    This is the unified schema used by both PyTorch and TensorFlow parsers
    to represent layer information in a framework-agnostic way.

    Attributes:
        id: Unique identifier for the layer node.
        name: Human-readable name of the layer (e.g., "conv1", "fc2").
        type: Type of the layer (e.g., "Conv2d", "Linear", "Dense").
        input_shape: Input tensor shape, None if unknown.
        output_shape: Output tensor shape, None if unknown.
        params: Number of trainable parameters in the layer.
        is_grouped: Whether this node represents a grouped block of layers.
        grouped_types: List of layer types if this is a grouped node.
    """

    id: int
    name: str
    type: str
    input_shape: Optional[tuple[int, ...]] = None
    output_shape: Optional[tuple[int, ...]] = None
    params: int = 0
    is_grouped: bool = False
    grouped_types: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate the layer node after initialization."""
        if self.id < 0:
            raise ValueError(f"Layer id must be non-negative, got {self.id}")
        if not self.name:
            raise ValueError("Layer name cannot be empty")
        if not self.type:
            raise ValueError("Layer type cannot be empty")

    @property
    def display_type(self) -> str:
        """
        Get the display type for rendering.

        For grouped nodes, shows all grouped types joined with '+'.
        For regular nodes, shows the layer type.
        """
        if self.is_grouped and self.grouped_types:
            return " + ".join(self.grouped_types)
        return self.type

    @property
    def formatted_output_shape(self) -> str:
        """Format output shape for display."""
        if self.output_shape is None:
            return "?"
        return str(self.output_shape)

    @property
    def formatted_params(self) -> str:
        """Format parameter count for display with K/M suffixes."""
        if self.params >= 1_000_000:
            return f"{self.params / 1_000_000:.2f}M"
        elif self.params >= 1_000:
            return f"{self.params / 1_000:.1f}K"
        return str(self.params)

    def __repr__(self) -> str:
        return (
            f"LayerNode(id={self.id}, name='{self.name}', type='{self.type}', "
            f"output_shape={self.output_shape}, params={self.params})"
        )
