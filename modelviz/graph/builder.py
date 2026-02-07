"""
GraphBuilder for constructing the visualization graph from layer nodes.
"""

from dataclasses import dataclass, field
from typing import Optional

from modelviz.graph.layer_node import LayerNode


@dataclass
class Edge:
    """
    Represents an edge in the graph connecting two layer nodes.

    Attributes:
        source_id: ID of the source layer node.
        target_id: ID of the target layer node.
        edge_type: Type of connection ('sequential', 'skip', 'residual', 'concat').
        label: Optional label for the edge.
    """

    source_id: int
    target_id: int
    edge_type: str = "sequential"  # sequential, skip, residual, concat
    label: Optional[str] = None


@dataclass
class Graph:
    """
    Represents the complete neural network graph.

    Attributes:
        nodes: List of layer nodes in the graph.
        edges: List of edges connecting the nodes.
    """

    nodes: list[LayerNode] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    @property
    def total_params(self) -> int:
        """Calculate total parameters across all nodes."""
        return sum(node.params for node in self.nodes)


class GraphBuilder:
    """
    Builder class for constructing neural network visualization graphs.

    This class takes a list of LayerNode objects and constructs a graph
    with appropriate edges based on the execution order.

    For the MVP, this implements a linear graph where nodes are connected
    sequentially. Future versions will support branching graphs.

    Example:
        >>> nodes = [LayerNode(0, "conv1", "Conv2d"), LayerNode(1, "relu1", "ReLU")]
        >>> builder = GraphBuilder(nodes)
        >>> graph = builder.build()
        >>> len(graph.edges)
        1
    """

    def __init__(self, nodes: list[LayerNode]) -> None:
        """
        Initialize the graph builder with layer nodes.

        Args:
            nodes: List of LayerNode objects in execution order.
        """
        self._nodes = nodes
        self._edges: list[Edge] = []

    def build(self) -> Graph:
        """
        Build the graph with linear edges connecting sequential nodes.

        Returns:
            Graph object containing nodes and edges.

        # TODO: Add support for branching graphs (residual connections, skip connections)
        # TODO: Add support for transformer architectures with attention patterns
        """
        self._build_linear_edges()
        return Graph(nodes=self._nodes, edges=self._edges)

    def _build_linear_edges(self) -> None:
        """
        Build edges for a linear (sequential) graph.

        Connects each node to the next node in sequence.
        """
        for i in range(len(self._nodes) - 1):
            edge = Edge(
                source_id=self._nodes[i].id,
                target_id=self._nodes[i + 1].id,
            )
            self._edges.append(edge)

    # TODO: Implement branching graph support
    # def _build_branching_edges(self, branch_info: dict) -> None:
    #     """Build edges for graphs with branches (residual connections, etc.)."""
    #     pass

    # TODO: Implement transformer-specific graph building
    # def _build_transformer_graph(self) -> None:
    #     """Build graph with attention visualization for transformers."""
    #     pass
