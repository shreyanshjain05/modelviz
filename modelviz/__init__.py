"""
modelviz - Framework-agnostic neural network architecture visualization.

A Python library for visualizing neural network architectures in Jupyter notebooks,
supporting both PyTorch and TensorFlow/Keras models.
"""

from modelviz.graph.layer_node import LayerNode
from modelviz.visualize import visualize, visualize_3d, visualize_threejs

__version__ = "0.1.0"
__all__ = ["visualize", "visualize_3d", "visualize_threejs", "LayerNode", "__version__"]
