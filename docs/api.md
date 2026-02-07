# API Reference

Complete API documentation for modelviz.

## Main Functions

### `visualize()`

Generate a 2D Graphviz diagram of a neural network.

```python
def visualize(
    model: Any,
    input_shape: Optional[tuple[int, ...]] = None,
    framework: Literal["auto", "pytorch", "tensorflow"] = "auto",
    show_shapes: bool = True,
    show_params: bool = True,
    group_blocks: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> graphviz.Digraph
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `Any` | required | PyTorch `nn.Module` or Keras `Model` |
| `input_shape` | `tuple[int, ...]` | `None` | Input tensor shape. Required for PyTorch |
| `framework` | `str` | `"auto"` | Framework: `"auto"`, `"pytorch"`, `"tensorflow"` |
| `show_shapes` | `bool` | `True` | Display output tensor shapes |
| `show_params` | `bool` | `True` | Display parameter counts |
| `group_blocks` | `bool` | `True` | Merge common patterns (Conv+ReLU) |
| `save_path` | `str` | `None` | File path to save (`.png`, `.svg`, `.pdf`) |
| `title` | `str` | `None` | Diagram title |

#### Returns

`graphviz.Digraph` — Graphviz Digraph object that renders inline in notebooks.

#### Example

```python
from modelviz import visualize
import torch.nn as nn

model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
graph = visualize(model, input_shape=(1, 10), title="MLP")
```

---

### `visualize_threejs()`

Generate an interactive 3D Three.js visualization.

```python
def visualize_threejs(
    model: Any,
    input_shape: Optional[tuple[int, ...]] = None,
    framework: Literal["auto", "pytorch", "tensorflow"] = "auto",
    show_shapes: bool = True,
    show_params: bool = True,
    group_blocks: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> str
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `Any` | required | PyTorch `nn.Module` or Keras `Model` |
| `input_shape` | `tuple[int, ...]` | `None` | Input tensor shape. Required for PyTorch |
| `framework` | `str` | `"auto"` | Framework: `"auto"`, `"pytorch"`, `"tensorflow"` |
| `show_shapes` | `bool` | `True` | Display shapes in labels |
| `show_params` | `bool` | `True` | Display params in tooltips |
| `group_blocks` | `bool` | `True` | Merge common patterns |
| `save_path` | `str` | `None` | File path to save (`.html`) |
| `title` | `str` | `None` | Visualization title |

#### Returns

`str` — Complete HTML string with embedded Three.js visualization.

#### Example

```python
from modelviz import visualize_threejs

html = visualize_threejs(
    model,
    input_shape=(1, 3, 224, 224),
    save_path="model.html",
    title="CNN 3D View"
)
```

---

### `visualize_3d()`

Generate a Plotly 3D visualization (simpler alternative to Three.js).

```python
def visualize_3d(
    model: Any,
    input_shape: Optional[tuple[int, ...]] = None,
    framework: Literal["auto", "pytorch", "tensorflow"] = "auto",
    show_shapes: bool = True,
    show_params: bool = True,
    group_blocks: bool = True,
    layout: Literal["tower", "spiral", "grid"] = "tower",
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plotly.graph_objects.Figure
```

#### Parameters

Same as `visualize_threejs()` plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layout` | `str` | `"tower"` | 3D layout: `"tower"`, `"spiral"`, `"grid"` |

#### Returns

`plotly.graph_objects.Figure` — Interactive Plotly figure.

---

## Data Classes

### `LayerNode`

Represents a single layer in the neural network.

```python
@dataclass
class LayerNode:
    id: int                                    # Unique layer ID
    name: str                                  # Full layer name (e.g., "features.conv1")
    type: str                                  # Layer type (e.g., "Conv2d")
    input_shape: Optional[tuple[int, ...]]     # Input tensor shape
    output_shape: Optional[tuple[int, ...]]    # Output tensor shape
    params: int                                # Number of trainable parameters
    is_grouped: bool = False                   # Whether merged with adjacent layers
    
    @property
    def display_type(self) -> str:            # For grouped: "Conv2d + ReLU"
    
    @property
    def formatted_output_shape(self) -> str:  # "(1, 64, 28, 28)"
    
    @property
    def formatted_params(self) -> str:        # "1.2M" or "4.5K"
```

---

## Utility Functions

### `detect_framework()`

Detect whether a model is PyTorch or TensorFlow.

```python
from modelviz.utils import detect_framework

framework = detect_framework(model)  # Returns "pytorch" or "tensorflow"
```

### `group_layers()`

Merge common layer patterns.

```python
from modelviz.utils import group_layers

nodes = [...]  # List of LayerNode
grouped = group_layers(nodes)  # Returns grouped LayerNode list
```

### `trace_pytorch_graph()`

Trace a PyTorch model with skip connections using torch.fx.

```python
from modelviz.parsers import trace_pytorch_graph

nodes, edges = trace_pytorch_graph(model, input_shape)
# edges include edge_type: 'sequential', 'residual', 'concat'
```

### `has_skip_connections()`

Check if a model has skip/residual connections.

```python
from modelviz.parsers import has_skip_connections

if has_skip_connections(model):
    print("Model has skip connections!")
```

---

## Exceptions

### `VisualizationError`

Base exception for all modelviz errors.

### `InputShapeRequiredError`

Raised when `input_shape` is required but not provided (PyTorch models).

### `UnsupportedFrameworkError`

Raised when model type is not supported.
