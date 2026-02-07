# Three.js 3D Visualization

modelviz includes a stunning Three.js-based 3D renderer that creates interactive visualizations of neural network architectures.

## Overview

The Three.js renderer creates self-contained HTML files with:
- **Distinct 3D shapes** for each layer type
- **Horizontal layout** with data flowing left to right
- **Text labels** above each layer
- **Animated particles** showing data flow
- **Interactive controls** for exploration

## Usage

```python
from modelviz import visualize_threejs

html = visualize_threejs(
    model,
    input_shape=(1, 3, 224, 224),
    title="My Network",
    save_path="network.html"
)

# Open network.html in any web browser
```

## Layer Shapes

Each layer type has a semantically meaningful 3D representation:

| Layer Type | Shape | Rationale |
|------------|-------|-----------|
| **Conv2d** | 3D Box | Feature maps are 3D volumes (Channels × Height × Width) |
| **Linear/Dense** | Flat Plane | Weight matrix is 2D (input features × output features) |
| **Pooling** | Small Cube | Reduces spatial dimensions → smaller representation |
| **Activation** | Glowing Sphere | Element-wise operation applied uniformly |
| **BatchNorm** | Thin Slab | Normalizes across batch, "flattens" distribution |
| **Flatten** | Cone | Funnels multi-dimensional data into 1D vector |
| **Dropout** | Wireframe Cube | Sparse/transparent = random neurons "dropped" |
| **RNN/LSTM** | Cylinder | Circular shape suggests recurrent/cyclical flow |
| **Attention** | Octahedron | Multi-faceted for multi-head attention patterns |

## Color Scheme

| Layer Type | Color | Hex Code |
|------------|-------|----------|
| Convolution | Indigo | `#6366f1` |
| Linear/Dense | Purple | `#8b5cf6` |
| Pooling | Cyan | `#06b6d4` |
| Normalization | Emerald | `#10b981` |
| Activation | Amber | `#f59e0b` |
| Dropout | Red | `#ef4444` |
| Flatten | Pink | `#ec4899` |
| Embedding | Lime | `#84cc16` |
| RNN/LSTM | Teal | `#14b8a6` |
| Attention | Orange | `#f97316` |

## Interactive Controls

| Action | Mouse/Keyboard |
|--------|----------------|
| **Rotate** | Click and drag |
| **Zoom** | Scroll wheel |
| **Pan** | Shift + drag |
| **Layer details** | Hover over shape |

## Labels

Each 3D shape has a floating label showing:
- **Layer type** (e.g., "Conv2d + BatchNorm2d + ReLU")
- **Output dimensions** (e.g., "16×16")

Labels always face the camera for readability.

## Tooltips

Hovering over any layer shows detailed information:
- Full layer name
- Complete output shape
- Parameter count
- Whether grouped with other layers

## Animations

- **Data flow particles**: Blue spheres animate along connection lines
- **Activation rotation**: Spheres slowly rotate
- **Attention oscillation**: Octahedrons gently rotate

## Examples

### Simple MLP

```python
import torch.nn as nn
from modelviz import visualize_threejs

model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

visualize_threejs(model, input_shape=(1, 784), save_path="mlp.html")
```

### CNN

```python
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(128 * 8 * 8, 10)
)

visualize_threejs(model, input_shape=(1, 3, 32, 32), save_path="cnn.html")
```

### Without grouping

To see all individual layers:

```python
visualize_threejs(
    model,
    input_shape=(1, 3, 32, 32),
    group_blocks=False,  # Show Conv, BatchNorm, ReLU separately
    save_path="cnn_detailed.html"
)
```

## Technical Details

- **Renderer**: Three.js r160 with WebGL
- **Labels**: CSS2DRenderer for crisp text
- **Self-contained**: No external dependencies (CDN imports)
- **File size**: ~25-35 KB per visualization
- **Browser support**: Chrome, Firefox, Safari, Edge (modern versions)
