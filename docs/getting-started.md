# Getting Started with modelviz

This guide will help you install modelviz and create your first neural network visualization.

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Install from PyPI

```bash
pip install modelviz-ai
```

### Install with framework support

```bash
# For PyTorch models
pip install modelviz-ai[torch]

# For TensorFlow/Keras models
pip install modelviz-ai[tf]

# For both frameworks
pip install modelviz-ai[torch,tf]
```

### System dependency (for 2D diagrams only)

For Graphviz 2D diagrams, install the system package:

```bash
# macOS
brew install graphviz

# Ubuntu/Debian
sudo apt-get install graphviz

# Windows
conda install -c conda-forge graphviz
```

> **Note**: Three.js 3D visualizations work without any system dependencies.

## Your First Visualization

### 1. Create a simple PyTorch model

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

### 2. Generate a 2D diagram

```python
from modelviz import visualize

# Renders inline in Jupyter notebooks
graph = visualize(model, input_shape=(1, 784))

# Save to file
visualize(model, input_shape=(1, 784), save_path="my_model.png")
```

### 3. Generate an interactive 3D visualization

```python
from modelviz import visualize_threejs

# Creates an HTML file you can open in any browser
visualize_threejs(
    model, 
    input_shape=(1, 784), 
    save_path="my_model_3d.html"
)
```

## Understanding the Output

### 2D Diagrams

The 2D Graphviz output shows:
- **Layer type** as the node label
- **Output shape** below the layer name
- **Parameter count** at the bottom
- **Color coding** by layer type

### 3D Visualizations

The Three.js 3D output includes:
- **Distinct shapes** for each layer type (boxes, spheres, planes, etc.)
- **Floating labels** above each shape
- **Animated particles** showing data flow
- **Interactive controls** (rotate, zoom, pan)

## Layer Grouping

By default, modelviz groups common layer patterns:

| Pattern | Grouped As |
|---------|------------|
| Conv2d → ReLU | Conv2d + ReLU |
| Conv2d → BatchNorm2d → ReLU | Conv2d + BatchNorm2d + ReLU |
| Linear → ReLU | Linear + ReLU |

This creates cleaner, more readable diagrams. Disable with:

```python
visualize(model, input_shape, group_blocks=False)
```

## TensorFlow/Keras Models

Keras models don't require `input_shape` since they're already built:

```python
import tensorflow as tf
from modelviz import visualize

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# No input_shape needed!
visualize(model, save_path="keras_model.svg")
```

## Next Steps

- See [API Reference](api.md) for detailed function documentation
- Check out [Examples](examples.md) for more complex models
- Learn about [3D Visualization](threejs.md) features
