<p align="center">
  <img src="docs/assets/logo.png" alt="modelviz" width="180"/>
</p>

<h1 align="center">modelviz-ai</h1>

<p align="center">
  <strong>Framework-agnostic neural network visualization for Jupyter notebooks</strong>
</p>

<p align="center">
  <a href="https://shreyanshjain05.github.io/modelviz/">Documentation</a> •
  <a href="#-features">Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-examples">Examples</a> •
  <a href="#-3d-visualization">3D Visualization</a> •
  <a href="#-api-reference">API</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

**modelviz** generates beautiful, publication-ready neural network architecture diagrams from your PyTorch and TensorFlow/Keras models. Simply pass your model object and get a stunning visualization — no manual diagram creation required.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Auto-detection** | Automatically detects PyTorch and TensorFlow/Keras models |
| 📊 **2D Diagrams** | Clean Graphviz diagrams with layer types, shapes, and parameters |
| 🎮 **3D Interactive** | Stunning Three.js visualizations with distinct shapes per layer |
| 🔄 **Skip Connections** | ResNet-style residual paths, dense connections, and branching architectures |
| 🎨 **Smart Styling** | Color-coded nodes for Conv, Linear, Pooling, Activation layers |
| 📦 **Block Grouping** | Auto-merges common patterns (Conv+ReLU, Conv+BN+ReLU) |
| 📓 **Notebook-native** | Renders inline in Jupyter, Colab, and VSCode notebooks |
| 💾 **Export** | Save as PNG, SVG, PDF, or interactive HTML |

## 🖥️ Demo


https://github.com/user-attachments/assets/4ed4c537-5d3e-45ba-a08a-cafbe57f5fbb



## 🎮 3D Visualization Preview

Each layer type has a distinct, meaningful 3D representation:

| Layer | Shape | Rationale |
|-------|-------|-----------|
| **Conv2d** | 3D Box | Feature maps are 3D volumes (C×H×W) |
| **Linear** | Flat Plane | Weight matrix is 2D |
| **Pooling** | Small Cube | Reduces spatial dimensions |
| **Activation** | Sphere | Element-wise uniform operation |
| **BatchNorm** | Thin Slab | Normalizes distribution |
| **Flatten** | Cone | Funnels data to 1D |
| **Dropout** | Wireframe | Sparse/dropped neurons |
| **RNN/LSTM** | Cylinder | Recurrent/cyclical flow |
| **Attention** | Octahedron | Multi-head patterns |

## 🚀 Installation

### From PyPI

```bash
# Basic installation
pip install modelviz-ai

# With PyTorch support
pip install modelviz-ai[torch]

# With TensorFlow support
pip install modelviz-ai[tf]

# All frameworks + development tools
pip install modelviz-ai[all,dev]
```

### From Source

```bash
git clone https://github.com/shreyanshjain05/modelviz.git
cd modelviz
pip install -e ".[dev]"
```

### System Requirements

For 2D Graphviz diagrams, install the Graphviz system package:

```bash
# macOS
brew install graphviz

# Ubuntu/Debian
sudo apt-get install graphviz

# Windows (or use Conda)
conda install -c conda-forge graphviz
```

> **Note**: Three.js 3D visualizations work without any system dependencies.

## 🎯 Quick Start

### 2D Visualization (Graphviz)

```python
import torch.nn as nn
from modelviz import visualize

model = nn.Sequential(
    nn.Conv2d(1, 32, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 13 * 13, 10)
)

# Renders inline in Jupyter
visualize(model, input_shape=(1, 1, 28, 28))

# Save to file
visualize(model, input_shape=(1, 1, 28, 28), save_path="model.png")
```

### 3D Visualization (Three.js)

```python
from modelviz import visualize_threejs

# Creates an interactive HTML file
visualize_threejs(
    model,
    input_shape=(1, 1, 28, 28),
    save_path="model_3d.html"
)
# Open model_3d.html in your browser!
```

## 📖 Examples

### PyTorch CNN

```python
import torch.nn as nn
from modelviz import visualize, visualize_threejs

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))

model = CNN()

# 2D diagram with layer grouping
visualize(model, input_shape=(1, 3, 32, 32), title="CNN Architecture")

# 3D interactive visualization
visualize_threejs(model, input_shape=(1, 3, 32, 32), save_path="cnn_3d.html")
```

### TensorFlow/Keras

```python
import tensorflow as tf
from modelviz import visualize

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax'),
])

# No input_shape needed - Keras models are already built
visualize(model, save_path="keras_model.svg")
```

## 🎮 3D Visualization

The Three.js renderer creates stunning interactive 3D diagrams:

```python
from modelviz import visualize_threejs

html = visualize_threejs(
    model,
    input_shape=(1, 3, 224, 224),
    title="ResNet Block",
    show_shapes=True,      # Show tensor dimensions
    show_params=True,      # Show parameter counts
    group_blocks=True,     # Merge Conv+BN+ReLU
    save_path="resnet.html"
)
```

### Controls

| Action | Control |
|--------|---------|
| Rotate | Drag mouse |
| Zoom | Scroll wheel |
| Pan | Shift + Drag |
| Details | Hover over layer |

### Features

- **Horizontal layout** — Data flows left to right
- **Text labels** — Layer type and output shape above each node
- **Animated particles** — Shows data flow between layers
- **Hover tooltips** — Full layer information on mouseover
- **Legend** — Color and shape guide

## ⚙️ API Reference

### `visualize()`

Generate a 2D Graphviz diagram.

```python
visualize(
    model,                          # PyTorch or Keras model
    input_shape=(1, 3, 224, 224),  # Required for PyTorch
    framework="auto",               # "auto", "pytorch", "tensorflow"
    show_shapes=True,               # Show output tensor shapes
    show_params=True,               # Show parameter counts
    group_blocks=True,              # Merge Conv+ReLU patterns
    save_path="model.png",          # Optional: save to file
    title="My Model",               # Optional: diagram title
) -> graphviz.Digraph
```

### `visualize_threejs()`

Generate an interactive 3D Three.js visualization.

```python
visualize_threejs(
    model,                          # PyTorch or Keras model
    input_shape=(1, 3, 224, 224),  # Required for PyTorch
    framework="auto",               # "auto", "pytorch", "tensorflow"
    show_shapes=True,               # Show shapes in labels
    show_params=True,               # Show params in tooltips
    group_blocks=True,              # Merge Conv+ReLU patterns
    save_path="model.html",         # Save as HTML file
    title="My Model 3D",            # Visualization title
) -> str  # Returns HTML string
```

### `visualize_3d()`

Generate a Plotly 3D visualization (simpler fallback).

```python
visualize_3d(
    model,
    input_shape=(1, 3, 224, 224),
    layout="tower",                 # "tower", "spiral", "grid"
    save_path="model.png",
) -> plotly.graph_objects.Figure
```

## 🎨 Styling

### 2D Node Colors (Graphviz)

| Layer Type | Color | Hex |
|------------|-------|-----|
| Convolution | Indigo | `#6366f1` |
| Linear/Dense | Purple | `#8b5cf6` |
| Pooling | Cyan | `#06b6d4` |
| Activation | Amber | `#f59e0b` |
| Normalization | Emerald | `#10b981` |
| Flatten | Pink | `#ec4899` |
| Dropout | Red | `#ef4444` |
| Embedding | Lime | `#84cc16` |
| RNN/LSTM | Teal | `#14b8a6` |
| Attention | Orange | `#f97316` |

### Block Grouping

Common patterns are automatically merged:

- `Conv2d` → `BatchNorm2d` → `ReLU` → **Conv2d + BatchNorm2d + ReLU**
- `Conv2d` → `ReLU` → **Conv2d + ReLU**
- `Linear` → `ReLU` → **Linear + ReLU**
- `Dense` → `Activation` → **Dense + Activation**

Disable with `group_blocks=False`.

## 🏗️ Architecture

```
modelviz/
├── modelviz/
│   ├── __init__.py              # Public API
│   ├── visualize.py             # Main API functions
│   ├── graph/
│   │   ├── layer_node.py        # LayerNode dataclass
│   │   └── builder.py           # Graph construction
│   ├── parsers/
│   │   ├── torch_parser.py      # PyTorch model parsing
│   │   ├── tf_parser.py         # TensorFlow/Keras parsing
│   │   └── fx_tracer.py         # Skip connection detection (NEW)
│   ├── renderers/
│   │   ├── graphviz_renderer.py # 2D Graphviz output
│   │   ├── plotly_renderer.py   # 3D Plotly output
│   │   └── threejs_renderer.py  # 3D Three.js output
│   └── utils/
│       ├── framework_detect.py  # Auto-detection
│       └── grouping.py          # Layer pattern grouping
├── tests/                       # Test suite
├── examples/                    # Demo scripts
├── docs/                        # Documentation
└── pyproject.toml              # Package config
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=modelviz --cov-report=html

# Run specific test
pytest tests/test_grouping.py -v
```

## 🗺️ Roadmap

- [x] Branching graph support (ResNet, UNet skip connections)
- [ ] Transformer attention pattern visualization
- [ ] Interactive web dashboard
- [ ] Custom color themes
- [ ] Model comparison (side-by-side)
- [ ] FLOPs/MACs calculation
- [ ] ONNX model support

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Start

```bash
git clone https://github.com/shreyanshjain05/modelviz.git
cd modelviz
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,torch,tf]"
pytest tests/ -v
```

### Code Style

- Python 3.10+
- Type hints on all public functions
- Google-style docstrings
- Black + isort formatting

## 📄 License

Apache 2.0 License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Graphviz](https://graphviz.org/) — 2D graph rendering
- [Three.js](https://threejs.org/) — 3D WebGL visualization
- [Plotly](https://plotly.com/) — Interactive 3D charts

---

<p align="center">
  Made with ❤️ for the deep learning community
  <br><br>
  ⭐ Star this repo if you find it useful!
  <br><br>
  ☕ You can also support me on Ko-fi: https://ko-fi.com/shreyanshjain05 — every coffee keeps me going!
</p>


