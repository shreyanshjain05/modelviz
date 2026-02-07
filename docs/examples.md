# Examples

Complete code examples for common modelviz use cases.

## PyTorch Examples

### Simple MLP

```python
import torch.nn as nn
from modelviz import visualize, visualize_threejs

model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# 2D visualization
visualize(model, input_shape=(1, 784), save_path="mlp.png")

# 3D visualization
visualize_threejs(model, input_shape=(1, 784), save_path="mlp_3d.html")
```

### Convolutional Neural Network

```python
import torch.nn as nn
from modelviz import visualize_threejs

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CNN(num_classes=10)
visualize_threejs(
    model,
    input_shape=(1, 3, 32, 32),
    title="CIFAR-10 CNN",
    save_path="cnn_3d.html"
)
```

### ResNet-style Block

```python
import torch.nn as nn
from modelviz import visualize

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

model = ResBlock(64)
visualize(model, input_shape=(1, 64, 32, 32), title="ResNet Block")
```

### RNN / LSTM

```python
import torch.nn as nn
from modelviz import visualize_threejs

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

model = LSTMClassifier(vocab_size=10000, embed_dim=128, hidden_dim=256, num_classes=5)
visualize_threejs(model, input_shape=(1, 100), title="LSTM Classifier")
```

---

## TensorFlow/Keras Examples

### Keras Sequential

```python
import tensorflow as tf
from modelviz import visualize

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

visualize(model, save_path="keras_cnn.svg")
```

### Keras Functional API

```python
import tensorflow as tf
from modelviz import visualize_threejs

inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(1000, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
visualize_threejs(model, title="ImageNet Classifier", save_path="imagenet_3d.html")
```

---

## Saving Options

### PNG (raster)
```python
visualize(model, input_shape, save_path="model.png")
```

### SVG (vector)
```python
visualize(model, input_shape, save_path="model.svg")
```

### PDF
```python
visualize(model, input_shape, save_path="model.pdf")
```

### Interactive HTML (Three.js)
```python
visualize_threejs(model, input_shape, save_path="model.html")
```

---

## Customization

### Disable grouping

```python
visualize(model, input_shape, group_blocks=False)
```

### Hide shapes

```python
visualize(model, input_shape, show_shapes=False)
```

### Hide parameters

```python
visualize(model, input_shape, show_params=False)
```

### Add title

```python
visualize(model, input_shape, title="My Amazing Model")
```
