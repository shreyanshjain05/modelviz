"""
PyTorch CNN Demo

Demonstrates modelviz visualization with a simple CNN for MNIST-like images.
"""

import torch.nn as nn
from modelviz import visualize


def create_simple_cnn() -> nn.Module:
    """
    Create a simple CNN for 28x28 grayscale images.
    
    Architecture:
    - Conv2d(1, 32, 3) + ReLU + MaxPool(2)
    - Conv2d(32, 64, 3) + ReLU + MaxPool(2)
    - Flatten
    - Linear(64*5*5, 128) + ReLU
    - Linear(128, 10)
    """
    return nn.Sequential(
        # First conv block
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        # Second conv block
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        # Classification head
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 10),
    )


def main():
    """Run the demo."""
    print("Creating simple CNN model...")
    model = create_simple_cnn()
    
    print("\nModel architecture:")
    print(model)
    
    print("\nGenerating visualization...")
    
    # Visualize with grouping enabled (default)
    diagram = visualize(
        model,
        input_shape=(1, 1, 28, 28),  # batch_size=1, channels=1, height=28, width=28
        title="Simple CNN for MNIST",
    )
    
    # Save to file
    diagram.render("cnn_visualization", format="svg", cleanup=True)
    print("Saved visualization to: cnn_visualization.svg")
    
    # Also show without grouping
    print("\nGenerating visualization without grouping...")
    diagram_ungrouped = visualize(
        model,
        input_shape=(1, 1, 28, 28),
        group_blocks=False,
        title="Simple CNN (No Grouping)",
    )
    diagram_ungrouped.render("cnn_visualization_ungrouped", format="svg", cleanup=True)
    print("Saved visualization to: cnn_visualization_ungrouped.svg")


if __name__ == "__main__":
    main()
