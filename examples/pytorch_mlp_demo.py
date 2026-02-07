"""
PyTorch MLP Demo

Demonstrates modelviz visualization with a simple Multi-Layer Perceptron.
"""

import torch.nn as nn
from modelviz import visualize


def create_simple_mlp() -> nn.Module:
    """
    Create a simple MLP for tabular data.
    
    Architecture:
    - Linear(20, 128) + ReLU
    - Linear(128, 64) + ReLU
    - Linear(64, 32) + ReLU
    - Linear(32, 10)
    """
    return nn.Sequential(
        nn.Linear(20, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        
        nn.Linear(64, 32),
        nn.ReLU(),
        
        nn.Linear(32, 10),
    )


def create_mlp_with_batchnorm() -> nn.Module:
    """
    Create an MLP with batch normalization.
    
    Demonstrates grouping of Linear + BatchNorm patterns.
    """
    return nn.Sequential(
        nn.Linear(20, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        
        nn.Linear(64, 10),
    )


def main():
    """Run the demo."""
    print("Creating simple MLP model...")
    model = create_simple_mlp()
    
    print("\nModel architecture:")
    print(model)
    
    print("\nGenerating visualization...")
    
    # Visualize
    diagram = visualize(
        model,
        input_shape=(1, 20),  # batch_size=1, features=20
        title="Simple MLP",
    )
    
    # Save to file
    diagram.render("mlp_visualization", format="svg", cleanup=True)
    print("Saved visualization to: mlp_visualization.svg")
    
    # Also visualize MLP with BatchNorm
    print("\nCreating MLP with BatchNorm...")
    model_bn = create_mlp_with_batchnorm()
    
    diagram_bn = visualize(
        model_bn,
        input_shape=(1, 20),
        title="MLP with BatchNorm",
    )
    diagram_bn.render("mlp_batchnorm_visualization", format="svg", cleanup=True)
    print("Saved visualization to: mlp_batchnorm_visualization.svg")


if __name__ == "__main__":
    main()
