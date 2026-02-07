#!/usr/bin/env python3
"""
Three.js 3D Visualization Demo

Generates a stunning interactive 3D visualization of a CNN using Three.js.
Each layer type has a distinct, meaningful 3D shape.
"""

import torch.nn as nn
from modelviz import visualize_threejs
import os


def create_demo_cnn():
    """Create a comprehensive CNN model to showcase all layer types."""
    return nn.Sequential(
        # Conv block 1
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        # Conv block 2
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        # Conv block 3
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        # Classification head
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 10),
    )


def main():
    print("=" * 60)
    print("🎮 Three.js 3D Visualization Demo")
    print("=" * 60)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model
    print("\n📦 Creating CNN model with multiple layer types...")
    model = create_demo_cnn()
    input_shape = (1, 3, 32, 32)  # CIFAR-10 style input
    
    # Generate Three.js visualization
    print("🎮 Generating Three.js 3D visualization...")
    output_path = os.path.join(output_dir, "cnn_threejs_3d.html")
    
    html = visualize_threejs(
        model,
        input_shape=input_shape,
        title="CNN Architecture - Interactive 3D",
        group_blocks=True,
        save_path=output_path,
    )
    
    print(f"\n✅ Saved: {output_path}")
    print(f"   Size: {len(html):,} bytes")
    
    # Also create ungrouped version
    print("\n🎮 Generating ungrouped version...")
    output_path_ungrouped = os.path.join(output_dir, "cnn_threejs_3d_all_layers.html")
    
    visualize_threejs(
        model,
        input_shape=input_shape,
        title="CNN Architecture - All Layers",
        group_blocks=False,
        save_path=output_path_ungrouped,
    )
    
    print(f"✅ Saved: {output_path_ungrouped}")
    
    print("\n" + "=" * 60)
    print("🚀 Open the HTML files in your browser for interactive 3D!")
    print("=" * 60)
    print("\nLayer Types → 3D Shapes:")
    print("  📦 Conv2d      → 3D Box (depth = channels)")
    print("  📋 Linear      → Flat Plane")
    print("  🔮 ReLU        → Glowing Sphere")
    print("  📐 MaxPool2d   → Small Cube")
    print("  📊 BatchNorm   → Thin Slab")
    print("  🔻 Flatten     → Cone")
    print("  🎲 Dropout     → Wireframe Cube")
    print("\n💡 Controls:")
    print("  🖱️  Drag to rotate")
    print("  📜 Scroll to zoom")
    print("  ⌨️  Shift+Drag to pan")
    print("  👆 Hover for layer details")


if __name__ == "__main__":
    main()
