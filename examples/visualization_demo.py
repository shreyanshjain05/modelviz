#!/usr/bin/env python3
"""
Visualization Demo Script

This script generates sample visualizations for both 2D (Graphviz) and 3D (Plotly)
renderers and saves them as images for review.
"""

import torch.nn as nn
from modelviz import visualize, visualize_3d


def create_cnn_model():
    """Create a sample CNN model."""
    return nn.Sequential(
        # First conv block
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        # Second conv block
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        # Classification head
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 10),
    )


def create_mlp_model():
    """Create a sample MLP model."""
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 10),
    )


def main():
    """Generate and save all visualizations."""
    import os
    
    # Create output directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(output_dir, "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🎨 modelviz Visualization Demo")
    print("=" * 60)
    
    # =========================================================
    # CNN Model Visualizations
    # =========================================================
    print("\n📦 Creating CNN model...")
    cnn = create_cnn_model()
    input_shape_cnn = (1, 1, 28, 28)  # MNIST-like
    
    # 2D Visualization (Graphviz)
    print("📊 Generating 2D visualization (with grouping)...")
    viz_2d = visualize(
        cnn,
        input_shape=input_shape_cnn,
        title="CNN Architecture (Grouped)",
        group_blocks=True,
    )
    # Save the DOT source (human-readable)
    dot_path = os.path.join(output_dir, "cnn_2d_grouped.dot")
    with open(dot_path, "w") as f:
        f.write(viz_2d.source)
    print(f"   ✅ Saved DOT source: {dot_path}")
    
    # 2D Visualization without grouping
    print("📊 Generating 2D visualization (without grouping)...")
    viz_2d_ungrouped = visualize(
        cnn,
        input_shape=input_shape_cnn,
        title="CNN Architecture (All Layers)",
        group_blocks=False,
    )
    dot_path_ungrouped = os.path.join(output_dir, "cnn_2d_ungrouped.dot")
    with open(dot_path_ungrouped, "w") as f:
        f.write(viz_2d_ungrouped.source)
    print(f"   ✅ Saved DOT source: {dot_path_ungrouped}")
    
    # 3D Visualization - Tower Layout
    print("🎮 Generating 3D visualization (tower layout)...")
    viz_3d_tower = visualize_3d(
        cnn,
        input_shape=input_shape_cnn,
        title="CNN Architecture - 3D Tower View",
        layout="tower",
        save_path=os.path.join(output_dir, "cnn_3d_tower.png"),
    )
    print(f"   ✅ Saved PNG: {os.path.join(output_dir, 'cnn_3d_tower.png')}")
    
    # 3D Visualization - Spiral Layout
    print("🎮 Generating 3D visualization (spiral layout)...")
    viz_3d_spiral = visualize_3d(
        cnn,
        input_shape=input_shape_cnn,
        title="CNN Architecture - 3D Spiral View",
        layout="spiral",
        save_path=os.path.join(output_dir, "cnn_3d_spiral.png"),
    )
    print(f"   ✅ Saved PNG: {os.path.join(output_dir, 'cnn_3d_spiral.png')}")
    
    # Save interactive HTML
    print("🌐 Generating interactive HTML...")
    html_path = os.path.join(output_dir, "cnn_3d_interactive.html")
    viz_3d_tower.write_html(html_path)
    print(f"   ✅ Saved HTML: {html_path}")
    
    # =========================================================
    # MLP Model Visualizations
    # =========================================================
    print("\n📦 Creating MLP model...")
    mlp = create_mlp_model()
    input_shape_mlp = (1, 784)  # Flattened MNIST
    
    # 3D Visualization - Grid Layout
    print("🎮 Generating 3D visualization (grid layout)...")
    viz_3d_mlp = visualize_3d(
        mlp,
        input_shape=input_shape_mlp,
        title="MLP Architecture - 3D View",
        layout="tower",
        save_path=os.path.join(output_dir, "mlp_3d.png"),
    )
    print(f"   ✅ Saved PNG: {os.path.join(output_dir, 'mlp_3d.png')}")
    
    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 60)
    print("✅ All visualizations generated successfully!")
    print("=" * 60)
    print(f"\n📁 Output directory: {output_dir}")
    print("\nGenerated files:")
    for filename in sorted(os.listdir(output_dir)):
        filepath = os.path.join(output_dir, filename)
        size = os.path.getsize(filepath)
        print(f"   • {filename} ({size:,} bytes)")
    
    print("\n💡 Tips:")
    print("   • Open .html files in a browser for interactive 3D view")
    print("   • Use Graphviz to render .dot files: dot -Tpng file.dot -o output.png")
    print("   • PNG files can be viewed directly")
    

if __name__ == "__main__":
    main()
