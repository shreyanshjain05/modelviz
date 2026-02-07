#!/usr/bin/env python3
"""
Demo: Skip Connections / ResNet Visualization

Shows how modelviz handles residual connections in ResNet-style blocks.
"""

import torch
import torch.nn as nn
from modelviz.parsers.fx_tracer import trace_pytorch_graph, has_skip_connections
from modelviz.renderers.graphviz_renderer import render_graph
import os


class ResidualBlock(nn.Module):
    """A basic ResNet residual block with skip connection."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        identity = x  # Skip connection
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # Residual addition
        out = self.relu(out)
        
        return out


class SimpleResNet(nn.Module):
    """A simple ResNet-style network with skip connections."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.block1 = ResidualBlock(64)
        self.block2 = ResidualBlock(64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def main():
    print("=" * 60)
    print("🔄 Skip Connections Demo - ResNet Style")
    print("=" * 60)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Test single residual block
    print("\n📦 Testing ResidualBlock...")
    block = ResidualBlock(64)
    input_shape = (1, 64, 32, 32)
    
    if has_skip_connections(block):
        print("✅ Skip connections detected!")
    else:
        print("⚠️ No skip connections detected")
    
    try:
        nodes, edges = trace_pytorch_graph(block, input_shape)
        print(f"   Nodes: {len(nodes)}")
        print(f"   Edges: {len(edges)}")
        
        # Count edge types
        edge_types = {}
        for edge in edges:
            et = edge.edge_type
            edge_types[et] = edge_types.get(et, 0) + 1
        print(f"   Edge types: {edge_types}")
        
        # Render
        graph = render_graph(nodes, edges, title="ResNet Block")
        output_path = os.path.join(output_dir, "resnet_block")
        graph.render(output_path, format="svg", cleanup=True)
        print(f"   Saved: {output_path}.svg")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test full SimpleResNet
    print("\n📦 Testing SimpleResNet...")
    model = SimpleResNet(num_classes=10)
    input_shape = (1, 3, 224, 224)
    
    if has_skip_connections(model):
        print("✅ Skip connections detected!")
    
    try:
        nodes, edges = trace_pytorch_graph(model, input_shape)
        print(f"   Nodes: {len(nodes)}")
        print(f"   Edges: {len(edges)}")
        
        edge_types = {}
        for edge in edges:
            et = edge.edge_type
            edge_types[et] = edge_types.get(et, 0) + 1
        print(f"   Edge types: {edge_types}")
        
        # Render
        graph = render_graph(nodes, edges, title="Simple ResNet")
        output_path = os.path.join(output_dir, "simple_resnet")
        graph.render(output_path, format="svg", cleanup=True)
        print(f"   Saved: {output_path}.svg")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
