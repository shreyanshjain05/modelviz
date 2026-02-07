"""
Three.js-based 3D rendering engine for modelviz.

Generates interactive HTML files with Three.js visualizations where each
layer type has a distinct, meaningful 3D representation.

Layer Type → 3D Shape Mapping:
- Conv layers → 3D rectangular prisms (width=channels, depth=kernel)
- Linear/Dense → Flat planes/rectangles
- Pooling → Smaller cubes with wireframe
- Activation → Glowing spheres
- BatchNorm → Thin transparent slabs
- Flatten → Cones pointing down
- Dropout → Dotted/transparent cubes
- Embedding → 3D stacked boxes
- LSTM/GRU → Cylinders
- Attention → Octahedrons
"""

import json
import math
import os
from typing import Literal, Optional, Sequence

from modelviz.graph.builder import Edge
from modelviz.graph.layer_node import LayerNode

# Color scheme for different layer types
LAYER_COLORS = {
    "conv": "#6366f1",  # Indigo
    "linear": "#8b5cf6",  # Purple
    "pool": "#06b6d4",  # Cyan
    "norm": "#10b981",  # Emerald
    "activation": "#f59e0b",  # Amber
    "dropout": "#ef4444",  # Red
    "flatten": "#ec4899",  # Pink
    "embed": "#84cc16",  # Lime
    "recurrent": "#14b8a6",  # Teal
    "attention": "#f97316",  # Orange
    "input": "#3b82f6",  # Blue
    "output": "#22c55e",  # Green
    "default": "#64748b",  # Slate
}


def render_threejs(
    nodes: Sequence[LayerNode],
    edges: Sequence[Edge],
    show_shapes: bool = True,
    show_params: bool = True,
    title: Optional[str] = None,
) -> str:
    """
    Render a neural network graph as an interactive Three.js 3D visualization.

    Each layer type gets a distinct 3D representation with labels:
    - Conv: 3D box proportional to channels
    - Linear: Flat rectangle
    - Pooling: Smaller cube with edges
    - Activation: Glowing sphere
    - BatchNorm: Thin slab
    - Flatten: Cone
    - Dropout: Transparent cube
    - Embedding: Stacked layers
    - RNN/LSTM: Cylinder
    - Attention: Octahedron

    Args:
        nodes: Sequence of LayerNode objects to render.
        edges: Sequence of Edge objects connecting nodes.
        show_shapes: Whether to show output shapes in tooltips.
        show_params: Whether to show parameter counts in tooltips.
        title: Optional title for the visualization.

    Returns:
        Complete HTML string with embedded Three.js visualization.
    """
    # Convert nodes and edges to JSON for JavaScript
    nodes_data = []
    for node in nodes:
        layer_type = _classify_layer_type(node.type)
        color = _get_layer_color(node.type)
        geometry = _get_layer_geometry(node)

        nodes_data.append(
            {
                "id": node.id,
                "name": node.name,
                "type": node.type,
                "displayType": node.display_type,
                "layerClass": layer_type,
                "color": color,
                "geometry": geometry,
                "inputShape": list(node.input_shape) if node.input_shape else None,
                "outputShape": list(node.output_shape) if node.output_shape else None,
                "params": node.params,
                "formattedParams": node.formatted_params,
                "isGrouped": node.is_grouped,
            }
        )

    edges_data = [
        {
            "source": edge.source_id,
            "target": edge.target_id,
            "edgeType": getattr(edge, "edge_type", "sequential"),
        }
        for edge in edges
    ]

    # Generate HTML with embedded Three.js
    html = _generate_html(
        nodes_json=json.dumps(nodes_data, indent=2),
        edges_json=json.dumps(edges_data, indent=2),
        title=title or "Neural Network Architecture - 3D View",
        show_shapes=show_shapes,
        show_params=show_params,
    )

    return html


def render_threejs_to_file(
    nodes: Sequence[LayerNode],
    edges: Sequence[Edge],
    filepath: str,
    show_shapes: bool = True,
    show_params: bool = True,
    title: Optional[str] = None,
) -> str:
    """
    Render the 3D graph and save to an HTML file.

    Args:
        nodes: Sequence of LayerNode objects.
        edges: Sequence of Edge objects.
        filepath: Output file path (should end with .html).
        show_shapes: Whether to show output shapes.
        show_params: Whether to show parameter counts.
        title: Optional title for the visualization.

    Returns:
        Path to the rendered file.
    """
    html = render_threejs(nodes, edges, show_shapes, show_params, title)

    if not filepath.endswith(".html"):
        filepath = f"{filepath}.html"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    return filepath


def _classify_layer_type(type_str: str) -> str:
    """Classify layer type for geometry selection."""
    type_lower = type_str.lower()

    if "conv" in type_lower:
        return "conv"
    elif "linear" in type_lower or "dense" in type_lower:
        return "linear"
    elif "pool" in type_lower:
        return "pool"
    elif "flatten" in type_lower:
        return "flatten"
    elif any(
        act in type_lower
        for act in [
            "relu",
            "sigmoid",
            "tanh",
            "gelu",
            "softmax",
            "activation",
            "silu",
            "mish",
        ]
    ):
        return "activation"
    elif "norm" in type_lower or "bn" in type_lower or "layernorm" in type_lower:
        return "norm"
    elif "dropout" in type_lower:
        return "dropout"
    elif "embed" in type_lower:
        return "embed"
    elif any(rnn in type_lower for rnn in ["lstm", "gru", "rnn"]):
        return "recurrent"
    elif "attention" in type_lower or "transformer" in type_lower:
        return "attention"
    elif "input" in type_lower:
        return "input"
    elif "output" in type_lower or "softmax" in type_lower:
        return "output"

    return "default"


def _get_layer_color(type_str: str) -> str:
    """Get color for a layer type."""
    layer_class = _classify_layer_type(type_str)
    return LAYER_COLORS.get(layer_class, LAYER_COLORS["default"])


def _get_layer_geometry(node: LayerNode) -> dict:
    """
    Calculate geometry parameters for a layer node.

    Returns dimensions that will be used by Three.js to create
    appropriate 3D shapes.
    """
    layer_class = _classify_layer_type(node.type)

    # Base dimensions
    base_width = 2.0
    base_height = 1.0
    base_depth = 2.0

    # Scale based on output shape if available
    if node.output_shape and len(node.output_shape) >= 2:
        # For conv layers: use channel count for depth
        if layer_class == "conv" and len(node.output_shape) >= 2:
            channels = node.output_shape[1] if len(node.output_shape) > 1 else 32
            base_depth = min(4, max(0.5, channels / 32))
            if len(node.output_shape) >= 3:
                spatial = node.output_shape[2] if len(node.output_shape) > 2 else 28
                base_width = min(3, max(1, spatial / 28))
                base_height = base_width

        # For linear layers: use output features
        elif layer_class == "linear":
            features = node.output_shape[-1] if node.output_shape else 128
            base_width = min(4, max(1.5, features / 64))
            base_depth = 0.3  # Thin plane

    # Scale based on parameter count
    if node.params > 0:
        param_scale = math.log10(node.params + 1) / 6
        base_width *= 1 + param_scale * 0.5
        base_height *= 1 + param_scale * 0.3

    return {
        "width": round(base_width, 2),
        "height": round(base_height, 2),
        "depth": round(base_depth, 2),
    }


def _generate_html(
    nodes_json: str,
    edges_json: str,
    title: str,
    show_shapes: bool,
    show_params: bool,
) -> str:
    """Generate the complete HTML file with Three.js visualization."""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
            overflow: hidden;
        }}
        #container {{
            width: 100vw;
            height: 100vh;
            position: relative;
        }}
        #title {{
            position: absolute;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            color: #f8fafc;
            font-size: 24px;
            font-weight: 600;
            text-shadow: 0 2px 10px rgba(0,0,0,0.5);
            z-index: 100;
            letter-spacing: 0.5px;
        }}
        #tooltip {{
            position: absolute;
            background: rgba(15, 23, 42, 0.95);
            color: #f8fafc;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 13px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            max-width: 280px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            border: 1px solid rgba(255,255,255,0.1);
            z-index: 1000;
        }}
        #tooltip.visible {{
            opacity: 1;
        }}
        #tooltip .layer-type {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 8px;
            color: #60a5fa;
        }}
        #tooltip .layer-name {{
            color: #94a3b8;
            font-size: 11px;
            margin-bottom: 8px;
        }}
        #tooltip .info-row {{
            display: flex;
            justify-content: space-between;
            margin: 4px 0;
            padding: 4px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        #tooltip .info-label {{
            color: #94a3b8;
        }}
        #tooltip .info-value {{
            color: #f8fafc;
            font-weight: 500;
        }}
        #legend {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(15, 23, 42, 0.9);
            padding: 16px;
            border-radius: 8px;
            color: #f8fafc;
            font-size: 12px;
            z-index: 100;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        #legend h3 {{
            font-size: 14px;
            margin-bottom: 12px;
            color: #94a3b8;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 6px 0;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
            margin-right: 10px;
        }}
        #controls {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(15, 23, 42, 0.9);
            padding: 12px 16px;
            border-radius: 8px;
            color: #94a3b8;
            font-size: 11px;
            z-index: 100;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        #controls div {{
            margin: 4px 0;
        }}
        #loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #60a5fa;
            font-size: 18px;
            z-index: 200;
        }}
        /* CSS for 3D labels */
        .layer-label {{
            color: #f8fafc;
            font-family: 'Segoe UI', system-ui, sans-serif;
            font-size: 12px;
            font-weight: 600;
            padding: 4px 10px;
            background: rgba(15, 23, 42, 0.85);
            border-radius: 6px;
            white-space: nowrap;
            border: 1px solid rgba(255,255,255,0.15);
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}
        .layer-label .shape-info {{
            font-size: 10px;
            font-weight: 400;
            color: #94a3b8;
            display: block;
            margin-top: 2px;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="title">{title}</div>
        <div id="tooltip"></div>
        <div id="loading">Loading 3D Scene...</div>
        <div id="legend">
            <h3>Layer Types</h3>
            <div class="legend-item">
                <div class="legend-color" style="background: #6366f1;"></div>
                <span>Convolution (Box)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #8b5cf6;"></div>
                <span>Linear/Dense (Plane)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #f59e0b;"></div>
                <span>Activation (Sphere)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #06b6d4;"></div>
                <span>Pooling (Small Box)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #10b981;"></div>
                <span>BatchNorm (Slab)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #ec4899;"></div>
                <span>Flatten (Cone)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #ef4444;"></div>
                <span>Dropout (Wireframe)</span>
            </div>
        </div>
        <div id="controls">
            <div>🖱️ Drag to rotate</div>
            <div>📜 Scroll to zoom</div>
            <div>⌨️ Shift+Drag to pan</div>
        </div>
    </div>

    <script type="importmap">
    {{
        "imports": {{
            "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
            "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
        }}
    }}
    </script>

    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
        import {{ CSS2DRenderer, CSS2DObject }} from 'three/addons/renderers/CSS2DRenderer.js';

        // Data from Python
        const nodesData = {nodes_json};
        const edgesData = {edges_json};
        const showShapes = {str(show_shapes).lower()};
        const showParams = {str(show_params).lower()};

        // Scene setup
        const container = document.getElementById('container');
        const scene = new THREE.Scene();
        
        // Camera - positioned for horizontal view
        const camera = new THREE.PerspectiveCamera(
            50,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        // Position camera to look at horizontal layout from above-front
        camera.position.set(0, 15, 25);

        // WebGL Renderer
        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setClearColor(0x0f172a, 1);
        container.appendChild(renderer.domElement);

        // CSS2D Renderer for labels
        const labelRenderer = new CSS2DRenderer();
        labelRenderer.setSize(window.innerWidth, window.innerHeight);
        labelRenderer.domElement.style.position = 'absolute';
        labelRenderer.domElement.style.top = '0px';
        labelRenderer.domElement.style.pointerEvents = 'none';
        container.appendChild(labelRenderer.domElement);

        // Controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.minDistance = 8;
        controls.maxDistance = 80;
        controls.target.set(0, 0, 0);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 20, 10);
        scene.add(directionalLight);

        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight2.position.set(-10, 10, -10);
        scene.add(directionalLight2);

        const pointLight = new THREE.PointLight(0x60a5fa, 0.5, 100);
        pointLight.position.set(0, 15, 0);
        scene.add(pointLight);

        // Store meshes for raycasting
        const meshes = [];
        const meshToNode = new Map();

        // Create materials for each layer type
        function createMaterial(color, type) {{
            const hexColor = parseInt(color.replace('#', '0x'));
            
            if (type === 'activation') {{
                return new THREE.MeshPhongMaterial({{
                    color: hexColor,
                    emissive: hexColor,
                    emissiveIntensity: 0.4,
                    shininess: 100,
                    transparent: true,
                    opacity: 0.9,
                }});
            }} else if (type === 'dropout') {{
                return new THREE.MeshPhongMaterial({{
                    color: hexColor,
                    transparent: true,
                    opacity: 0.4,
                    wireframe: false,
                }});
            }} else if (type === 'norm') {{
                return new THREE.MeshPhongMaterial({{
                    color: hexColor,
                    transparent: true,
                    opacity: 0.7,
                    shininess: 50,
                }});
            }} else {{
                return new THREE.MeshPhongMaterial({{
                    color: hexColor,
                    shininess: 80,
                    transparent: true,
                    opacity: 0.95,
                }});
            }}
        }}

        // Create a text label for a layer
        function createLabel(node) {{
            const div = document.createElement('div');
            div.className = 'layer-label';
            
            let labelText = node.displayType;
            
            // Add shape info if available
            if (showShapes && node.outputShape) {{
                const shapeStr = node.outputShape.slice(-2).join('×');
                labelText += `<span class="shape-info">${{shapeStr}}</span>`;
            }}
            
            div.innerHTML = labelText;
            
            const label = new CSS2DObject(div);
            return label;
        }}

        // Create geometry based on layer type
        function createLayerMesh(node, xPosition) {{
            const {{ geometry, color, layerClass }} = node;
            let mesh;
            let geo;
            const scale = 1.2;  // Overall scale factor

            switch (layerClass) {{
                case 'conv':
                    // 3D Box - represents feature maps
                    geo = new THREE.BoxGeometry(
                        geometry.depth * scale * 1.5,  // Depth as width for horizontal view
                        geometry.height * scale,
                        geometry.width * scale * 1.2
                    );
                    break;

                case 'linear':
                    // Flat plane - represents dense connections
                    geo = new THREE.BoxGeometry(
                        geometry.depth * scale * 0.8,
                        0.2,
                        geometry.width * scale * 1.5
                    );
                    break;

                case 'pool':
                    // Smaller box with beveled edges
                    geo = new THREE.BoxGeometry(
                        geometry.depth * scale * 0.8,
                        geometry.height * scale * 0.6,
                        geometry.width * scale * 0.8
                    );
                    break;

                case 'activation':
                    // Glowing sphere
                    geo = new THREE.SphereGeometry(0.7 * scale, 32, 32);
                    break;

                case 'norm':
                    // Thin slab
                    geo = new THREE.BoxGeometry(
                        geometry.depth * scale * 1.2,
                        0.15,
                        geometry.width * scale * 1.2
                    );
                    break;

                case 'flatten':
                    // Cone - funneling data (rotated for horizontal)
                    geo = new THREE.ConeGeometry(0.9 * scale, 1.4 * scale, 32);
                    break;

                case 'dropout':
                    // Wireframe cube
                    geo = new THREE.BoxGeometry(
                        geometry.depth * scale * 0.7,
                        geometry.height * scale * 0.7,
                        geometry.width * scale * 0.7
                    );
                    break;

                case 'embed':
                    // Stacked boxes effect
                    geo = new THREE.BoxGeometry(
                        geometry.depth * scale * 0.5,
                        geometry.height * scale * 1.5,
                        geometry.width * scale
                    );
                    break;

                case 'recurrent':
                    // Cylinder for RNN/LSTM/GRU (rotated for horizontal)
                    geo = new THREE.CylinderGeometry(0.8 * scale, 0.8 * scale, 1.6 * scale, 32);
                    break;

                case 'attention':
                    // Octahedron for attention
                    geo = new THREE.OctahedronGeometry(1.0 * scale);
                    break;

                default:
                    // Default box
                    geo = new THREE.BoxGeometry(
                        geometry.depth * scale,
                        geometry.height * scale,
                        geometry.width * scale
                    );
            }}

            const material = createMaterial(color, layerClass);
            mesh = new THREE.Mesh(geo, material);
            
            // HORIZONTAL LAYOUT: Position along X-axis
            mesh.position.set(xPosition, 0, 0);

            // Rotate cone and cylinder for horizontal orientation
            if (layerClass === 'flatten') {{
                mesh.rotation.z = -Math.PI / 2;  // Point cone to the right
            }}
            if (layerClass === 'recurrent') {{
                mesh.rotation.z = Math.PI / 2;  // Lay cylinder horizontal
            }}

            // Add edges for certain types
            if (['conv', 'pool', 'linear', 'default'].includes(layerClass)) {{
                const edges = new THREE.EdgesGeometry(geo);
                const lineMaterial = new THREE.LineBasicMaterial({{ 
                    color: 0xffffff, 
                    opacity: 0.3,
                    transparent: true 
                }});
                const wireframe = new THREE.LineSegments(edges, lineMaterial);
                mesh.add(wireframe);
            }}

            // Add wireframe overlay for dropout
            if (layerClass === 'dropout') {{
                const wireGeo = new THREE.WireframeGeometry(geo);
                const wireMat = new THREE.LineBasicMaterial({{ color: 0xef4444 }});
                const wireframe = new THREE.LineSegments(wireGeo, wireMat);
                mesh.add(wireframe);
            }}

            // Add label above the mesh
            const label = createLabel(node);
            label.position.set(0, 2.5, 0);  // Above the mesh
            mesh.add(label);

            return mesh;
        }}

        // Create connection lines between layers (horizontal)
        function createConnection(startX, endX) {{
            const points = [];
            points.push(new THREE.Vector3(startX, 0, 0));
            points.push(new THREE.Vector3(endX, 0, 0));

            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const material = new THREE.LineBasicMaterial({{ 
                color: 0x60a5fa,
                opacity: 0.7,
                transparent: true,
            }});

            return new THREE.Line(geometry, material);
        }}

        // Create flowing particles along connections (horizontal)
        function createFlowParticles(startX, endX) {{
            const particleCount = 4;
            const particles = new THREE.Group();
            
            for (let i = 0; i < particleCount; i++) {{
                const geometry = new THREE.SphereGeometry(0.1, 8, 8);
                const material = new THREE.MeshBasicMaterial({{
                    color: 0x60a5fa,
                    transparent: true,
                    opacity: 0.9
                }});
                const particle = new THREE.Mesh(geometry, material);
                
                // Store animation data (horizontal)
                particle.userData.startX = startX;
                particle.userData.endX = endX;
                particle.userData.offset = i / particleCount;
                particle.userData.speed = 0.4 + Math.random() * 0.2;
                
                particles.add(particle);
            }}
            
            return particles;
        }}

        // Build the scene - HORIZONTAL LAYOUT
        const layerSpacing = 5.0;  // Spacing between layers
        const totalWidth = (nodesData.length - 1) * layerSpacing;
        let xPos = -totalWidth / 2;  // Start from left, center the whole thing
        const allParticles = [];

        nodesData.forEach((node, index) => {{
            const mesh = createLayerMesh(node, xPos);
            scene.add(mesh);
            meshes.push(mesh);
            meshToNode.set(mesh, node);

            // Add connection to next layer
            if (index < nodesData.length - 1) {{
                const nextXPos = xPos + layerSpacing;
                const connection = createConnection(xPos + 1.5, nextXPos - 1.5);
                scene.add(connection);
                
                // Add flow particles
                const particles = createFlowParticles(xPos + 1.5, nextXPos - 1.5);
                scene.add(particles);
                allParticles.push(particles);
            }}

            xPos += layerSpacing;
        }});

        // Add a ground grid (horizontal)
        const gridHelper = new THREE.GridHelper(totalWidth + 20, 40, 0x334155, 0x1e293b);
        gridHelper.position.y = -3;
        scene.add(gridHelper);

        // Center camera target on the model
        controls.target.set(0, 0, 0);
        controls.update();

        // Tooltip handling
        const tooltip = document.getElementById('tooltip');
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();

        function updateTooltip(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(meshes);

            if (intersects.length > 0) {{
                const mesh = intersects[0].object;
                const node = meshToNode.get(mesh);

                if (node) {{
                    let html = `<div class="layer-type">${{node.displayType}}</div>`;
                    html += `<div class="layer-name">${{node.name}}</div>`;
                    
                    if (showShapes && node.outputShape) {{
                        html += `<div class="info-row">
                            <span class="info-label">Output Shape</span>
                            <span class="info-value">${{JSON.stringify(node.outputShape)}}</span>
                        </div>`;
                    }}
                    
                    if (showParams) {{
                        html += `<div class="info-row">
                            <span class="info-label">Parameters</span>
                            <span class="info-value">${{node.formattedParams}}</span>
                        </div>`;
                    }}
                    
                    if (node.isGrouped) {{
                        html += `<div class="info-row">
                            <span class="info-label">Grouped</span>
                            <span class="info-value">Yes</span>
                        </div>`;
                    }}

                    tooltip.innerHTML = html;
                    tooltip.style.left = event.clientX + 15 + 'px';
                    tooltip.style.top = event.clientY + 15 + 'px';
                    tooltip.classList.add('visible');

                    // Highlight the mesh
                    mesh.material.emissive = new THREE.Color(0x60a5fa);
                    mesh.material.emissiveIntensity = 0.4;
                }}
            }} else {{
                tooltip.classList.remove('visible');
                
                // Reset all mesh highlights
                meshes.forEach(m => {{
                    if (m.material.emissive) {{
                        m.material.emissive = new THREE.Color(
                            meshToNode.get(m)?.layerClass === 'activation' 
                                ? meshToNode.get(m)?.color 
                                : 0x000000
                        );
                        m.material.emissiveIntensity = 
                            meshToNode.get(m)?.layerClass === 'activation' ? 0.4 : 0;
                    }}
                }});
            }}
        }}

        container.addEventListener('mousemove', updateTooltip);

        // Animation loop
        let time = 0;
        function animate() {{
            requestAnimationFrame(animate);
            time += 0.016;
            
            // Animate particles (horizontal movement)
            allParticles.forEach(group => {{
                group.children.forEach(particle => {{
                    const t = ((time * particle.userData.speed + particle.userData.offset) % 1);
                    particle.position.x = particle.userData.startX + 
                        (particle.userData.endX - particle.userData.startX) * t;
                    particle.position.y = Math.sin(t * Math.PI) * 0.3;  // Slight arc
                    particle.material.opacity = Math.sin(t * Math.PI) * 0.9;
                }});
            }});

            // Subtle animation for special layers
            meshes.forEach((mesh, index) => {{
                const node = meshToNode.get(mesh);
                if (node && node.layerClass === 'activation') {{
                    mesh.rotation.y = time * 0.5;
                }}
                if (node && node.layerClass === 'attention') {{
                    mesh.rotation.y = time * 0.3;
                    mesh.rotation.x = Math.sin(time * 0.5) * 0.1;
                }}
            }});

            controls.update();
            renderer.render(scene, camera);
            labelRenderer.render(scene, camera);
        }}

        // Handle resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
            labelRenderer.setSize(window.innerWidth, window.innerHeight);
        }});

        // Hide loading and start
        document.getElementById('loading').style.display = 'none';
        animate();
    </script>
</body>
</html>"""
