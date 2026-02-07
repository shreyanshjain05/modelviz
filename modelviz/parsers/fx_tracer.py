"""
PyTorch FX-based graph tracer for modelviz.

Uses torch.fx symbolic tracing to capture the full computation graph,
including skip connections, residual connections, and branching paths.
"""

from collections import OrderedDict
from typing import Any, Optional

from modelviz.graph.builder import Edge
from modelviz.graph.layer_node import LayerNode


class TracingError(Exception):
    """Raised when FX tracing fails."""

    pass


def trace_pytorch_graph(
    model: Any,
    input_shape: tuple[int, ...],
) -> tuple[list[LayerNode], list[Edge]]:
    """
    Trace a PyTorch model's computation graph using torch.fx.

    This captures the full graph structure including:
    - Skip/residual connections (x + residual)
    - Concatenation operations (torch.cat)
    - Branching paths

    Args:
        model: A PyTorch nn.Module model.
        input_shape: Shape of the input tensor.

    Returns:
        Tuple of (nodes, edges) where edges include skip connections.

    Raises:
        TracingError: If FX tracing fails.
    """
    import torch
    import torch.fx as fx
    import torch.nn as nn

    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module, got {type(model)}")

    # Try FX tracing
    try:
        traced = fx.symbolic_trace(model)
    except Exception as e:
        raise TracingError(
            f"FX tracing failed: {e}. "
            f"Model may use dynamic control flow not supported by torch.fx."
        )

    # Maps from FX node name to our node info
    node_map: dict[str, dict[str, Any]] = {}
    fx_to_id: dict[str, int] = {}
    edges: list[Edge] = []
    node_id = 0

    # Get module parameter counts
    module_params: dict[str, int] = {}
    for name, module in model.named_modules():
        if name:
            params = sum(p.numel() for p in module.parameters(recurse=False))
            module_params[name] = params

    # Process FX graph nodes
    for fx_node in traced.graph.nodes:
        if fx_node.op == "placeholder":
            # Input node
            node_map[fx_node.name] = {
                "id": node_id,
                "name": "input",
                "type": "Input",
                "input_shape": input_shape,
                "output_shape": input_shape,
                "params": 0,
            }
            fx_to_id[fx_node.name] = node_id
            node_id += 1

        elif fx_node.op == "call_module":
            # Module call (Conv2d, Linear, etc.)
            module_name = fx_node.target
            module = traced.get_submodule(module_name)

            node_map[fx_node.name] = {
                "id": node_id,
                "name": module_name,
                "type": module.__class__.__name__,
                "input_shape": None,  # Will be filled during forward pass
                "output_shape": None,
                "params": module_params.get(module_name, 0),
            }
            fx_to_id[fx_node.name] = node_id

            # Add edges from input nodes
            for arg in fx_node.args:
                if hasattr(arg, "name") and arg.name in fx_to_id:
                    edges.append(
                        Edge(
                            source_id=fx_to_id[arg.name],
                            target_id=node_id,
                            edge_type="sequential",
                        )
                    )

            node_id += 1

        elif fx_node.op == "call_function":
            # Function call (add, cat, etc.)
            func_name = (
                fx_node.target.__name__
                if hasattr(fx_node.target, "__name__")
                else str(fx_node.target)
            )

            # Check for skip/residual connections
            if func_name in ("add", "add_"):
                # This is likely a residual connection
                node_map[fx_node.name] = {
                    "id": node_id,
                    "name": f"add_{node_id}",
                    "type": "Add (Residual)",
                    "input_shape": None,
                    "output_shape": None,
                    "params": 0,
                }
                fx_to_id[fx_node.name] = node_id

                # Add edges from both inputs - one sequential, one skip
                args_with_ids = [
                    arg
                    for arg in fx_node.args
                    if hasattr(arg, "name") and arg.name in fx_to_id
                ]
                for i, arg in enumerate(args_with_ids):
                    edge_type = "residual" if i > 0 else "sequential"
                    edges.append(
                        Edge(
                            source_id=fx_to_id[arg.name],
                            target_id=node_id,
                            edge_type=edge_type,
                        )
                    )

                node_id += 1

            elif func_name in ("cat", "concat"):
                # Concatenation
                node_map[fx_node.name] = {
                    "id": node_id,
                    "name": f"concat_{node_id}",
                    "type": "Concat",
                    "input_shape": None,
                    "output_shape": None,
                    "params": 0,
                }
                fx_to_id[fx_node.name] = node_id

                # Add edges from all concatenated inputs
                if fx_node.args and isinstance(fx_node.args[0], (list, tuple)):
                    for i, arg in enumerate(fx_node.args[0]):
                        if hasattr(arg, "name") and arg.name in fx_to_id:
                            edge_type = "concat" if i > 0 else "sequential"
                            edges.append(
                                Edge(
                                    source_id=fx_to_id[arg.name],
                                    target_id=node_id,
                                    edge_type=edge_type,
                                )
                            )

                node_id += 1

            else:
                # Other functions (relu, etc.) - treat as sequential
                # Skip common pass-through functions
                if func_name not in ("getattr", "getitem"):
                    node_map[fx_node.name] = {
                        "id": node_id,
                        "name": f"{func_name}_{node_id}",
                        "type": func_name.capitalize(),
                        "input_shape": None,
                        "output_shape": None,
                        "params": 0,
                    }
                    fx_to_id[fx_node.name] = node_id

                    for arg in fx_node.args:
                        if hasattr(arg, "name") and arg.name in fx_to_id:
                            edges.append(
                                Edge(
                                    source_id=fx_to_id[arg.name],
                                    target_id=node_id,
                                    edge_type="sequential",
                                )
                            )

                    node_id += 1

        elif fx_node.op == "output":
            # Output node - connect from last node
            for arg in fx_node.args:
                if hasattr(arg, "name") and arg.name in fx_to_id:
                    # Mark this as output
                    if arg.name in node_map:
                        node_map[arg.name]["is_output"] = True

    # Run forward pass to get shapes
    try:
        dummy_input = torch.zeros(input_shape)
        with torch.no_grad():
            model.eval()
            _ = model(dummy_input)
    except Exception:
        pass  # Shapes will be None if forward pass fails

    # Build LayerNode list
    nodes: list[LayerNode] = []
    for fx_name, info in node_map.items():
        node = LayerNode(
            id=info["id"],
            name=info["name"],
            type=info["type"],
            input_shape=info.get("input_shape"),
            output_shape=info.get("output_shape"),
            params=info["params"],
        )
        nodes.append(node)

    # Sort nodes by ID
    nodes.sort(key=lambda n: n.id)

    return nodes, edges


def has_skip_connections(model: Any) -> bool:
    """
    Check if a model has skip/residual connections.

    Args:
        model: A PyTorch nn.Module model.

    Returns:
        True if the model has skip connections.
    """
    import torch.fx as fx

    try:
        traced = fx.symbolic_trace(model)
        for node in traced.graph.nodes:
            if node.op == "call_function":
                func_name = getattr(node.target, "__name__", "")
                if func_name in ("add", "add_", "cat", "concat"):
                    return True
        return False
    except Exception:
        return False
