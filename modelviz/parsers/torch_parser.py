"""
PyTorch model parser for modelviz.

Extracts layer information from PyTorch nn.Module models using forward hooks.
"""

from collections import OrderedDict
from typing import Any, Optional

from modelviz.graph.layer_node import LayerNode


class DynamicControlFlowError(Exception):
    """Raised when dynamic control flow is detected in the model."""

    pass


class ForwardPassError(Exception):
    """Raised when the forward pass fails."""

    pass


def parse_pytorch_model(
    model: Any,
    input_shape: tuple[int, ...],
) -> list[LayerNode]:
    """
    Parse a PyTorch model and extract layer information.

    Uses forward hooks to capture the execution order and tensor shapes
    during a forward pass with dummy input.

    Args:
        model: A PyTorch nn.Module model.
        input_shape: Shape of the input tensor (including batch dimension).

    Returns:
        List of LayerNode objects representing the model's layers.

    Raises:
        ForwardPassError: If the forward pass fails.
        DynamicControlFlowError: If dynamic control flow is detected.

    Example:
        >>> import torch.nn as nn
        >>> model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        >>> nodes = parse_pytorch_model(model, (1, 10))
        >>> len(nodes)
        2
    """
    import torch
    import torch.nn as nn

    # Validate input
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module, got {type(model)}")

    # Storage for captured layer info
    layer_info: OrderedDict[str, dict[str, Any]] = OrderedDict()
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    execution_order: list[str] = []

    def create_hook(name: str):
        """Create a forward hook for a specific module."""

        def hook(
            module: nn.Module,
            input_tensors: tuple[torch.Tensor, ...],
            output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> None:
            # Track execution order
            if name not in execution_order:
                execution_order.append(name)

            # Get input shape
            input_shape_captured = None
            if input_tensors and len(input_tensors) > 0:
                if isinstance(input_tensors[0], torch.Tensor):
                    input_shape_captured = tuple(input_tensors[0].shape)

            # Get output shape
            output_shape = None
            if isinstance(output, torch.Tensor):
                output_shape = tuple(output.shape)
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                if isinstance(output[0], torch.Tensor):
                    output_shape = tuple(output[0].shape)

            # Count parameters
            params = sum(p.numel() for p in module.parameters(recurse=False))

            # Store layer info
            layer_info[name] = {
                "type": module.__class__.__name__,
                "input_shape": input_shape_captured,
                "output_shape": output_shape,
                "params": params,
            }

        return hook

    def is_leaf_module(module: nn.Module) -> bool:
        """
        Check if a module is a leaf (has no children with parameters).

        Skip containers like Sequential, ModuleList, ModuleDict.
        """
        container_types = (nn.Sequential, nn.ModuleList, nn.ModuleDict)
        if isinstance(module, container_types):
            return False

        # Check if it has children that are actual layers
        children = list(module.children())
        if not children:
            return True

        # If all children are containers, treat as leaf
        return all(isinstance(c, container_types) for c in children)

    try:
        # Set model to eval mode to disable dropout, etc.
        was_training = model.training
        model.eval()

        # Register hooks on leaf modules only
        for name, module in model.named_modules():
            if is_leaf_module(module) and name:  # Skip root module (empty name)
                hook = module.register_forward_hook(create_hook(name))
                hooks.append(hook)

        # Create dummy input
        try:
            import torch

            dummy_input = torch.zeros(input_shape)
        except Exception as e:
            raise ForwardPassError(f"Failed to create dummy input: {e}")

        # Run forward pass
        try:
            with torch.no_grad():
                model(dummy_input)
        except Exception as e:
            # Check for common dynamic control flow patterns
            error_str = str(e).lower()
            if "split" in error_str or "chunk" in error_str or "if" in error_str:
                raise DynamicControlFlowError(
                    f"Dynamic control flow detected in model: {e}. "
                    f"modelviz currently only supports static computation graphs."
                )
            raise ForwardPassError(f"Forward pass failed: {e}")

        # Build LayerNode list from captured info
        nodes: list[LayerNode] = []
        for idx, name in enumerate(execution_order):
            if name in layer_info:
                info = layer_info[name]
                node = LayerNode(
                    id=idx,
                    name=name,
                    type=info["type"],
                    input_shape=info["input_shape"],
                    output_shape=info["output_shape"],
                    params=info["params"],
                )
                nodes.append(node)

        return nodes

    finally:
        # Always remove hooks to avoid memory leaks
        for hook in hooks:
            hook.remove()

        # Restore training mode if it was on
        if was_training:
            model.train()


def get_pytorch_summary(model: Any, input_shape: tuple[int, ...]) -> dict[str, Any]:
    """
    Get a summary of the PyTorch model.

    Args:
        model: A PyTorch nn.Module model.
        input_shape: Shape of the input tensor.

    Returns:
        Dictionary with model summary information.
    """
    nodes = parse_pytorch_model(model, input_shape)
    total_params = sum(node.params for node in nodes)

    return {
        "num_layers": len(nodes),
        "total_params": total_params,
        "layers": [
            {
                "name": node.name,
                "type": node.type,
                "output_shape": node.output_shape,
                "params": node.params,
            }
            for node in nodes
        ],
    }
