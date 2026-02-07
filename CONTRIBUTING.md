# Contributing to modelviz

Thank you for your interest in contributing to modelviz! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Style Guide](#style-guide)

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct. Please be respectful and inclusive in all interactions.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- A GitHub account

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/shreyanshjain05/modelviz.git
   cd modelviz
   ```

## Development Setup

### Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

### Install Development Dependencies

```bash
pip install -e ".[dev,torch,tf]"
```

### Install System Dependencies

For 2D Graphviz diagrams:

```bash
# macOS
brew install graphviz

# Ubuntu/Debian
sudo apt-get install graphviz
```

### Verify Installation

```bash
pytest tests/ -v
```

## Making Changes

### Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### Branch Naming Convention

- `feature/` — New features
- `fix/` — Bug fixes
- `docs/` — Documentation changes
- `refactor/` — Code refactoring
- `test/` — Test additions/changes

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=modelviz --cov-report=html
open htmlcov/index.html
```

### Run Specific Tests

```bash
# Run a specific test file
pytest tests/test_grouping.py -v

# Run a specific test
pytest tests/test_layer_node.py::TestLayerNode::test_create_basic_node -v
```

### Test Requirements

- All new code must have tests
- Maintain >80% code coverage
- All tests must pass before submitting PR

## Pull Request Process

### Before Submitting

1. ✅ Run all tests: `pytest tests/ -v`
2. ✅ Format code: `black modelviz tests`
3. ✅ Sort imports: `isort modelviz tests`
4. ✅ Check types: `mypy modelviz`
5. ✅ Update documentation if needed

### PR Title Format

```
type: short description

Examples:
feat: add ONNX model support
fix: correct layer grouping for Sequential models
docs: update API reference for visualize_threejs
```

### PR Description Template

```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] New feature
- [ ] Bug fix
- [ ] Documentation update
- [ ] Refactoring
- [ ] Test addition

## Testing
Describe how you tested your changes.

## Related Issues
Closes #123
```

## Style Guide

### Python Code Style

- **Formatter**: Black (line length 88)
- **Import sorting**: isort
- **Type hints**: Required for all public functions
- **Docstrings**: Google style

### Example Function

```python
def visualize(
    model: Any,
    input_shape: Optional[tuple[int, ...]] = None,
    *,
    show_shapes: bool = True,
) -> graphviz.Digraph:
    """Generate a 2D visualization of a neural network.

    Args:
        model: A PyTorch nn.Module or Keras Model.
        input_shape: Shape of input tensor. Required for PyTorch.
        show_shapes: Whether to display output shapes.

    Returns:
        A Graphviz Digraph that can be rendered inline or saved.

    Raises:
        InputShapeRequiredError: If input_shape not provided for PyTorch.

    Example:
        >>> model = nn.Linear(10, 5)
        >>> graph = visualize(model, input_shape=(1, 10))
    """
    ...
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new feature
fix: fix a bug
docs: update documentation
test: add or update tests
refactor: code refactoring
chore: maintenance tasks
```

## Project Structure

```
modelviz/
├── modelviz/           # Main package
│   ├── __init__.py     # Public API exports
│   ├── visualize.py    # Main visualization functions
│   ├── graph/          # Data structures
│   ├── parsers/        # Framework-specific parsers
│   ├── renderers/      # Output renderers
│   └── utils/          # Utilities
├── tests/              # Test suite
├── docs/               # Documentation
├── examples/           # Demo scripts
└── pyproject.toml      # Package configuration
```

## Need Help?

- 📖 [Documentation](docs/)
- 🐛 [Open an Issue](https://github.com/shreyanshjain05/modelviz/issues)

Thank you for contributing! 🎉
