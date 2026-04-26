# Contributing to CIA (Controllable Image Augmentation)

Thank you for your interest in contributing! This document provides guidelines and instructions.

## Quick Start

1. **Fork** the repository
2. **Clone** your fork locally
3. Create a **virtual environment** and install the project with dev dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev,all]"
   ```
4. Create a **feature branch**: `git checkout -b feature/my-feature`
5. Make your changes, add tests, and ensure everything passes
6. **Commit** with a clear message
7. **Push** and open a Pull Request

## Development Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (recommended for generation/training tasks)

### Install

```bash
# Core library + dev tools
pip install -e ".[dev]"

# With all optional dependencies (captioning, training, datasets)
pip install -e ".[dev,all]"
```

### Docker (optional)

```bash
./run_and_build_docker_file.sh nvidia
docker exec -it ciagen zsh
```

## Project Structure

```
ciagen/
├── api/                 # Public Python API (start here)
├── generators/          # Stable Diffusion + ControlNet pipeline
├── extractors/          # Condition extractors (canny, openpose, etc.)
├── metrics/             # Quality metrics (FID, IS, Mahalanobis)
├── feature_extractors/  # Deep feature extractors (ViT, Inception)
├── data/                # Data loading, paths, dataset utilities
├── captioning/          # Auto-captioning (OpenAI, Ollama)
├── utils/               # Shared utilities (IO, image, bbox)
├── conf/                # Hydra configuration files
└── _cli.py              # CLI entry point

examples/                # Dataset preparation and training scripts
tests/                   # Test suite
docs/                    # Documentation source
```

## Code Style

- **Python 3.10+** with type annotations on all public functions
- **Line length**: 120 characters max
- **Linting**: [Ruff](https://docs.astral.sh/ruff/) is configured in `pyproject.toml`
- **No comments** unless they explain *why*, not *what*
- Follow existing patterns in the codebase

### Running the linter

```bash
ruff check ciagen/ tests/
ruff format ciagen/ tests/
```

## Adding a New Feature

### New Condition Extractor

1. Create a new file in `ciagen/extractors/` (e.g., `depth.py`)
2. Subclass `ExtractorABC` and implement `extract(image) -> Image`
3. Register it in `ciagen/extractors/__init__.py`:
   - Add to `AVAILABLE_EXTRACTORS`
   - Add to `instantiate_extractor()`
4. Add a corresponding ControlNet config in `ciagen/conf/config.yaml`

### New Quality Metric

1. Create a new file in `ciagen/metrics/` (e.g., `kid.py`)
2. Subclass `QualityMetric` and implement `score()` and `name()`
3. Register it in the appropriate API module (`ciagen/api/evaluate.py`)
4. If it uses a new distance function, add it to `ciagen/metrics/distances/`

### New Feature Extractor

1. Create a new file in `ciagen/feature_extractors/` (e.g., `clip.py`)
2. Subclass `FeatureExtractor` (inherits from `ABC` and `torch.nn.Module`)
3. Implement `forward()`, `name()`, and `allows_for_gpu()`
4. Register in `ciagen/feature_extractors/__init__.py` with a transform function

## Testing

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_structure.py

# Run with verbose output
pytest -v
```

Tests should cover:
- Import integrity (all modules load correctly)
- Core logic (accumulators, distance functions, transforms)
- API contracts (functions accept and return documented types)

## Commit Messages

Use clear, imperative-style commit messages:

```
Add depth-map condition extractor
Fix FID score computation for single-batch inputs
Remove deprecated AU feature extractor
Update ViT feature extractor to use transformers v4.40 API
```

## Pull Requests

- Keep PRs focused on a single change
- Include tests for new functionality
- Update documentation if you change public API
- Ensure `ruff check` and `pytest` pass before submitting

## Reporting Issues

When filing a bug report, please include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Relevant configuration (sanitize API keys)
- Full error traceback

## License

By contributing, you agree that your contributions will be licensed under the [GNU AGPL v3](LICENSE).
