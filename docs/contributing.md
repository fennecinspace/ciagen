---
title: Contributing
description: How to contribute to CIA : dev setup, code style, testing, adding extractors and metrics.
keywords: contributing, development, guide
---

# Contributing

Thank you for your interest in contributing to CIA.

## Quick Start

1. **Fork** the repository
2. **Clone** your fork locally
3. Create a **virtual environment** and install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev,all]"
   ```
4. Create a **feature branch**: `git checkout -b feature/my-feature`
5. Make your changes, add tests, and ensure everything passes
6. **Commit** with a clear message
7. **Push** and open a Pull Request

## Code Style

- Python 3.10+ with type annotations on all public functions
- 120 character line limit
- [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
- No comments unless they explain *why*, not *what*

```bash
ruff check ciagen/ tests/
ruff format ciagen/ tests/
```

## Testing

```bash
pytest                      # All tests
pytest tests/test_structure.py  # Specific file
pytest -v                   # Verbose
```

## Adding New Features

### New Condition Extractor

1. Create `ciagen/extractors/my_extractor.py` subclassing `ExtractorABC`
2. Register in `ciagen/extractors/__init__.py`
3. Add a ControlNet config in `ciagen/conf/config.yaml`

See [Custom Extractors](extending/extractors.md) for full details.

### New Quality Metric

1. Create `ciagen/metrics/my_metric.py` subclassing `QualityMetric`
2. Register in `ciagen/api/evaluate.py`
3. If needed, add distance math to `ciagen/metrics/distances/`

See [Custom Metrics](extending/metrics.md) for full details.

### New Feature Extractor

1. Create `ciagen/feature_extractors/my_extractor.py` subclassing `FeatureExtractor`
2. Register in `ciagen/feature_extractors/__init__.py` with a transform function

See [Custom Feature Extractors](extending/feature-extractors.md) for full details.

## Commit Messages

Use clear, imperative-style messages:

```
Add depth-map condition extractor
Fix FID score computation for single-batch inputs
Remove deprecated AU feature extractor
```

## Pull Requests

- Keep PRs focused on a single change
- Include tests for new functionality
- Update documentation if you change public API
- Ensure `ruff check` and `pytest` pass before submitting

## License

By contributing, you agree that your contributions will be licensed under the [GNU AGPL v3](https://github.com/user/synthetic-augmentation/blob/main/LICENSE).
