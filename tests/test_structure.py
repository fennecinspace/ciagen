"""Basic import tests for the ciagen package structure."""


def test_import_ciagen():
    import ciagen
    assert hasattr(ciagen, "generate")
    assert hasattr(ciagen, "evaluate")
    assert hasattr(ciagen, "filter_generated")
    assert hasattr(ciagen, "caption")


def test_import_extractors():
    from ciagen.extractors import (
        AVAILABLE_EXTRACTORS,
    )
    assert "canny" in AVAILABLE_EXTRACTORS
    assert "openpose" in AVAILABLE_EXTRACTORS


def test_import_generators():
    pass


def test_import_metrics():
    pass


def test_import_feature_extractors():
    pass


def test_import_data():
    pass


def test_import_utils():
    pass


def test_import_distances():
    pass


def test_accumulators():
    import torch

    from ciagen.metrics.accumulators import CovCalculator, MeanCalculator

    m = MeanCalculator()
    data = torch.randn(100, 10)
    m(data)
    mean = m.state()
    expected = data.mean(dim=0)
    assert torch.allclose(mean, expected, atol=1e-5)

    c = CovCalculator()
    c(data)
    cov = c.state()
    expected_cov = torch.cov(data.T)
    assert torch.allclose(cov, expected_cov, atol=1e-3)
