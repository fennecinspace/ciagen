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
    from ciagen.generators import SDCN

    assert SDCN is not None


def test_import_metrics():
    from ciagen.metrics import fid, mahalanobis

    assert fid is not None
    assert mahalanobis is not None


def test_import_feature_extractors():
    from ciagen.feature_extractors import InceptionFE, VitFE

    assert VitFE is not None
    assert InceptionFE is not None


def test_import_data():
    from ciagen.data import create_local_dataloader, load_images_from_directory

    assert load_images_from_directory is not None
    assert create_local_dataloader is not None


def test_import_utils():
    from ciagen.utils import list_files, read_caption

    assert list_files is not None
    assert read_caption is not None


def test_import_distances():
    from ciagen.metrics.distances import frechet_distance_gaussian_version

    assert frechet_distance_gaussian_version is not None


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
