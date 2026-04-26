"""Basic import tests for the ciagen package structure."""


def test_import_ciagen():
    import ciagen
    assert hasattr(ciagen, "generate")
    assert hasattr(ciagen, "evaluate")
    assert hasattr(ciagen, "filter_generated")
    assert hasattr(ciagen, "caption")


def test_import_extractors():
    from ciagen.extractors import (
        Canny,
        OpenPose,
        Segmentation,
        MediaPipeFace,
        AVAILABLE_EXTRACTORS,
        instantiate_extractor,
    )
    assert "canny" in AVAILABLE_EXTRACTORS
    assert "openpose" in AVAILABLE_EXTRACTORS


def test_import_generators():
    from ciagen.generators import SDCN, NaivePromptGenerator


def test_import_metrics():
    from ciagen.metrics.fid import FID
    from ciagen.metrics.inception_score import IS
    from ciagen.metrics.mahalanobis import MLD
    from ciagen.metrics.accumulators import MeanCalculator, CovCalculator, KLISCalculator


def test_import_feature_extractors():
    from ciagen.feature_extractors import (
        VitFE,
        InceptionFE,
        instance_transform,
        instance_feature_extractor,
        available_feature_extractors,
    )


def test_import_data():
    from ciagen.data import (
        generate_all_paths,
        get_model_config,
        create_local_dataloader,
        load_images_from_directory,
        select_equal_classes,
        create_csv_file,
    )


def test_import_utils():
    from ciagen.utils import (
        list_files,
        read_caption,
        calculate_iou,
        bbox_min_max_to_center_dims,
    )


def test_import_distances():
    from ciagen.metrics.distances import frechet_distance_gaussian_version
    from ciagen.metrics.distances.wasserstein import wasserstein_distance_gaussian_version
    from ciagen.metrics.distances.mmd import mmd_unbiased_estimator
    from ciagen.metrics.distances.kernel import rbf_kernel_generator
    from ciagen.metrics.distances.mahalanobis import mahalanobis_distance_calc


def test_accumulators():
    import torch
    from ciagen.metrics.accumulators import MeanCalculator, CovCalculator

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
