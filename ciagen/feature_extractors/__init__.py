from .auc_extractor import AUExtractor, test_au_extractor, au_transform
from .vit_extractor import VitExtractor, vit_transform
from .inception_extractor import (
    InceptionFeatureExtractor,
    test_inception_extractor,
    inception_transform,
)

AVAILABLE_FEATURE_EXTRACTORS = (
    "au",
    "vit",
    "inception",
)


def instance_transform(feature_extractor_name: str, **kwargs):
    if feature_extractor_name not in AVAILABLE_FEATURE_EXTRACTORS:
        raise ValueError(
            f"Unknown feature extractor: {feature_extractor_name}. Please verify your configuration file."
        )

    if feature_extractor_name == "vit":
        return vit_transform()
    elif feature_extractor_name == "inception":
        return inception_transform(to_tensor=True)
        # return inception_transform()
    elif feature_extractor_name == "au":
        return au_transform()


def instance_feature_extractor(feature_extractor_name: str, **kwargs):
    if feature_extractor_name not in AVAILABLE_FEATURE_EXTRACTORS:
        raise ValueError(
            f"Unknown feature extractor: {feature_extractor_name}. Please verify your configuration file."
        )

    if feature_extractor_name == "au":
        return AUExtractor(**kwargs)
    elif feature_extractor_name == "vit":
        return VitExtractor(**kwargs)
    elif feature_extractor_name == "inception":
        return InceptionFeatureExtractor(**kwargs)
