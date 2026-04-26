from .inception_extractor import InceptionFE, inception_transform
from .vit_extractor import VitFE, vit_transform

AVAILABLE_FEATURE_EXTRACTORS = (
    "vit",
    "inception",
)


def instance_transform(feature_extractor_name: str, **kwargs):
    if feature_extractor_name not in AVAILABLE_FEATURE_EXTRACTORS:
        raise ValueError(f"Unknown feature extractor: {feature_extractor_name}. Please verify your configuration file.")

    if feature_extractor_name == "vit":
        return vit_transform()
    elif feature_extractor_name == "inception":
        return inception_transform(to_tensor=True)


def available_feature_extractors():
    return {
        "vit": VitFE,
        "inception": InceptionFE,
    }


def instance_feature_extractor(feature_extractor_name: str, **kwargs):
    if feature_extractor_name not in AVAILABLE_FEATURE_EXTRACTORS:
        raise ValueError(f"Unknown feature extractor: {feature_extractor_name}. Please verify your configuration file.")

    if feature_extractor_name == "vit":
        return VitFE(**kwargs)
    elif feature_extractor_name == "inception":
        return InceptionFE(**kwargs)
