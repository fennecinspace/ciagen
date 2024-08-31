from .auc_extractor import AUExtractor, test_au_extractor
from .vit_extractor import VitExtractor, test_vit_extractor


AVAILABLE_FEATURE_EXTRACTORS = (
    "au_extractor",
    "vit_extractor",
)


def instance_feature_extractor(feature_extractor_name: str, **kwargs):
    if feature_extractor_name not in AVAILABLE_FEATURE_EXTRACTORS:
        raise ValueError(
            f"Unknown feature extractor: {feature_extractor_name}. Please verify your configuration file."
        )

    if feature_extractor_name == "au_extractor":
        return AUExtractor(**kwargs)
    elif feature_extractor_name == "vit_extractor":
        return VitExtractor(**kwargs)
