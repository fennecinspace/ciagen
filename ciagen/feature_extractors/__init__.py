from .auc_extractor import AUFE, au_transform
from .vit_extractor import VitFE, vit_transform
from .inception_extractor import InceptionFE, inception_transform

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


def available_feature_extractors():
    return {
        "au": AUFE,
        "vit": VitFE,
        "inception": InceptionFE,
    }


def instance_feature_extractor(feature_extractor_name: str, **kwargs):
    if feature_extractor_name not in AVAILABLE_FEATURE_EXTRACTORS:
        raise ValueError(
            f"Unknown feature extractor: {feature_extractor_name}. Please verify your configuration file."
        )

    if feature_extractor_name == "au":
        return AUFE(**kwargs)
    elif feature_extractor_name == "vit":
        return VitFE(**kwargs)
    elif feature_extractor_name == "inception":
        return InceptionFE(**kwargs)
