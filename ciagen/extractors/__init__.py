from .canny import Canny
from .mediapipe import MediaPipeFace
from .openpose import OpenPose
from .segmentation import Segmentation

AVAILABLE_EXTRACTORS = ("openpose", "canny", "mediapipe_face", "segmentation")

__all__ = [
    "Canny",
    "MediaPipeFace",
    "OpenPose",
    "Segmentation",
    "AVAILABLE_EXTRACTORS",
    "extract_model_from_name",
    "instantiate_extractor",
]


def extract_model_from_name(raw_name: str) -> str:
    """Map a model name substring to the canonical extractor type."""
    if "openpose" in raw_name:
        return "openpose"
    elif "canny" in raw_name:
        return "canny"
    elif "mediapipe" in raw_name:
        return "mediapipe_face"
    elif "segmentation" in raw_name:
        return "segmentation"
    else:
        raise ValueError(f"Unknown model: {raw_name}. Available extractors: {AVAILABLE_EXTRACTORS}")


def instantiate_extractor(control_model: str, **kwargs):
    """Factory function to create an extractor instance by name."""
    if control_model not in AVAILABLE_EXTRACTORS:
        raise ValueError(f"Unknown control model: {control_model}. Available: {AVAILABLE_EXTRACTORS}")

    extractors = {
        "openpose": OpenPose,
        "canny": Canny,
        "mediapipe_face": MediaPipeFace,
        "segmentation": Segmentation,
    }

    return extractors[control_model](**kwargs)
