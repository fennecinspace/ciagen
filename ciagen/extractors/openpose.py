import numpy as np
from controlnet_aux import OpenposeDetector
from PIL.Image import Image

from ciagen.extractors.abs_extractor import ExtractorABC


class OpenPose(ExtractorABC):
    name = "OpenPose"

    def __init__(self, model: str = "lllyasviel/ControlNet", **kwargs):
        self.model = OpenposeDetector.from_pretrained(model)

    def extract(self, image: Image) -> Image:
        image = np.array(image)
        pose = self.model(image)
        return pose
