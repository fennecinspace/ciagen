import sys
import os

import numpy as np
from controlnet_aux import OpenposeDetector
from PIL.Image import Image


sys.path.append(os.path.join(os.getcwd(), "ultralytics"))
from ultralytics import YOLO


class OpenPose:
    def __init__(self, model: str = "lllyasviel/ControlNet", **kwargs):
        self.model = OpenposeDetector.from_pretrained(model)

    def extract(self, image: Image) -> Image:
        image = np.array(image)
        pose = self.model(image)
        return pose

    def __str__(self) -> str:
        return "Extractor(openpose)"
