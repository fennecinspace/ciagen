import os

import numpy as np
import torch
from PIL.Image import Image

from ciagen import add_ultralytics_path

from ultralytics import YOLO


class FalseSegmentation(ExtractorABC):
    name = "FalseSegmentation"

    def __init__(self, **kwargs):
        self.model = YOLO(os.path.join("models", "yolov8m-seg.pt"))

    def extract(self, image: Image) -> Image:
        image = np.array(image)

        results = self.model.predict(image)
        result = results[0].masks[0].data[0]

        seg_image = torch.t(result)
        seg_image = result[None, None, ...]  # ToPILImage()(result[None, :])
        seg_image = torch.concat((seg_image, seg_image, seg_image), axis=1)

        return seg_image
