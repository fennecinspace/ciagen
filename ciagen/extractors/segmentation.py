import os
import sys

import numpy as np
import torch
from PIL.Image import Image
from torchvision.transforms import ToPILImage

from ciagen import add_ultralytics_path

add_ultralytics_path()
from ultralytics import YOLO

from ciagen.extractors.abs_extractor import ExtractorABC


class Segmentation(ExtractorABC):
    name = "Segmentation"

    def __init__(self, **kwargs):
        self.model = YOLO(os.path.join("models", "yolov8m-seg.pt"))

    def extract(self, image: Image) -> Image:
        image = np.array(image)

        seg_image = self.model.predict(image)
        seg_image = seg_image[0].masks[0].data[0]

        # seg_image = torch.t(result)
        seg_image = torch.stack(3 * (seg_image,))  #
        seg_image = ToPILImage()(seg_image)

        return seg_image
