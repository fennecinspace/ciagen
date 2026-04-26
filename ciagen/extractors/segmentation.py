import os

import numpy as np
import torch
from PIL.Image import Image
from torchvision.transforms import ToPILImage

from ciagen.extractors.abc_extractor import ExtractorABC


class Segmentation(ExtractorABC):
    name = "Segmentation"

    def __init__(self, **kwargs):
        from ultralytics import YOLO

        model_path = os.path.join("models", "yolov8m-seg.pt")
        self.model = YOLO(model_path)

    def extract(self, image: Image) -> Image:
        image = np.array(image)
        seg_image = self.model.predict(image)
        seg_image = seg_image[0].masks[0].data[0]
        seg_image = torch.stack(3 * (seg_image,))
        seg_image = ToPILImage()(seg_image)
        return seg_image
