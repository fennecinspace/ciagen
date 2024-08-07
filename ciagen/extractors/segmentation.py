import sys
import os

import numpy as np
import torch
from torchvision.transforms import ToPILImage
from PIL.Image import Image


sys.path.append(os.path.join(os.getcwd(), "ultralytics"))
from ultralytics import YOLO


class Segmentation:
    def __init__(self, **kwargs):
        self.model = YOLO("yolov8m-seg.pt")

    def extract(self, image: Image) -> Image:
        image = np.array(image)

        seg_image = self.model.predict(image)
        seg_image = seg_image[0].masks[0].data[0]

        # seg_image = torch.t(result)
        seg_image = torch.stack(3 * (seg_image,))  #
        seg_image = ToPILImage()(seg_image)

        return seg_image
