import sys
import os

from PIL.Image import Image
from controlnet_aux.processor import Processor

sys.path.append(os.path.join(os.getcwd(), "ultralytics"))
from ultralytics import YOLO


class MediaPipeFace:
    def __init__(self, **kwargs):
        pass

    def extract(self, image: Image) -> Image:

        processor_id = "mediapipe_face"
        processor = Processor(processor_id)

        processed_image = processor(image, to_pil=True)

        return processed_image

    def __str__(self) -> str:
        return "Extractor(mediapipe_face)"
