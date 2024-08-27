from controlnet_aux.processor import Processor
from PIL.Image import Image

from ciagen.extractors.abs_extractor import ExtractorABC


class MediaPipeFace(ExtractorABC):
    name = "MediaPipeFace"

    def __init__(self, **kwargs):
        pass

    def extract(self, image: Image) -> Image:

        processor_id = "mediapipe_face"
        processor = Processor(processor_id)

        processed_image = processor(image, to_pil=True)

        return processed_image
