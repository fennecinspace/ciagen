import json
import os
import re
from pathlib import Path
from typing import List

import hydra
import numpy as np
import torch
import torchvision
from torchvision.transforms import Compose, ToTensor, Resize, PILToTensor
#from feat import Detector
from omegaconf import DictConfig
from PIL.Image import Image
from tqdm import tqdm

from ciagen.feature_extractors.abc_feature_extractor import FeatureExtractor
from ciagen.utils.common import ciagen_logger


def au_transform():
    # We need the same size for all images
    return Compose(
        [
            PILToTensor(),
            Resize((299, 299)),
        ]
    )


class AUFE(FeatureExtractor):
    def __init__(self, device="cpu", **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.detector = Detector(device=self.device)

    @classmethod
    def allows_for_gpu(cls) -> bool:
        return False

    def name(self):
        return "AUFE"

    def forward(self, x, **kwargs):
        face_model_kwargs = kwargs.pop("face_model_kwargs", dict())
        landmark_model_kwargs = kwargs.pop("landmark_model_kwargs", dict())
        au_model_kwargs = kwargs.pop("au_model_kwargs", dict())
        face_detection_threshold = kwargs.pop("face_detection_threshold", 0.5)

        faces = self.detector.detect_faces(
            x, face_detection_threshold, **face_model_kwargs
        )
        landmarks = self.detector.detect_landmarks(
            x, detected_faces=faces, **landmark_model_kwargs
        )
        aus = self.detector.detect_aus(x, landmarks, **au_model_kwargs)

        # TODO: is this the fastest way ?
        for i in range(len(aus)):
            if not len(aus[i]):
                aus[i] = np.ones((20))
            else:
                aus[i] = aus[i][0]
        result = torch.stack([torch.from_numpy(au) for au in aus])

        return result
