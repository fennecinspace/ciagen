import torch
from torchvision.models import inception_v3
import torchvision.transforms as transforms
from torch.nn import Softmax
import torch.utils
from PIL import Image
from torchvision.models import inception_v3
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

import numpy as np

from typing import List
from ciagen.feature_extractors.abc_feature_extractor import FeatureExtractor, SampleT

IncSample = (
    torch.Tensor | Image.Image | torch.utils.data.DataLoader | torch.utils.data.Dataset
)


class InceptionFeatureExtractor(FeatureExtractor):
    def __init__(self):
        self.inc_model = InceptionModel()

    def extract(
        self, samples: List[SampleT | IncSample] | SampleT | IncSample, **kwargs
    ) -> List[SampleT] | SampleT:
        return self.inc_model(samples)


class InceptionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inceptionv3 = inception_v3(weights="DEFAULT")

    def forward(self, x):
        if len(x.size()) == 3:
            x = torch.unsqueeze(x, 0)
        # x= torch.unsqueeze(x,0)
        x = self.inceptionv3(x)

        return x


def inception_transform():
    return transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class InceptionSoftmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # https://pytorch.org/vision/stable/models/generated/torchvision.models.inception_v3.html#torchvision.models.Inception_V3_Weights
        self.inceptionv3 = inception_v3(weights="DEFAULT")
        self.softmax = Softmax()

    def forward(self, x):
        if len(x.size()) == 3:
            x = torch.unsqueeze(x, 0)

        x = self.inceptionv3(x)
        x = self.softmax(x)

        return x


def test_inception_extractor():
    pass
