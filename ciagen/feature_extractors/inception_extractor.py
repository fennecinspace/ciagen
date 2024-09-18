from typing import List

import numpy as np
import torch
import torch.utils
import torchvision.transforms as transforms
from PIL import Image
from torch.nn import Softmax
from torchvision.models import inception_v3
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from ciagen.feature_extractors.abc_feature_extractor import FeatureExtractor


# TODO: maybe we need to add to tensor here
def inception_transform(to_tensor=False):
    if to_tensor:
        return transforms.Compose(
            [
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        return Compose(
            [
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


class InceptionFeatureExtractor(FeatureExtractor):
    def __init__(self, softmaxed=True, weights="DEFAULT"):
        super().__init__()
        self.inc_model = (
            InceptionModelSoftmaxed(weights=weights)
            if softmaxed
            else InceptionModel(weights=weights)
        )

    def forward(self, x):
        return self.inc_model(x)


class InceptionModel(torch.nn.Module):
    def __init__(self, weights="DEFAULT"):
        super().__init__()
        self.inceptionv3 = inception_v3(weights=weights)

    def forward(self, x):
        if len(x.size()) == 3:
            x = torch.unsqueeze(x, 0)
        x = self.inceptionv3(x)

        return x


class InceptionModelSoftmaxed(torch.nn.Module):
    def __init__(self, weights="DEFAULT"):
        super().__init__()

        # https://pytorch.org/vision/stable/models/generated/torchvision.models.inception_v3.html#torchvision.models.Inception_V3_Weights
        self.inception_model = InceptionModel(weights=weights)
        self.softmax = Softmax()

    def forward(self, x):
        x = self.inception_model(x)
        x = self.softmax(x)

        return x


def test_inception_extractor():
    pass
