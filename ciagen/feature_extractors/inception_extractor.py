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


class InceptionFE(FeatureExtractor):
    def __init__(self, softmaxed: bool = True, weights: str = "DEFAULT"):
        super().__init__()

        self.inception_model = inception_v3(weights=weights)
        self.softmaxed = softmaxed

        if self.softmaxed:
            self.softmax = Softmax(dim=1)

    def name(self):
        return "InceptionFE"

    def forward(self, x):
        self.eval()

        if len(x.size()) == 3:
            x = torch.unsqueeze(x, 0)
        x = self.inception_model(x)

        if not isinstance(x, torch.Tensor):
            x = x.logits

        if self.softmaxed:
            x = self.softmax(x)

        return x


class InceptionModel(torch.nn.Module):
    def __init__(self, weights="DEFAULT", softmaxed: bool = True):
        super().__init__()
        self.softmaxed = softmaxed
        self.inceptionv3 = inception_v3(weights=weights)

        if self.softmaxed:
            self.softmax = Softmax(dim=1)

    def forward(self, x):
        if len(x.size()) == 3:
            x = torch.unsqueeze(x, 0)
        x = self.inceptionv3(x).logits  # TODO: verify this !!!

        if self.softmaxed:
            x = self.softmax(x)

        return x
