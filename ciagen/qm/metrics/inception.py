import torch
from torch.nn import Softmax
import torch.utils
from PIL import Image
from torchvision.models import inception_v3
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor


IncSample = (
    torch.Tensor | Image.Image | torch.utils.data.DataLoader | torch.utils.data.Dataset
)


class InceptionSoftmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.inceptionv3 = inception_v3()
        self.softmax = Softmax()

    def forward(self, x):
        if len(x.size()) == 3:
            x = torch.unsqueeze(x, 0)

        x = self.inceptionv3(x)
        x = self.softmax(x)

        return x


class AddedSoftmax(torch.nn.Module):
    def __init__(self, model):
        self.model = model
        self.softmax = Softmax()

    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)

        return x


def inception_transform():
    return Compose(
        [
            Resize(299),
            CenterCrop(299),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
