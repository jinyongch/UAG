import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models.VGG.layer import Conv2d


class VGG(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG, self).__init__()
        vgg = models.vgg16(pretrained=pretrained)

        features = list(vgg.features.children())
        self.features4 = nn.Sequential(*features[0:23])

        self.de_pred = nn.Sequential(
            Conv2d(512, 128, 1, same_padding=True, NL="relu"),
            Conv2d(128, 1, 1, same_padding=True, NL=None),
        )

    def forward(self, x):
        x = self.features4(x)
        x = self.de_pred(x)

        x = F.upsample(x, scale_factor=8)

        return x
