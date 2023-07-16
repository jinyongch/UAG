import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models.UDG.ptb_model import DistributionUncertainty
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


class ReproVGG(VGG):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return super().forward(x)


class UDGVGG(ReproVGG):
    def __init__(
        self,
        *args,
        p_drop=0.25,
        drop_stages_id=[1, 2],
        p_ptb=0.5,
        ptb_stages_id=[1, 2],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dropout = nn.Dropout(p=p_drop)
        self.ptb = DistributionUncertainty(p=p_ptb)

        self.drop_stages_dict = {
            0: "",
            1: "after_vgg",
            2: "after_ca",
            3: "after_sa",
            4: "after_front",
            5: "inner_mid",
            6: "after_mid",
            7: "after_offset1",
            8: "after_offset2",
            9: "after_offset3",
            10: "after_all",
        }

        self.ptb_stages_dict = self.drop_stages_dict

        self.drop_stages = [self.drop_stages_dict[id] for id in drop_stages_id]
        self.ptb_stages = [self.ptb_stages_dict[id] for id in ptb_stages_id]

        print(f"Dropout stages: {self.drop_stages}")
        print(f"Purterbation stages: {self.ptb_stages}")

        vgg = models.vgg16(pretrained=True)
        features = list(vgg.features.children())
        self.features1 = nn.Sequential(*features[0:4])
        self.features2 = nn.Sequential(*features[4:9])
        self.features3 = nn.Sequential(*features[9:16])
        self.features4 = nn.Sequential(*features[16:23])

    def forward(self, x):
        x = self.features1(x)
        if "after_vgg" in self.drop_stages:
            x1 = self.dropout(x)
            x2 = self.ptb(x)
            x = (x1 + x2) / 2

        x = self.features2(x)
        if "after_ca" in self.drop_stages:
            x1 = self.dropout(x)
            x2 = self.ptb(x)
            x = (x1 + x2) / 2

        x = self.features3(x)
        if "after_sa" in self.drop_stages:
            x1 = self.dropout(x)
            x2 = self.ptb(x)
            x = (x1 + x2) / 2

        x = self.features4(x)
        if "after_front" in self.drop_stages:
            x1 = self.dropout(x)
            x2 = self.ptb(x)
            x = (x1 + x2) / 2

        x = self.de_pred(x)
       
        x = F.upsample(x, scale_factor=8)

        return x
