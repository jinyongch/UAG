import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models.ResNet.Res50 import Res50
from models.UDG.ptb_model import DistributionUncertainty


class UDGRes50(Res50):
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

        res = models.resnet50(pretrained=True)

        self.frontend = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)

        self.reslayer_1 = res.layer1
        self.reslayer_2 = res.layer2

    def forward(self, x):
        x = self.frontend(x)
        if "after_vgg" in self.drop_stages:
            x1 = self.dropout(x)
            x2 = self.ptb(x)
            x = (x1 + x2) / 2

        x = self.reslayer_1(x)
        if "after_ca" in self.drop_stages:
            x1 = self.dropout(x)
            x2 = self.ptb(x)
            x = (x1 + x2) / 2

        x = self.reslayer_2(x)
        if "after_sa" in self.drop_stages:
            x1 = self.dropout(x)
            x2 = self.ptb(x)
            x = (x1 + x2) / 2

        x = self.own_reslayer_3(x)
        if "after_front" in self.drop_stages:
            x1 = self.dropout(x)
            x2 = self.ptb(x)
            x = (x1 + x2) / 2

        x = self.de_pred(x)

        x = F.upsample(x, scale_factor=8)
        return x
