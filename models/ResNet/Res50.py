import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models.ResNet.Res50 import Res50
from models.ResNet.utils import initialize_weights
from models.UDG.ptb_model import DistributionUncertainty
from models.VGG.layer import Conv2d


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


class Res50(nn.Module):
    def __init__(self, pretrained=True):
        super(Res50, self).__init__()

        self.de_pred = nn.Sequential(
            Conv2d(1024, 128, 1, same_padding=True, NL="relu"),
            Conv2d(128, 1, 1, same_padding=True, NL=None),
        )

        initialize_weights(self.modules())

        res = models.resnet50(pretrained=pretrained)

        self.frontend = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2
        )
        self.own_reslayer_3 = make_res_layer(Bottleneck, 256, 6, stride=1)
        self.own_reslayer_3.load_state_dict(res.layer3.state_dict())

    def forward(self, x):
        x = self.frontend(x)

        x = self.own_reslayer_3(x)

        x = self.de_pred(x)

        x = F.upsample(x, scale_factor=8)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.fill_(1)
                m.bias.data.fill_(0)


def make_res_layer(block, planes, blocks, stride=1):
    downsample = None
    inplanes = 512
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
