import torch
import torch.nn as nn
import torch.nn.functional as F

from models.UDG.ASPDNet import ASPDNet
from models.UDG.ptb_model import DistributionUncertainty


class UDG(ASPDNet):
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

        assert (
            self.ptb_stages == self.drop_stages
        ), f"stages of ptb / drop mismatch. \n ptb: {self.ptb_stages}, \
            \n drop: {self.drop_stages}"

    def forward(self, x):
        x = self.frontend(x)

        residual = x
        if "after_vgg" in self.drop_stages:
            x1 = self.dropout(x)
            x2 = self.ptb(x)
            x = (x1 + x2) / 2

        x = self.ca(x) * x
        if "after_ca" in self.drop_stages:
            x1 = self.dropout(x)
            x2 = self.ptb(x)
            x = (x1 + x2) / 2

        x = self.sa(x) * x
        if "after_sa" in self.drop_stages:
            x1 = self.dropout(x)
            x2 = self.ptb(x)
            x = (x1 + x2) / 2

        x += residual
        if "after_front" in self.drop_stages:
            x1 = self.dropout(x)
            x2 = self.ptb(x)
            x = (x1 + x2) / 2

        x1 = self.conv4_3_1(x)
        x2 = self.conv4_3_2(x)
        x3 = self.conv4_3_2(x)
        x4 = self.conv4_3_2(x)
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.conv5(x)
        if "inner_mid" in self.drop_stages:
            x1 = self.dropout(x)
            x2 = self.ptb(x)
            x = (x1 + x2) / 2
        x = self.mid_end(x)

        if "after_mid" in self.drop_stages:
            x1 = self.dropout(x)
            x2 = self.ptb(x)
            x = (x1 + x2) / 2

        offset1 = self.offset1(x)
        x = F.relu(self.conv6_1(x, offset1))
        x = self.bn6_1(x)
        if "after_offset1" in self.drop_stages:
            x1 = self.dropout(x)
            x2 = self.ptb(x)
            x = (x1 + x2) / 2

        offset2 = self.offset2(x)
        x = F.relu(self.conv6_2(x, offset2))
        x = self.bn6_2(x)
        if "after_offset2" in self.drop_stages:
            x1 = self.dropout(x)
            x2 = self.ptb(x)
            x = (x1 + x2) / 2

        offset3 = self.offset3(x)
        x = F.relu(self.conv6_3(x, offset3))
        x = self.bn6_3(x)
        if "after_offset3" in self.drop_stages:
            x1 = self.dropout(x)
            x2 = self.ptb(x)
            x = (x1 + x2) / 2

        x = self.output_layer(x)
        if "after_all" in self.drop_stages:
            x1 = self.dropout(x)
            x2 = self.ptb(x)
            x = (x1 + x2) / 2

        return x
