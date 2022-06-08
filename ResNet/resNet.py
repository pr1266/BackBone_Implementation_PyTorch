import torch
import torch.nn as nn

#! por citation tarin maghale computer vision donya !!!!
#! ajab shabake ii !!
#! falsafe ResNet :
#! agha mige ke age ma, biaim tedad layer haro ziad konim
#! ghaedatan bayad accuracy bere bala
#! ama dar amal in shekli nist
#! pas che mikonim? residual intorie ke miaim natayej
#! layer haro ke mohasebe kardim, ba natije conputation
#! 2 laye baadi jam mikonim
#! intori yademoon mimoone chi ghablan extract kardim

class Block(nn.Module):

    def __init__(self, in_channels, out_channels, identity_downsample=None,stride=1):
        super(Block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )