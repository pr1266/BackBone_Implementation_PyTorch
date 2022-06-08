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
#! migim agha! harchi tedad layer ha bishtar bashan
#! ma feature haye high level tari ro mitoonim az tasvir
#! extract konim
#! strategy chie? mige agha man miam ye residual block tarif mikonam
#! x ro voroodi midim, ye seri conv operation roosh anjam mishe
#! dar nahayat ye F(x) be dast miad, hala ma baraye har block,
#! miaim output ro F(x) + x dar nazar migirim
#! tavajoh shavad ke agar size F(x) va x barabar nabashe,
#! bayad down_sample konim

class ResBlock(nn.Module):

    def __init__(self, in_channels, intermediate_channels, identity_downsample=None,stride=1):
        #! agha, in ResBlock ha baraye arc haye mokhtalef fargh daran
        #! memari 18 layer o 34 layer, residual block hashoon 2 taiie va
        #! oon expansion e akharam nadaran
        #! vali too arc haye 50, 101, 152 layer in shekli dar miad:
        super(ResBlock, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        #! moteghayer e identity, hamoon x e
        #! ke baadan be F(x) ezafe mikonim:
        identity = x.clone()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        out += identity
        out = self.relu(out)
        return out
    