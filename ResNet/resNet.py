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
    

class ResNet(nn.Module):
    def __init__(self, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            ResBlock(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(ResBlock(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)