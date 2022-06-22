import torch
import torch.nn as nn

#! tebgh e mamool ye architecture dige ba main idea e building block:)
#! ye concept e DenseBlock tarif karde ke be in shekl amal mikone:
#! idea paper ine ke resNet ro expand kone
#! yani har laye faghat be khorooji khodesh vasl nabashe
#! balke be output hame DenseBlock ha vasle
#! feature maps from prev layers are concatenated onto the inputs of future layers
#! hala bottleNeck chie? hamoon conv 1*1 ke bahash depth ro control mikardim
#! kolan network az 2 bakhsh tashkil shode:
#! 1 - Dense Blocks ke idea aslish channel re-use hast
#! 2 - BottleNeck ke az conv + pool tashkhil shode va vazife downSampling dare
#! fargh ba ResNet: resNet miumad jaam mikard in concat mikone:)

#! ye harekat vase architecture haye mokhtalef:
#! in tedad repeat haye 1*1 -> 3*3 conv too har DenseBlock hast
#! tebgh e gofte author ha, conv inja be mani BN-ReLU-Conv hast
arc = {
    '121': [6, 12, 24, 16],
    '169': [6, 12, 32, 32],
    '201': [6, 12, 48, 32],
    '264': [6, 12, 64, 48],
}

#! Transition Layer : Done!
class TransitionalLayer(nn.Module):

    def __init__(self, input_channel, output_channel):
        super(TransitionalLayer, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=(1,1))
        self.pool = nn.AvgPool2d(kernel_size=(2,2))

    def forward(self, x):
        out = self.conv(nn.ReLU(nn.BatchNorm2d(x)))
        out = self.pool(out)
        return out

class CnnBlock(nn.Module):

    def __init__(Self, in_channels, out_channels, kernel_size, stride):
        super(CnnBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        )

    def forward(self, x):

        return self.conv(x)

class DenseBlock(nn.Module):

    def __init__(self, in_channels, prev_input, num_repeats):
        super(DenseBlock, self).__init__()
        self.prev_input = prev_input
        layers = []
        for _ in range(num_repeats):
            layers.append(CnnBlock(in_channels, in_channels*4, 1, 1))
            layers.append(CnnBlock(in_channels*4, in_channels, 3, 1))
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = torch.concatenate((out, self.prev_input))
        return out

class DenseNet(nn.Module):

    def __init__(self, arc, in_channels=3) -> None:
        super(DenseNet, self).__init__()
        self.arc = arc
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=(7,7),
            stride=(2,2),
            padding=3
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
        main_layers = []
        for i , num_layers in enumerate(self.arc):
            main_layers.append(
                DenseBlock(
                    in_channels=
                    num_repeats=num_layers
                    )
            )

            main_layers.append(
                TransitionalLayer()
            )
        self.main = nn.Sequential(*main_layers)
        self.avg_pool = nn.AdaptiveAvgPool2d()

    def forward(self, x):
        pass


def Test():
    model = DenseBlock(in_channels)

if __name__ == '__main__':
    Test()