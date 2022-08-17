import torch
import torch.nn as nn
import torch.nn.functional as F


cfg = {
    'out_planes': [200, 400, 800],
    'num_blocks': [4, 8, 4],
    'groups': 2
}

def shuffle(x, groups):
    N, C, H, W = x.size()
    out = x.view(N, groups, C//groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

    return out

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups):
        super().__init__()

        mid_channles = int(out_channels/4)

        if in_channels == 24:
            self.groups = 1
        else:
            self.groups = groups

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channles, 1, groups=self.groups, bias=False),
            nn.BatchNorm2d(mid_channles),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channles, mid_channles, 3, stride=stride, padding=1, groups=mid_channles, bias=False),
            nn.BatchNorm2d(mid_channles),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channles, out_channels, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = shuffle(out, self.groups)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.stride == 2:
            res = self.shortcut(x)
            out = F.relu(torch.cat([out, res], 1))
        else:
            out = F.relu(out+x)

        return out

class ShuffleNet(nn.Module):
    def __init__(self, groups, channel_num, class_num=10):

        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = self.make_layers(24, channel_num[0], 4, 2, groups)
        self.stage3 = self.make_layers(channel_num[0], channel_num[1], 8, 2, groups)
        self.stage4 = self.make_layers(channel_num[1], channel_num[2], 4, 2, groups)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channel_num[2], class_num)

    def make_layers(self, input_channels, output_channels, layers_num, stride, groups):

        layers = []
        layers.append(Bottleneck(input_channels, output_channels-input_channels, stride, groups))
        input_channels = output_channels

        for i in range(layers_num-1):
            Bottleneck(input_channels, output_channels, 1, groups)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x


model = ShuffleNet(2, cfg['out_planes'])
dummy = torch.randn(1, 3, 224, 224)
pred = model(dummy)
print(pred)