import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.system('cls')

cfg = {
    'out_planes': [200, 400, 800],
    'num_blocks': [4, 8, 4],
    'groups': 2
}

def shuffle(x, groups):
    #! here we split tensor to batch_size, num_channels, height and width
    N, C, H, W = x.size()
    #! shuffle operation:
    out = x.view(N, groups, C//groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

    return out

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups):
        super().__init__()
        #! as mentioned in paper, first we downsample tensor channels to out_channels/4 as number of middle channels of bottleneck and then upsample it to output channels
        mid_channles = int(out_channels/4)

        #! if in_channels == 24 then we are in stage 2 and use 1 as group because we dont want to use depthwise seperable convolution
        if in_channels == 24:
            self.groups = 1
        else:
            self.groups = groups

        #! first 1X1 conv for down sampling
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channles, 1, groups=self.groups, bias=False),
            nn.BatchNorm2d(mid_channles),
            nn.ReLU(inplace=True)
        )

        #! then 3X3 conv for feature extraction
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channles, mid_channles, 3, stride=stride, padding=1, groups=mid_channles, bias=False),
            nn.BatchNorm2d(mid_channles),
            nn.ReLU(inplace=True),
        )

        #! finaly 1X1 conv for up sampling to output channels
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channles, out_channels, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        #! skip connection:
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

def test():
    
    model = ShuffleNet(2, cfg['out_planes'])
    dummy = torch.randn(1, 3, 224, 224)
    pred = model(dummy)
    print(model)

if __name__ == '__main__':
    test()