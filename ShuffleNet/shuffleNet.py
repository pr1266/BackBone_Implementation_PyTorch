import torch
import torch.nn as nn
import torch.nn.functional as functions #for non linear functions

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups
    def forward(self, x):
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)

class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride
        mid_planes = int(out_planes/4)
        g = 1 if in_planes == 24 else groups
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups= g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride==2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3,stride=2, padding =1))
    def forward(self,x):
        out = functions.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = functions.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = functions.relu(torch.cat([out,res], 1)) if self.stride==2 else functions.relu(out+res)
        return out