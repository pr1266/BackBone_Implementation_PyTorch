import torch
import torch.nn as nn
import torch.nn.functional as functions

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        #! here we identify groups count:
        super(ShuffleBlock, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        #! here split tensor size into batch_size, channels, height and width:
        N,C,H,W = x.size()
        g = self.groups
        #! and here we perform shuffle operation:
        #! indeed, we dont have any weights and computations but a single reshape operation in this block:
        return x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)

class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride
        #! here we identify number of middle channels of bottleneck sub-network:
        mid_planes = int(out_planes/4)
        #! as mentioned in paper, we dont apply group convolution for stage 2
        #! and only stage 2's input channel is 24
        #! in addition, we dont use channel shuffle in stage 2 so we define g to seperate second stage operation from other stages:
        g = 1 if in_planes == 24 else groups
        #! first of all, 1 X 1 point-wise convolution:
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        
        #! channel shuffle as mentioned above:
        self.shuffle1 = ShuffleBlock(groups=g)
        
        #! then 3 X 3 conv convolution
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        
        #! and finaly last 1 X 1 point-wise convolution:
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        
        self.shortcut = nn.Sequential()
        #! if stride == 2, we dont have repeats, then as mentioned in paper, we use an average pool layer
        #! with 3 X 3 kernel size and we concat the result of bottleneck with skip-connection that accualy is avg-pooling applied on input tensor
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))
    
    def forward(self,x):
        out = functions.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = functions.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        print(f'out shape :{out.shape}')
        res = self.shortcut(x)
        print(f'res shape :{res.shape}')
        out = functions.relu(torch.cat([out,res], 1)) if self.stride==2 else functions.relu(out+res)
        return out

class ShuffleNet(nn.Module):
  def __init__(self, cfg):
    super(ShuffleNet, self).__init__()
    out_planes = cfg['out_planes']
    num_blocks = cfg['num_blocks']
    groups = cfg['groups']
    self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias = False)
    self.bn1 = nn.BatchNorm2d(24)
    self.in_planes = 24
    self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
    self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
    self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
    self.linear = nn.Linear(out_planes[2], 10) #10 as there are 10 classes

  def _make_layer(self, out_planes, num_blocks, groups):
    layers = []
    for i in range(num_blocks):
      stride = 2 if i == 0 else 1
      cat_planes = self.in_planes if i==0 else 0
      layers.append(Bottleneck(self.in_planes, out_planes-cat_planes, stride=stride, groups=groups))
      self.in_planes = out_planes
    return nn.Sequential(*layers)
  
  def forward(self,x):
    out = functions.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = functions.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out
    
def ShuffleNetG2():
    cfg = {
        'out_planes': [200, 400, 800],
        'num_blocks': [4, 8, 4],
        'groups': 2
    }
    return ShuffleNet(cfg)
