import torch
import torch.nn as nn
import torchvision.models as models
import os
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
growthRate = 32
arc = {
    '121': [6, 12, 24, 16],
    '169': [6, 12, 32, 32],
    '201': [6, 12, 48, 32],
    '264': [6, 12, 64, 48],
}

#! Transition Layer : Done!
class TransitionalLayer(nn.Module):

    def __init__(self, input_channel):
        super(TransitionalLayer, self).__init__()
        #! transition bayad nesf kone channel haro
        self.input_channel = input_channel
        self.conv = nn.Conv2d(in_channels=int(input_channel), out_channels=int(input_channel//2), kernel_size=(1,1), padding=0, stride=1)
        self.pool = nn.AvgPool2d(kernel_size=(2,2))

    def forward(self, x):
        out = self.conv(x)
        out = self.pool(out)
        return out

class PrintLayer(nn.Module):

    def __init__(self, name='default', index='NaN'):
        super(PrintLayer, self).__init__()
        self.name = name
        self.index = index
    
    def forward(self, x):
        print(f'\n{self.name} layer {self.index} output size : {x.size()}')
        return x

class CnnBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super(CnnBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(int(in_channels)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=int(in_channels), out_channels=int(out_channels), kernel_size=kernel_size, padding=padding, stride=stride,)
        )

    def forward(self, x):
        return self.conv(x)

class DenseBlock(nn.Module):

    def __init__(self, in_channels, num_repeats, index):
        super(DenseBlock, self).__init__()
        self.in_channels = in_channels
        self.num_repeats = num_repeats
        self.index = index
    
    def forward(self, x):
        for _ in range(self.num_repeats):
            inp = x
            x = CnnBlock(in_channels = self.in_channels, out_channels = growthRate*4, kernel_size=1, padding=0)(x)
            x = CnnBlock(in_channels = growthRate*4, out_channels = growthRate, kernel_size=3, padding=1)(x)
            x = torch.cat((x, inp), 1)            
            self.in_channels = x.size(1)
        return x
      
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
        self.pool1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=1)
        main_layers = []
        dense_shape = 64
        for i , num_layers in enumerate(self.arc):            
            
            main_layers.append(
                DenseBlock(
                    in_channels=dense_shape,
                    num_repeats=num_layers,
                    index=i
                    )
                )
            main_layers.append(PrintLayer(name='DenseBlock', index=i+1))
            s = dense_shape*4
            if i != 3:
                main_layers.append(
                    TransitionalLayer(input_channel=s)
                )
                main_layers.append(PrintLayer(name='TransitionBlock', index=i+1))
            
            dense_shape = s/2
            
        self.main = nn.Sequential(*main_layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(in_features=int(dense_shape), out_features=1000)
        self.dense_shape = dense_shape

    def forward(self, x):
        print('input size of DenseNet : ', x.size())
        out = self.conv1(x)
        out = PrintLayer(name='normal conv', index='1')(out)
        out = self.pool1(out)
        out = PrintLayer(name='normal max pooling', index='1')(out)
        out = self.main(out)
        out = self.avg_pool(out)
        out = PrintLayer(name='last average pooling (global)', index='1')(out)
        out = out.view(out.size(1), -1)
        out = out.transpose(0, 1)
        out = self.linear(out)
        out = PrintLayer(name='linear', index='1')(out)
        
        return out

def Test():
    model = DenseNet(arc=arc['121'])
    x = torch.randn(1, 3, 224, 224)
    model(x)
    print('\n', model)

if __name__ == '__main__':
    os.system('cls')
    Test()