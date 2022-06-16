import torch
import torch.nn as nn

#! too paper harf jalebi zade
#! oomade mige accuracy AlexNet ro midam
#! vali ba parameter haye kamtar ve size kamtar az
#! 0.5 MB!!!! AlexNet 250 MB e!!!! jaleb shod
#! falsafash chie?
#! oomade mige ke agha fa biaim be jaye 3 * 3 az
#! kernel 1 * 1 estefade konim (hala in dige dare sampling mikone)
#! vali parameter ha 9 barabar kamtar mishan
#! in oomade az ye Fire Block estefade karde
#! fire block chie? tarkib squeeze va expand e
#! yani ye dor input data ba filter haye 1*1 conv mishe
#! baad ReLU anjam mishe roosh baad miad tooye laye expand
#! aval ba 1*1 baad ba 3*3 conv mishe baad dobare ReLU
#! in mishe fire block, albate, expand 2 tast:
#! 1*1 va 3*3 ke khorooji 1*1 voroodi 3*3 nist
#! balke ba ham concat mishan
#! hala bbinim in 1*1 vase chie?
#! ma vaghti conv ba kernel 1*1 mizanim, darvaghe darim
#! down sample mikonim. yani etelaat feature map haro summarize mikonim
#! chera? intori etelaat e ba arzesh mimoonan va etelaat e kam arzaesh filter mishan
#! be che dardi mikhore? parameter ha be shedat kam mishan va time o complexity miad paiin

class PrintLayer(nn.Module):
    def __init__(self, type='default', index='NaN'):
        super(PrintLayer, self).__init__()
        self.type = type
        self.index = index

    def forward(self, x):
        print(f'size of {self.type} layer {self.index} : {x.size()}')
        return x

class FireBlock(nn.Module):

    def __init__(self, in_channels, s_channels, exp1_channels, exp3_channels):
        super(FireBlock, self).__init__()
        #! fire block ham az conv tashkil shode pas input output dare.
        #! hala too fireBlock ma miaim ye dor squeeze! anjam midim ba kernel 1*1
        #! rooye input, baadesh miaim rooye khorooji in squeeze, 2 bar conv mizanim
        #! 1 bar dobare downsample mikonim ba kernel 1*1, 1 bar ham ba kernel 3*3 ba padding 1
        #! miaim conv mikonim baad natije ro concat mikonim. tabiiatan output shape mishe 2 barabar e 
        #! channel haye khorooji marhale expand chon 2 ta tensor ham size ro darim concat mikonim
        #! pas chi shod golaye too khone?
        #? first stage : input shape -> squeeze conv output
        #? second stage : squeeze conv output -> expand conv output
        #? return value : 2 * expand conv output
        self.in_channels = in_channels
        self.s_channels = s_channels
        self.exp1_channels = exp1_channels
        self.exp3_channels = exp3_channels

        self.print = PrintLayer()
        #! inja activation esh:
        self.activation = nn.ReLU(inplace=True)
        #! inja structuresh darmiad:
        self.s = nn.Conv2d(in_channels, s_channels, kernel_size=(1, 1), padding=0, stride=1)
        self.e1 = nn.Conv2d(s_channels, exp1_channels, kernel_size=(1, 1), padding=0, stride=1)
        self.e3 = nn.Conv2d(s_channels, exp3_channels, kernel_size=(3, 3), padding=1, stride=1)

    def forward(self, x):
        #? be shedat vazeh:
        #! aval squeeze:
        out = self.s(x)
        out = self.activation(out)
        out = self.print(out)
        #! baad expand:
        exp1 = self.e1(out)
        exp1 = self.activation(exp1)

        exp3 = self.e3(out)
        exp3 = self.activation(exp3)
        #! hala concat:
        out = torch.cat((exp1, exp3), dim=1)
        out = self.print(out)
        return out

class SqueezeNet(nn.Module):

    def __init__(self, in_channels=3):
        super(SqueezeNet, self).__init__()
        #! injam ke implement kole architucture tebgh e paper:
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(7, 7), stride=2),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.ReLU(),
            FireBlock(in_channels=96, s_channels=16, exp1_channels=64, exp3_channels=64),
            FireBlock(in_channels=128, s_channels=16, exp1_channels=64, exp3_channels=64),
            FireBlock(in_channels=128, s_channels=32, exp1_channels=128, exp3_channels=128),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            FireBlock(in_channels=256, s_channels=32, exp1_channels=128, exp3_channels=128),
            FireBlock(in_channels=256, s_channels=48, exp1_channels=192, exp3_channels=192),
            FireBlock(in_channels=384, s_channels=48, exp1_channels=192, exp3_channels=192),
            FireBlock(in_channels=384, s_channels=64, exp1_channels=256, exp3_channels=256),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            FireBlock(in_channels=512, s_channels=64, exp1_channels=256, exp3_channels=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=1000, kernel_size=(1, 1)),
            nn.AvgPool2d(kernel_size=(13, 13), stride=1)
            )

    def forward(self, x):
        out = self.net(x)
        return out.view(out.size(0), -1)


if __name__ == '__main__':
    x = torch.randn((64, 3, 224, 224))
    sn = SqueezeNet()
    out = sn(x)
    print (sn)
    print (out.shape)