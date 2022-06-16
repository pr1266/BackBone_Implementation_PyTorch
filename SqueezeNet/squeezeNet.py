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
        self.print = PrintLayer()
        #! inja activation esh:
        self.activation = nn.ReLU(inplace=True)
        #! inja structuresh darmiad:
        self.s = nn.Conv2d(in_channels, s_channels, kernel_size=(1, 1), padding=0, stride=1)
        self.e1 = nn.Conv2d(s_channels, exp1_channels, kernel_size=(1, 1), padding=0, stride=1)
        self.e3 = nn.Conv2d(in_channels, exp3_channels, kernel_size=(3, 3), padding=0, stride=1)

    def forward(self, x):
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
        out = torch.cat([exp1, exp3], 1)
        out = self.print(out)
        return out

class SqueezeNet(nn.Module):

    def __init__(self, in_channels=3):
        super(SqueezeNet, self).__init__()
        self.input_conv = nn.Conv2D(in_channels=in_channels, out_channels=96, kernel_size=(7,7), stride=2, stride=1)

    def forward(self, x):
        pass

x = FireBlock()