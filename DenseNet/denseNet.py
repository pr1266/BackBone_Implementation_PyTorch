import torch
import torch.nn as nn

#! tebgh e mamool ye architecture dige ba main idea e building block:)
#! ye concept e DenseBlock tarif karde ke be in shekl amal mikone:
#! idea paper ine ke resNet ro expand kone
#! yani har laye faghat be khorooji khodesh vasl nabashe
#! balke be output hame resBlock ha vasle
#! feature maps from prev layers are concatenated onto the inputs of future layers
#! hala bottleNeck chie? hamoon conv 1*1 ke bahash depth ro control mikardim
#! kolan network az 2 bakhsh tashkil shode:
#! 1 - Dense Blocks ke idea aslish channel re-use hast
#! 2 - BottleNeck ke az conv + pool tashkhil shode va vazife downSampling dare

#! ye harekat vase architecture haye mokhtalef:
#! in tedad repeat haye 1*1 -> 3*3 conv too har DenseBlock hast:
arc = {
    '121': [6, 12, 24, 16],
    '169': [6, 12, 32, 32],
    '201': [6, 12, 48, 32],
    '264': [6, 12, 64, 48],
}

class DenseBlock(nn.Module):

    def __init__(self,):
        super(DenseBlock, self).__init__()

    def forward(self, x):
        pass

class DenseNet(nn.Module):

    def __init__(self) -> None:
        super(DenseNet, self).__init__()
    
    def forward(self, x):
        pass


def Test():
    pass

if __name__ == '__main__':
    Test()