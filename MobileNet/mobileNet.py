from tokenize import group
import torch
import torch.nn as nn
from collections import OrderedDict
import os
"""
miresim be architecture mobile net
idea asli paper moarefi "DepthWise Seperable Conv Layer" ast
in dw layer az 2 bakhsh tashkil shode:
1 - marhale aval miaim be jaye conv az depthwise conv estefade mikonim
yani har kernel faghat ba yeki az filter ha ke filter e naziresh hast conv mishe
yani age voroodi 28 * 28 * 196 e, miaim 196 ta kernel estefade mikonim va
har kodoom az filter ha ba ye kernel conv mishe.
2 - hala baadesh miaim az conv 1*1 estefade mikonim ta feature map haye
dar dastres ro summarize konim va hajm mohasebat biad paiin ta beshe azash
too app haye mobile estefade konim
"""

#! 2 ta class mizaram yeki depth wise yeki point wise
#! depth wise hamoone ke kernel ha nazir be nazir ba channel han
#! point wise amaliat conv 1*1 ro anjam mide
#! tavajoh shavad ke vaghti dar conv, groups barabar e in_channel bashe
#! shabake depth wise mishe

class DepthWise(nn.Module):
    def __init__(self, in_channels, strides):
        super(DepthWise, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), padding=(1,1), stride=strides, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class PointWise(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointWise, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    

#! hala jofteshoono mizarim too ye class ta depth-wise-seperable-conv besazam:
#! mishod hamoon aval ham hamaro too ye class gozasht
class DepthWiseSeparableConv(nn.Module):

    def __init__(self, in_features, out_features, strides):
        super(DepthWiseSeparableConv, self).__init__()
        self.dw = DepthWise(in_channels = in_features, strides=strides)
        self.pw = PointWise(in_channels = in_features, out_channels = out_features)

    def forward(self, x):
        return self.pw(self.dw(x))


class MyMobileNet(nn.Module):
    def __init__(self, in_channels=3, num_filter=32, num_classes=1000):
        super(MyMobileNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, num_filter, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(num_filter),
            nn.ReLU(inplace=True)
        )

        self.in_fts = num_filter
        #! in amoo oomade mige age len element 1 bood stride 1,1
        #! age list bood ba len 1, stride 2,2
        #! age list bood va len 2, tedad repeat ha va tedad filter ha moshakhas mishe:

        # if type of sublist is list --> means make stride=(2,2)
        # also check for length of sublist
        # if length = 1 --> means stride=(2,2)
        # if length = 2 --> means (num_times, num_filter)
        self.nlayer_filter = [
            num_filter * 2,  # no list() type --> default stride=(1,1)
            [num_filter * pow(2, 2)],  # list() type and length is 1 --> means put stride=(2,2)
            num_filter * pow(2, 2),
            [num_filter * pow(2, 3)],
            num_filter * pow(2, 3),
            [num_filter * pow(2, 4)],
            # list() type --> check length for this list = 2 --> means (n_times, num_filter)
            [5, num_filter * pow(2, 4)],
            [num_filter * pow(2, 5)],
            num_filter * pow(2, 5)
        ]

        self.DSC = self.layer_construct()

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Linear(1024, num_classes),
            nn.Softmax()
        )

    def forward(self, input_image):
        N = input_image.shape[0]
        x = self.conv(input_image)
        x = self.DSC(x)
        x = self.avgpool(x)
        x = x.reshape(N, -1)
        x = self.fc(x)
        return x

    def layer_construct(self):
        block = OrderedDict()
        index = 1
        for l in self.nlayer_filter:
            if type(l) == list:
                if len(l) == 2:  # (num_times, out_channel)
                    for _ in range(l[0]):
                        block[str(index)] = DepthWiseSeparableConv(self.in_fts, l[1], strides=(1,1))
                        index += 1
                else:  # stride(2,2)
                    block[str(index)] = DepthWiseSeparableConv(self.in_fts, l[0], strides=(2, 2))
                    self.in_fts = l[0]
                    index += 1
            else:
                block[str(index)] = DepthWiseSeparableConv(self.in_fts, l, strides=(1,1))
                self.in_fts = l
                index += 1

        return nn.Sequential(block)

if __name__ == '__main__':
    os.system('cls')
    x = torch.randn((1,3,224,224))
    model = MyMobileNet()
    print(model(x))