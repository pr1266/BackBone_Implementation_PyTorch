from tokenize import group
import torch
import torch.nn as nn

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
    def __init__(self, in_channels):
        super(DepthWise, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), padding=(1,1), stride=(1,1), groups=in_channels),
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
    
