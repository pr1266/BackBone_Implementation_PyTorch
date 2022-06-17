import torch
import torch.nn as nn

#! paper idea: kodoom kernel size khoobe?
#! baad miad mige bitch please hamasho baram bezar baad concat kon
#! yani 1*1, 3*3, 5*5 conv va 3*3 max pooling ro anjam mide va concat mikone
#! dar vaghea inam mes e resnet o SE o ina idea aslish ye building block e
#! inception block size height o width ro taghir nemide balke channel haro expand mikone
#! baad ye chizi, tedad filter ha tooye harkodoom az conv ha ke goftam fargh dare
#! va tebgh e paper taghir mikone

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))