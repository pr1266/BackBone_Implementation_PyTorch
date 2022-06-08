import torch
import torch.nn as nn
#! khob hajimoon VGG
#! yeki az maroof tarin backbone haye transfer learning

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M",],
    "VGG19": [ 64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M",],
}

class VGG(nn.Module):

    def __init__(self, type='VGG16'):
        self.num_classes = 1000
        self.in_channels = 3
        self.type = type
        self.conv_layers = self.create_conv_layers(VGG_types[self.type])

        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.num_classes)
        )
    
    def forward(self, x):
        out = self.conv_layers(x)
        #! flatten:
        out = out.reshape(out.shape[0], -1)
        out = self.fcs(out)
        return out

    