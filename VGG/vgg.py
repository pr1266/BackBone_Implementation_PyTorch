import torch
import torch.nn as nn
import os
#! khob hajimoon VGG
#! yeki az maroof tarin backbone haye transfer learning

VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M",],
    "VGG19": [ 64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M",],
}

#! yeki az moshkelat e Sequential API ha ine ke nemitooni output ro moshakhas koni
#! hala ye Custom Layer Dorost mikonim ta betoonim shape ro print konim:
class PrintLayer(nn.Module):
    def __init__(self, type, index):
        super(PrintLayer, self).__init__()
        self.type = type
        self.index = index

    def forward(self, x):
        print(f'size of {self.type} layer {self.index} : {x.size()}')
        return x

class VGG(nn.Module):

    def __init__(self, type='VGG16'):
        super(VGG, self).__init__()
        self.num_classes = 1000
        self.in_channels = 3
        self.type = type
        self.conv_layers = self.create_conv_layers(VGG_types[self.type])
        fc_index = 1
        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            PrintLayer(type='dense', index=1),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            PrintLayer(type='dense', index=2),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.num_classes),
            PrintLayer(type='dense', index=3)
        )
    
    def forward(self, x):
        out = self.conv_layers(x)
        #! flatten:
        out = out.reshape(out.shape[0], -1)
        out = self.fcs(out)
        return out
    
    def create_conv_layers(self, architecture):
        #! inja bar asas e cfg miaim architecture ro misazim
        #! moshabehesh ro too yolo dashtim:
        layers = []
        in_channels = self.in_channels
        conv_index = 1
        max_pool_index = 1
        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    ),
                    PrintLayer(type='conv', index=conv_index),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                conv_index += 1
                in_channels = x

            elif x == "M":
                
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), PrintLayer(type='max pool', index=max_pool_index)]
                max_pool_index += 1
        return nn.Sequential(*layers)


if __name__ == "__main__":
    os.system('cls')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'used device : {device}')
    model = VGG().to(device)
    x = torch.randn(20, 3, 224, 224).to(device)
    print(model(x).shape)