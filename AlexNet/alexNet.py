import torch
import torch.nn as nn
import os

class PrintLayer(nn.Module):
    def __init__(self, type, index):
        super(PrintLayer, self).__init__()
        self.type = type
        self.index = index

    def forward(self, x):
        print(f'size of {self.type} layer {self.index} : {x.size()}')
        return x

#! alexNet chizi nadare mese leNet e yekam pichide tar:
class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 2):
        super(AlexNet, self).__init__()

        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            PrintLayer(type='conv', index=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            PrintLayer(type='pool', index=1),
            nn.Conv2d(64, 192, kernel_size=(5, 5), padding=(2, 2)),
            PrintLayer(type='conv', index=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            PrintLayer(type='pool', index=2),
            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
            PrintLayer(type='conv', index=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            PrintLayer(type='conv', index=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            PrintLayer(type='conv', index=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            PrintLayer(type='pool', index=3),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            PrintLayer(type='linear', index=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            PrintLayer(type='linear', index=2),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),
            PrintLayer(type='linear', index=3),
        )

    def forward(self, x):
        print(f'input size : {x.size()}')
        x = self.convolutional(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return torch.softmax(x, 1)

def Test():
    os.system('cls')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AlexNet().to(device)
    x = torch.randn(20, 3, 224, 224).to(device)
    model(x)

Test()