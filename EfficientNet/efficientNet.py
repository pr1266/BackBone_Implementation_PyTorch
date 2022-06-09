import torch
import torch.nn as nn
import math

class PrintLayer(nn.Module):
    def __init__(self, type, index):
        super(PrintLayer, self).__init__()
        self.type = type
        self.index = index

    def forward(self, x):
        print(f'size of {self.type} layer {self.index} : {x.size()}')
        return x

