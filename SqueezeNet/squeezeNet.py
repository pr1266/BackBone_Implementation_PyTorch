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


class PrintLayer(nn.Module):
    def __init__(self, type='default', index='NaN'):
        super(PrintLayer, self).__init__()
        self.type = type
        self.index = index

    def forward(self, x):
        print(f'size of {self.type} layer {self.index} : {x.size()}')
        return x

class FireBlock(nn.Module):

    def __init__(self) -> None:
        super(FireBlock, self).__init__()

    def forward(self, x):
        pass