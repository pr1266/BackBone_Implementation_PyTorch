import torch
import  torch.nn as nn

#! agha leNet tasvir 32*32 migire
#! too laye conv aval: 28 * 28 * 6 (kernel size 5, padding 0, stride 1)
#! baad average pooling  layer ba kernel size 2 va stride 2
#! ke size ro mikone 14 * 14 * 6 (too pooling layer tedad channel ha sabete)
#! conv 2vom ein e conv aval kernel size 5, padding 0 va stride 1
#! ke size ro mikone 10 * 10 * 16
#! dobare avg pooling zade shode 5 * 5 * 16
#! baad conv zade 120 ta channel dare 1 * 1 baad flatten mizane o miad
#! roo FC va tamam. kheili simple :)

class leNet(nn.Module):

    def __init__(self):
        super(leNet, self).__init__()
        