import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Takes a 3 channel input image
class FSRCNN(nn.Module):
    def __init__(self, d, s, m):
        super(FSRCNN, self).__init__()

        self.prelu_extract = nn.PReLU()
        self.prelu_shrink = nn.PReLU()
        self.prelu_expand = nn.PReLU()

        self.feature_extraction = nn.Conv2d(3, d, 5)
        self.shrinking = nn.Conv2d(d, s, 1)
        self.mapping = nn.Sequential(*[nn.Sequential(nn.Conv2d(s, s, 3), nn.PReLU()) for _ in range(m)])
        self.expanding = nn.Conv2d(s, d, 1)
        self.deconv = nn.ConvTranspose2d(d,3,9) # This is the part which will be changed for different super HR images.
    
    def forward(self, x):
        x = self.prelu_extract(self.feature_extraction(x))
        x = self.prelu_shrink(self.shrinking(x))
        x = self.mapping(x)
        x = self.prelu_expand(self.expanding(x))
        x = self.deconv(x)
        return x