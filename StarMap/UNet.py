import torch.nn as nn
import torch
from functools import reduce

class UNET(nn.Module):

    def __init__(self):
        super().__init__()
        core = reduce(nest, [64, 32], conv(128, 128))

        # original
        # self.model = nn.Sequential(conv(8, 32), core, conv(64, 3))

        # high res
        self.model = nn.Sequential(conv(16, 32), core, nn.Conv2d(32, 5, 1, padding="same"))


    def forward(self, img):
        return self.model(img)
    

def nest(mdl, n): 
    return nn.Sequential(
        encode(n), 
        SkipConnection(nn.Sequential(downscale(), mdl, upscale(n))),
        decode(n)
    )

def conv(n_in, n_out, kernel_size=3): 
    return nn.Sequential(
        nn.Conv2d(n_in, n_out, kernel_size, padding="same"),
        nn.BatchNorm2d(n_out),
        nn.ReLU()
    )

def encode(n): return nn.Sequential(conv(n, 2 * n), conv(2 * n,  2 * n))
def decode(n): return nn.Sequential(conv(4 * n,  2 * n), conv(2 * n, n))
def downscale(): return nn.MaxPool2d(2)
def upscale(n): return nn.ConvTranspose2d(2 * n, 2 * n, 2, stride=2)
#def upscale(n): return nn.UpsamplingBilinear2d(scale_factor=2)

class SkipConnection(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X):
        X_out = self.model(X)
        # dx = (X.shape[2] - X_out.shape[2]) // 2
        # dy = (X.shape[3] - X_out.shape[3]) // 2
        return torch.cat((X, X_out), 1)
    
        