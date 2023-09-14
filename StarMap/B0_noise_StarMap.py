import torch.nn as nn
import torch
from functools import reduce

class UNET(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.M0_model           = sub_model()
        self.M1_model           = sub_model()
        self.R2_model           = sub_model()
        self.B0_offset_model    = sub_model()
        self.B0_grad            = sub_model()
        self.noise_model        = sub_model()


    def forward(self, img):
        M0          = self.M0_model(img)
        M1          = self.M1_model(img)
        R2_star     = self.R2_model(img)
        B0_offset   = self.B0_offset_model(img)
        B0_grad     = self.B0_grad(img)
        noise       = self.noise_model(img)

        return torch.cat((M0, M1, R2_star, B0_offset, B0_grad, noise), 1)
    

def sub_model(first = 10, middle = 5, last = 1):
    return nn.Sequential(
        SkipConnection(
            nn.Sequential(
                nn.Conv3d(16, first, 3, padding="same"),
                nn.BatchNorm3d(first),
                nn.ReLU(),
                #nn.Dropout3d(0.3),
                nn.Conv3d(first, middle, 3, padding="same"),
                nn.BatchNorm3d(middle),
                nn.ReLU(),
                #nn.Dropout3d(0.3),
                nn.Conv3d(middle, last, 3, padding="same"),
                nn.BatchNorm3d(last),
                nn.ReLU()
            )
        ),
        nn.Conv3d(16 + last, last, 1, padding="same")
    )

class SkipConnection(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X):
        return torch.cat((X, self.model(X)), 1)
    