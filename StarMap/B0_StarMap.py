import torch.nn as nn
import torch
from functools import reduce

class UNET(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.M0_model           = sub_model()
        self.R2_model           = sub_model()
        self.B0_offset_model    = sub_model()
        self.B0_x_model         = sub_model()
        self.B0_y_model         = sub_model()


    def forward(self, img):
        M0          = self.M0_model(img)
        R2_star     = self.R2_model(img)
        B0_offset   = self.B0_offset_model(img)
        B0_x        = self.B0_x_model(img)
        B0_y        = self.B0_y_model(img)

        return torch.cat((M0, R2_star, B0_offset, B0_x, B0_y), 1)
    

def sub_model():
    return nn.Sequential(
        SkipConnection(
            nn.Sequential(
                nn.Conv2d(16, 5, 3, padding="same"),
                nn.BatchNorm2d(5),
                nn.ReLU(),
                nn.Conv2d(5, 1, 3, padding="same"),
                nn.BatchNorm2d(1),
                nn.ReLU()
            )
        ),
        nn.Conv2d(17, 1, 1, padding="same")
    )

class SkipConnection(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X):
        return torch.cat((X, self.model(X)), 1)
    