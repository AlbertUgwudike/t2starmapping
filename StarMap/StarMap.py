import torch.nn as nn
import torch

class StarMap(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.M0_model           = sub_model()
        self.R2_model           = sub_model()


    def forward(self, img):
        M0          = self.M0_model(img)
        R2_star     = self.R2_model(img)

        return torch.cat((M0, R2_star), 1)
    

def sub_model(first = 10, middle = 5, last = 1):
    return nn.Sequential(
        SkipConnection(
            nn.Sequential(
                nn.Conv3d(16, first, 3, padding="same"),
                nn.BatchNorm3d(first),
                nn.ReLU(),
                nn.Conv3d(first, middle, 3, padding="same"),
                nn.BatchNorm3d(middle),
                nn.ReLU(),
                nn.Conv3d(middle, middle, 3, padding="same"),
                nn.BatchNorm3d(middle),
                nn.ReLU(),
                nn.Conv3d(middle, last, 3, padding="same"),
                nn.BatchNorm3d(last),
                nn.ReLU()
            )
        ),
        nn.Conv3d(16 + last, last, 1, padding="same"),
        nn.BatchNorm3d(last),
        nn.ReLU()
    )

class SkipConnection(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X):
        return torch.cat((X, self.model(X)), 1)
    
def load_starmap(fname):
    starmap = StarMap()
    starmap.load_state_dict(torch.load(fname, map_location=torch.device('cpu')))
    starmap.eval()
    return starmap
    