import torch.nn as nn
import torch

'''
    Input: [R2, B0_x, B0_y, t]
    Output: [real, imag]
'''
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(27, 20),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Linear(20, 15),
            nn.Dropout(0.1, inplace=True),
            nn.Linear(15, 10),
            nn.Dropout(0.1, inplace=True),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Linear(10, 8),
            nn.LeakyReLU(0.3, inplace=True)
        )
    
    def forward(self, B0_map):
        return self.model(B0_map)
