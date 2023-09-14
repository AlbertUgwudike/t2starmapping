import torch
from operator import __mul__
import numpy as np

from utility import pt_irange

def estimate_delta_omega(vol_):
    vol = vol_.permute(1, 0, 2, 3, 4)
    t = pt_irange(0.005, 0.04, 8)
    angles = vol.angle().detach().cpu()
    unwrapped = np.unwrap(angles, axis=0).reshape(8, np.prod(angles.shape[1:]))
    B0_offset, init_phase = np.polyfit(t[:4], unwrapped[:4, :], 1)
    return (
        torch.tensor(B0_offset.reshape(angles.shape[1:]), dtype=torch.float).unsqueeze(1), 
        torch.tensor(init_phase.reshape(angles.shape[1:]), dtype=torch.float).unsqueeze(1)
    )



