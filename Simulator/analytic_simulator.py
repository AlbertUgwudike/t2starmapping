import torch
from torch.nn.functional import conv3d
from functools import reduce
from operator import __mul__

from utility import pt_irange, ep
from .filters import sobel3D

# Analytic MR Signal Simulator

def sinc(B0_grad, B0_offset, t):
    a = 1/3
    return torch.exp(1j * B0_offset * t) * torch.sin(a * B0_grad * t) / (a * B0_grad * t)

def sinc_3d(B0_offset, B0_x, B0_y, B0_z, vol_t):
    return reduce(__mul__, [ sinc(grad + ep, B0_offset / 3, vol_t) for grad in [B0_x, B0_y, B0_z] ])

def simulate_volume(param_map, B0_offset_map, init_phase_map, t=pt_irange(0.005, 0.04, 8), device = 'cpu'):

    b, _, d, h, w = param_map.shape
    vol_t = t.reshape(1, t.shape[0], 1, 1, 1).repeat(b, 1, d, h, w).to(device)

    R2_star     = param_map[:, 1:2, :, :, :]

    B0_offset_map   = B0_offset_map.to(device)
    grads = differentiate_3d(B0_offset_map, sobel3D, "same", device)

    init_phase  = ((init_phase_map.to(device) + torch.pi) % (2 * torch.pi)) - torch.pi
    M0_         = param_map[:, 0:1, :, :, :]
    M0          = M0_ * torch.polar(torch.ones(M0_.shape).to(device) , init_phase)

    return M0 * torch.exp(-R2_star * vol_t) * sinc_3d(B0_offset_map, *grads, vol_t)


def differentiate_3d(vol, kernel = sobel3D, padding = "same", device = 'cpu'):

    vol = vol.to(device)

    sobel_z = kernel.reshape(1, 1, 3, 3, 3).to(device)
    sobel_x = kernel.reshape(1, 1, 3, 3, 3).permute(0, 1, 3, 4, 2).to(device)
    sobel_y = kernel.reshape(1, 1, 3, 3, 3).permute(0, 1, 4, 2, 3).to(device)

    dx = conv3d(vol, sobel_x, padding=padding)
    dy = conv3d(vol, sobel_y, padding=padding)
    dz = conv3d(vol, sobel_z, padding=padding)

    return dx + ep, dy + ep, dz + ep

