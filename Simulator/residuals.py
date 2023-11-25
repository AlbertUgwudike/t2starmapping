import torch
from time import time 
import pandas as pd

from utility import pt_irange, NLLS
from .numerical_simulator import numerical_simulator
from .analytic_simulator import differentiate_3d, sinc_3d
from .interpolator import polynomial_interpolator
from Data.DataLoader import Voxel_Cube

def residuals_of_interpolation(vsf_func):

    n_t = 8
    t  = pt_irange(0.005, 0.04, n_t)
    dataset = Voxel_Cube(1000)
    residuals = []

    start = time()

    for idx in range(0, 1000, 2):
        cube, B0_env = dataset[idx]
        cube_vsf = vsf_func(B0_env)
        cube_corr = cube[:, 1, 1, 1] #/ cube_vsf[0, :]
        M0, R2 = NLLS(cube_corr.abs(), t)
        # print((M0 * torch.exp(-R2 * t) - cube_corr.abs()).shape)
        # exit()
        residuals.append((M0 * torch.exp(-R2 * t) - cube_corr.abs()).pow(2).mean().data.item())
        # residuals.append((M0 * torch.exp(-R2 * t) - cube[:, 1, 1, 1].abs()).pow(2).mean().data.item())

        if idx % 100 == 0: print(f"Voxel {idx} complete!")

    return residuals, time() - start


def simulator_residuals():
    for degree in range(1, 6):
        vsf_func = lambda B0_env: numerical_simulator(
            delta_omega_env=B0_env.unsqueeze(0).unsqueeze(0), interpolator = polynomial_interpolator(degree)
        )
        residuals, elapsed = residuals_of_interpolation(vsf_func)
        df = pd.DataFrame(data=residuals, columns=[f"Residuals"])
        df.to_csv(f"../data/simulation_residuals/Degree_{degree}_Residuals.csv")
        print(f"Degree {degree} complete in {elapsed}s!")

def analytic_residuals():
    t = pt_irange(0.005, 0.04, 8).reshape(1, 8, 1, 1, 1).repeat(1, 1, 3, 3, 3)
    vsf_func = lambda B0_env: sinc_3d(2* B0_env.unsqueeze(0).unsqueeze(0), *differentiate_3d(0.5 * B0_env.unsqueeze(0).unsqueeze(0)), t)[:, :, 1, 1, 1]
    residuals, elapsed = residuals_of_interpolation(vsf_func)
    df = pd.DataFrame(data=residuals, columns=[f"Residuals"])
    df.to_csv(f"../data/simulation_residuals/Bogus_Residuals.csv")
    print(f"Analytics complete in {elapsed}s!")