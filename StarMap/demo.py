import matplotlib.pyplot as plt
import torch
import pandas as pd
from hampel import hampel

from StarMap.StarMap import load_starmap
from Data.DataLoader import Complex_Volumes
from utility import pt_irange
from Simulator.analytic_simulator import simulate_volume
from FieldEstimator.field_estimator import estimate_delta_omega


def loss_curve():
    losses = pd.read_csv("./losses/unet1_loss.csv").iloc[:, 1:].values
    d = hampel(pd.Series(losses[:, 0]), window_size=10, n=1, imputation=True)
    print(d)
    plt.plot(d)
    plt.plot(losses[:, 1])
    plt.show()


def pixel_demo_big():
    starmap = load_starmap("./trained_models/starmap1.pt")
    slice_idx = 15
    vol = Complex_Volumes()[3].unsqueeze(0)
    param_map = starmap(torch.cat((vol.real, vol.imag), 1))
    B0_offset, initial_phase = estimate_delta_omega(vol)
    sim_vol = simulate_volume(param_map, B0_offset, initial_phase)
    t = pt_irange(0.005, 0.04, 8)
    points = [(81, 15), (76, 83), (150, 40), (142, 63)]

    _, ax = plt.subplots(2, 2)

    for i, (x, y) in enumerate(points):
        ax[i // 2][i % 2].set_ylim(0, 0.00012)
        # ax[i // 2][i % 2].plot(t, sim_vol[0, :, slice_idx, x, y].abs().detach(), c='green')
        ax[i // 2][i % 2].plot(t, vol[0, :, slice_idx, x, y].abs().detach(), c='red')
    
    plt.show()

