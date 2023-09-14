import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time

from utility import pt_irange
from Simulator.pt_simulator import pt_signal
from Data.DataLoader import B0_Voxels

# Visually compare the output of a trained MLP with
# the 'ground-truth' output of the simulator


# --------- parameters -------- #
# mlp:  Trained instance of MLP class

def demo_mlp(mlp):
    mlp.eval()

    B0_env = B0_Voxels(1000)[220].unsqueeze(0)
    print(B0_env)
    n_t = 8

    # -------------  Test MLP -------------- #
    t = pt_irange(0.005, 0.04, n_t).unsqueeze(1)

    model_start = time()
    mlp_sig = mlp(B0_env).data.cpu().squeeze()
    elapsed = round((time() - model_start) * 1000)

    print(f"MLP executed in {elapsed}ms")
    # -------------------------------------- #



    # ----------  Test Simulator ----------- #
    simulator_params = {
        "R2_star": torch.tensor([0]),
        "B0_env": 20 * B0_env,
    }

    simulator_start = time()
    sim_sig = pt_signal(**simulator_params)
    elapsed = round((time() - simulator_start) * 1000)

    print(f"Simulator executed in {elapsed}ms")
    # -------------------------------------- #


    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(5)
    fig.set_figwidth(15)

    for i in range(1):

        mse = (torch.sum((mlp_sig - sim_sig.abs()) ** 2) / n_t).item()
        print(mse)

        ax.set_ylabel("MR Signal (Arbitrary Units)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0.5, 1.1)
        # ax[i].plot(t, imag_sim_signal, c="red", label="MLP T2* decay")

        ax.plot(t.squeeze(), mlp_sig.squeeze(), c="blue", label="MLP T2* decay")
        ax.plot(t.squeeze(), sim_sig.abs().squeeze(), c="green", label="Sim T2* decay")
        ax.legend()

        print(mlp_sig)
        print(sim_sig.abs())


    plt.show()