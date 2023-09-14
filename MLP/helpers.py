import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from time import time

from Data.DataLoader import SimulatedData
from .model import MLP
from Simulator.pt_simulator import pt_signal
from utility import pt_irange

def load_mlp(file_name):
    simulated_dataset = SimulatedData('./data/mlp_training_data_100_000.csv')
    mlp = MLP(input_mean=simulated_dataset.x_mean(), input_std=simulated_dataset.x_std())
    mlp.load_state_dict(torch.load(file_name))
    return mlp

def apply_to_image(mlp, original, param_map):
    mlp.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original = original.to(device)
    
    batch_size = param_map.shape[0]
    h = param_map.shape[2]
    w = param_map.shape[3]

    R2_star = param_map[:, 0, :, :].flatten().split(h)
    B0_x = param_map[:, 1, :, :].flatten().split(h)
    B0_y = param_map[:, 2, :, :].flatten().split(h)
    t = pt_irange(0.005, 0.04, 8)
    
    reals = []
    imags = []

    for i in range(w):
        s = pt_signal(R2_star=R2_star[i], B0_x=B0_x[i], B0_y=B0_y[i], t=t, n_isochromats=10_000)
        reals.append(s["real"])
        imags.append(s["imag"])

    real = torch.cat(reals, 0).T.reshape((batch_size, 8, h, w))
    imag = torch.cat(imags, 0).T.reshape((batch_size, 8, h, w))

    sim_mag = ((real[:, 0, :, :] ** 2 + imag[:, 0, :, :] ** 2) ** 0.5)
    ori_mag = ((original[:, 0, :, :] ** 2 + original[:, 8, :, :] ** 2) ** 0.5)

    M0 = (ori_mag / sim_mag).unsqueeze(1)

    return torch.cat((M0 * real, M0 * imag), 1)

def main():
    h = 256
    w = 256
    param_map = torch.tensor([400, 200, 200], dtype=torch.float32).reshape(1, 3, 1, 1).repeat(1, 1, h, w)
    original = torch.ones(1, 16, h, w)
    mlp = load_mlp("./models/mlp.pt")

    start = time()
    sig = apply_to_image(mlp, original, param_map)
    print(f"Elapsed: {time() - start}")
    print(sig[0, :8, 1, 1])

    # plt.plot(sig[0, :8, 1, 1].detach())
    # plt.show()


if __name__ == "__main__": main()

def apply_mlp_to_image(mlp, original, param_map):
    mlp.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = param_map.shape[0]
    h = param_map.shape[2]
    w = param_map.shape[3]

    R2_star = param_map[:, 0, :, :].unsqueeze(1).repeat(1, 8, 1, 1).flatten().unsqueeze(1)
    B0_x = param_map[:, 1, :, :].unsqueeze(1).repeat(1, 8, 1, 1).flatten().unsqueeze(1)
    B0_y = param_map[:, 2, :, :].unsqueeze(1).repeat(1, 8, 1, 1).flatten().unsqueeze(1)
    t = pt_irange(0.005, 0.04, 8).reshape(1, 8, 1, 1).repeat(batch_size, 1, h, w).flatten().unsqueeze(1).to(device)
    model_params = torch.cat((R2_star, B0_x, B0_y, t), 1)
    
    sig = mlp(model_params)

    real = sig[:, 0].reshape((batch_size, 8, h, w))
    imag = sig[:, 1].reshape((batch_size, 8, h, w))

    sim_mag = ((real[:, 0, :, :] ** 2 + imag[:, 0, :, :] ** 2) ** 0.5)
    ori_mag = ((original[:, 0, :, :] ** 2 + original[:, 8, :, :] ** 2) ** 0.5)

    M0 = (ori_mag / sim_mag).unsqueeze(1)

    return torch.cat((M0 * real, M0 * imag), 1)
