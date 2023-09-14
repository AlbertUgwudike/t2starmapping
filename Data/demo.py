import torch

from .DataLoader import Complex_Volumes
import matplotlib.pyplot as plt

def distributions():
    dataset = Complex_Volumes("train")

    vol = dataset[0].unsqueeze(1)
    
    # for idx in range(1, len(dataset)): torch.cat((vol, dataset[idx].unsqueeze(1)), 1)

    real_hist = vol.real.histogram(bins=200)
    imag_hist = vol.imag.histogram(bins=200)
    
    _, ax = plt.subplots(1, 2)

    ax[0].plot(real_hist[1][:-1], real_hist[0])
    ax[1].plot(imag_hist[1][:-1], imag_hist[0])

    print(vol.real.pow(2).mean().sqrt(), vol.imag.pow(2).mean().sqrt())

    plt.show()