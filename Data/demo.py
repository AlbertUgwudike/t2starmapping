import torch
import matplotlib.pyplot as plt

from .DataLoader import Complex_Volumes, Voxel_Cube
from utility import pt_irange, hide_ticks, fmap

def distributions():
    dataset = Complex_Volumes("train")

    vol = dataset[0].unsqueeze(1).abs()
    
    for idx in range(1, len(dataset)): vol = torch.cat((vol, dataset[idx].unsqueeze(1).abs()), 1)

    hist = vol.log().histogram(bins=200)
    
    _, ax = plt.subplots(1, 1)

    ax.plot(hist[1][:-1], hist[0])

    print(vol.std())

    plt.show()


def present_voxel_cubes():
    n_t = 100
    t  = pt_irange(0, 0.04, n_t)
    
    delta_omega = Voxel_Cube(1000)[550][1].unsqueeze(0).unsqueeze(0)
    delta_omegas = fmap(lambda n : n * delta_omega, [0, 20, 60])

    _, axs = plt.subplots(3, 3)

    for i, j in [(x, y) for x in range(3) for y in range(3)]:
        axs[i][j].imshow(delta_omegas[i][0, 0, j, :, :] - delta_omegas[i].mean(), vmin=-1200, vmax=1500)
        hide_ticks(axs[i][j])

    plt.show()