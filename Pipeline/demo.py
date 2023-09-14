import matplotlib.pyplot as plt
import torch
from scipy.stats import normaltest
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from Data.DataLoader import Complex_Volumes
from utility import pt_irange, arlo, hide_ticks
from StarMap.StarMap import load_starmap
from .pipeline import arlo_corrected, run_pipelines

def param_map_demo():
    vol = Complex_Volumes("train")[0].unsqueeze(0)

    b, n_t, d, h, w = vol.shape
    vol_t = pt_irange(0.005, 0.04, n_t).reshape(1, n_t, 1, 1 ,1).repeat(b, 1, d, h, w)

    starmap = load_starmap("./trained_models/StarMap500.pt")

    pipelines = [
        lambda : starmap(torch.cat((vol.real, vol.imag), 1)), 
        lambda : arlo_corrected(vol, vol_t),
    ]

    param_maps, _, _, _ = run_pipelines(pipelines, vol)

    starmap_param_map, arlo_param_map = param_maps

    fig, axs = plt.subplots(3, 5)

    ylabels = ["In Vivo Image", "StarMap T2* Map", "ARLO T2* Map"]

    for i, z in enumerate([3, 7, 11, 15, 19]):
        axs[0][i].imshow(vol[0, 0, z, :, :].abs(), cmap="gray")
        axs[1][i].imshow(1/starmap_param_map[0, 1, z, :, :].detach(), cmap="gray", vmin=0, vmax=0.06)
        axs[2][i].imshow(1/arlo_param_map[0, 1, z, :, :], cmap="gray", vmin=0, vmax=0.06)
        for j in range(3): 
            hide_ticks(axs[j][i])
            if i == 0: axs[j][i].set_ylabel(ylabels[j], fontsize=13)
        axs[2][i].set_xlabel(f"z = {z}", fontsize=13)
    
    plt.show()

def reconstructions():
    vol = Complex_Volumes("train")[0].unsqueeze(0)

    b, n_t, d, h, w = vol.shape
    vol_t = pt_irange(0.005, 0.04, n_t).reshape(1, n_t, 1, 1 ,1).repeat(b, 1, d, h, w)
    slice_idx = 20

    starmap = load_starmap("./trained_models/StarMap.pt")

    pipelines = [
        lambda : starmap(torch.cat((vol.real, vol.imag), 1)), 
        lambda : arlo_corrected(vol, vol_t),
    ]

    _, _, recons, _ = run_pipelines(pipelines, vol)

    starmap_recon, arlo_recon = recons

    fig, axs = plt.subplots(8, 3)
    xlabels = ["In Vivo Image", "StarMap", "ARLO"]

    for i in range(8):
        for j in range(3):
            hide_ticks(axs[i][j])
            axs[i][j].xaxis.set_label_position('top') 
            if j == 0: axs[i][j].set_ylabel(f"{round(0.005 + 0.005 * i, 3)}s", fontsize=10)
            if i == 0: axs[0][j].set_xlabel(xlabels[j], fontsize=13, position="top")
        axs[i][0].imshow(vol[0, i, slice_idx, :, :].abs().T, vmin=0, vmax=4.5, cmap="gray")
        axs[i][1].imshow(starmap_recon[0, i, slice_idx, :, :].abs().T.detach(), vmin=0, vmax=4.5, cmap="gray")
        axs[i][2].imshow(arlo_recon[0, i, slice_idx, :, :].abs().T, vmin=0, vmax=4.5, cmap="gray")

    plt.show()

def pixel_reconstructions():
    vol = Complex_Volumes("train")[10].unsqueeze(0)

    b, n_t, d, h, w = vol.shape
    t = pt_irange(0.005, 0.04, n_t)
    vol_t = t.reshape(1, n_t, 1, 1 ,1).repeat(b, 1, d, h, w)
    slice_idx = 20
    pts = [(x, y) for y in pt_irange(20, 80, 6).int() for x in pt_irange(40, 160, 6).int()]

    starmap = load_starmap("./trained_models/StarMap.pt")

    pipelines = [
        lambda : starmap(torch.cat((vol.real, vol.imag), 1)), 
        lambda : arlo_corrected(vol, vol_t),
    ]

    _, _, recons, _ = run_pipelines(pipelines, vol)

    starmap_recon, arlo_recon = recons

    fig = plt.figure()

    imax = fig.add_subplot(3, 1, 1)
    imax.imshow(vol[0, 0, slice_idx, :, :].abs().T, cmap="gray")
    imax.xaxis.set_tick_params(labelbottom=False)
    imax.yaxis.set_tick_params(labelleft=False)
    imax.set_xticks([])
    imax.set_yticks([])

    for i, (x, y) in enumerate(pts):

        starmap_recon_pt = starmap_recon[0, :, slice_idx, x, y]
        arlo_recon_pt = arlo_recon[0, :, slice_idx, x, y]

        idx = i + 19
        ax = fig.add_subplot(9, 6, idx)

        a, = ax.plot(t, starmap_recon_pt.abs().detach(), color="#20b2aa", label="StarMap Reconstruction")
        b, = ax.plot(t, arlo_recon_pt.abs(), color="gray", label="ARLO Reconstruction")
        c, = ax.plot(t, vol[0, :, slice_idx, x, y].abs(), "k.", markersize=2, label="Multi Echo MRI Signal")
        ax.set_ylim(0, 5)
        ax.annotate(f"{i + 1}", xy=(0.7, 0.7), xycoords='axes fraction')

        hide_ticks(ax)

        if idx == 49:
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.yaxis.set_tick_params(labelleft=True)
            ax.set_yticks([0, 1, 2, 3, 4, 5], ["0", "1", "2", "3", "4", "5"])
            ax.set_xticks([0, 0.02, 0.04])
            ax.set_xlabel("t (s)")
            ax.set_ylabel("MR Signal (AU)")

        imax.plot(x, y, 'r.')
    
    fig.legend(handles=[a, b, c], loc='lower center')

    plt.show()


def speed_and_residuals():
    vols = Complex_Volumes("train")
    n_vols = len(vols)
    vol = vols[0].unsqueeze(0)

    # for i in range(1, n_vols):
    #     vol = torch.cat((vol, vols[i].unsqueeze(0)), 0)

    b, n_t, d, h, w = vol.shape

    t = pt_irange(0.005, 0.04, n_t)
    vol_t = t.reshape(1, n_t, 1, 1 ,1).repeat(b, 1, d, h, w)

    starmap = load_starmap("./trained_models/StarMap.pt")

    pipelines = [
        lambda : starmap(torch.cat((vol.real, vol.imag), 1)), 
        lambda : arlo_corrected(vol.abs(), vol_t),
        lambda : arlo(vol.abs(), vol_t),
    ]

    _, durations, _, diffs = run_pipelines(pipelines, vol)

    starmap_duration,  arlo_duration,  un_arlo_duration  = durations
    starmap_diff,      arlo_diff,      un_arlo_diff      = diffs

    # rerieved from standard output of this function
    starmap_durations = torch.tensor([0.461, 0.469, 0.498, 0.524, 0.476, 0.582, 0.442, 0.564, 0.632, 0.613, 0.563])
    arlo_durations = torch.tensor([0.627, 0.638, 0.636, 0.650, 0.632, 0.968, 0.615, 0.760, 0.835, 0.849, 0.744])

    fig, ax = plt.subplots(1, 2)

    params = dict(
        x=["Arlo\n(Uncorrected)", "Arlo", "StarMap"], 
        align='center', color=["#eeeeee", "#20b2aa", "#20b2aa"], ecolor='k', capsize=10, width=0.5
    )

    ax[0].bar(
        height=[un_arlo_diff.mean().abs(), arlo_diff.mean().abs(), starmap_diff.mean().abs().detach()], 
        yerr=[un_arlo_diff.abs().var(), arlo_diff.abs().var(), starmap_diff.abs().var().detach()], 
        **params
    )
    ax[0].plot([1.1, 2], [0.18, 0.18], "k-")
    ax[0].plot(1.5, 0.19, "k*")
    
    ax[0].plot([0, 0.9], [0.18, 0.18], "k-")
    ax[0].plot(0.5, 0.19, "k*")
    
    ax[0].plot([0, 2], [0.21, 0.21], "k-")
    ax[0].plot(1, 0.22, "k*")

    ax[0].set_ylabel("Root Mean Squared Error (AU)")

    ax[1].bar( 
        height=[un_arlo_duration, arlo_durations.mean(), starmap_durations.mean()], 
        yerr=[0.01, arlo_durations.var(), starmap_durations.var()],
        **params
    )
    ax[1].plot([1.1, 2], [0.8, 0.8], "k-")
    ax[1].plot(1.5, 0.83, "k*")

    ax[1].plot([0, 0.9], [0.8, 0.8], "k-")
    ax[1].plot(0.5, 0.83, "k*")
    
    ax[1].plot([0, 2], [0.9, 0.9], "k-")
    ax[1].plot(1, 0.93, "k*")

    ax[1].set_ylabel("Duration (s)")

    t_tests = CompareMeans(
        DescrStatsW(starmap_diff.flatten().abs().detach()), 
        DescrStatsW(arlo_diff.flatten().abs().detach())
    ).ztest_ind(usevar="unequal")

    starmap_diff_n_tests = normaltest(starmap_diff.flatten().detach().abs())
    arlo_diff_n_tests = normaltest(arlo_diff.flatten().detach().abs())

    print(t_tests)
    print(starmap_diff_n_tests)
    print(arlo_diff_n_tests)
    print(starmap_duration)
    print(arlo_duration)
    print(un_arlo_duration)

    plt.show()