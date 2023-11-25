import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from scipy.stats import normaltest
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from Data.DataLoader import Complex_Volumes
from utility import pt_irange, arlo, hide_ticks, fmap
from StarMap.StarMap import load_starmap
from StarMap.FlatMap import load_flatmap
from .pipeline import arlo_corrected, run_pipelines

def param_map_demo():
    vol = Complex_Volumes("test", cropped=True)[1].unsqueeze(0)

    b, n_t, d, h, w = vol.shape
    vol_t = pt_irange(0.005, 0.04, n_t).reshape(1, n_t, 1, 1 ,1).repeat(b, 1, d, h, w)

    starmap = load_starmap("./trained_models/StarMap.pt")

    pipelines = [
        lambda : arlo(vol.abs(), vol_t),
        lambda : arlo_corrected(vol, vol_t),
        lambda : starmap(torch.cat((vol.real, vol.imag), 1)), 
    ]

    param_maps, _, _, _ = run_pipelines(pipelines, vol)

    arlo_param_map, arlo_corr_param_map, starmap_param_map = param_maps


    region_intensities = fmap(
        lambda v: (1/v[0, 1, 11:16, 80:101, 33:54]), 
        [arlo_param_map, arlo_corr_param_map, starmap_param_map.detach()]
    )

    titles = ["ARLO\n(Uncorrected)", "ARLO\n(Corrected)", "CNN"]

    fig, axs = plt.subplots(1, 3)

    fig.text(0.5, 0.01, 'T2* (ms)', ha='center', fontsize=13)

    for i, t in enumerate(region_intensities): 
        axs[i].hist(1000 * t.flatten(), 75, range = (0, 100), density=True, color="#20b2aa")
        axs[i].set_ylim(0, 0.12)
        axs[i].set_title(titles[i])
        meanline = axs[i].axvline(1000 * t.mean(), linestyle="--", color="black", label = "Mean")
        medianline = axs[i].axvline(1000 * t.median(), linestyle="dotted", color="black", label = "Median")
        if (i == 0): axs[i].set_ylabel("Frequency Density", fontsize=13)
        else: 
            axs[i].yaxis.set_tick_params(labelleft=False)
            axs[i].set_yticks([])

        print(t.mean())

    fig.legend(handles=[meanline, medianline], loc=(0.52, 0.6))
    plt.show()

    exit()

    # fig, axs = plt.subplots(2, 1)
    # a = axs[0].imshow(vol[0, 0, 13, :, :].abs().T, cmap="gray", vmin=0, vmax=10)
    # b = axs[1].imshow(1/starmap_param_map[0, 1, 13, :, :].detach().T, cmap="gray", vmin=0, vmax=0.06)
    # fmap(hide_ticks, axs)
    # c_bar = plt.colorbar(b, ax=axs[1], ticks=[0, 0.06])
    # c_bar.set_ticklabels([0, 60])
    # c_bar.set_label("T2* (ms)")
    # plt.show()
    # exit()

    fig, axs = plt.subplots(5, 4)

    xlabels = ["First echo", "ARLO\n(Uncorrected)", "ARLO\n(Corrected)", "CNN"]

    
    for i, z in enumerate([11, 12, 13, 14, 15]):
        a = axs[i][0].imshow(vol[0, 0, z, :, :].abs().T, cmap="gray", vmin=0, vmax=10)
        axs[i][1].imshow(1/arlo_param_map[0, 1, z, :, :].T, cmap="gray", vmin=0, vmax=0.1)
        b = axs[i][2].imshow(1/arlo_corr_param_map[0, 1, z, :, :].T, cmap="gray", vmin=0, vmax=0.1)
        axs[i][3].imshow(1/starmap_param_map[0, 1, z, :, :].detach().T, cmap="gray", vmin=0, vmax=0.1)

        for j in range(4): 
            hide_ticks(axs[i][j])
            if i == 4: axs[i][j].set_xlabel(xlabels[j], fontsize=13)

    for i in range(4):
        rect = Rectangle((80, 33), width=20, height=20, facecolor='none', edgecolor='red')
        axs[0][i].add_patch(rect)

    
    cax = fig.add_axes([0.33, 0.92, 0.56, 0.02], transform=axs[0, 2].transData)
    c_bar = plt.colorbar(b, ax=axs[0][2], cax = cax, orientation = "horizontal", ticks=[0, 0.025, 0.05, 0.075, 0.1])
    c_bar.ax.set_xticklabels([0, 25, 50, 75, 100]) 
    c_bar.ax.set_title("T2* (ms)")

    cax = fig.add_axes([0.13, 0.92, 0.16, 0.02], transform=axs[0, 0].transData)
    c_bar = plt.colorbar(a, ax=axs[0][0], cax = cax, orientation = "horizontal", ticks=[0, 10])
    c_bar.ax.set_xticklabels([0, 10]) 
    c_bar.ax.set_title("First Echo (AU)")
    
    plt.show()

    

def reconstructions():
    vol = Complex_Volumes("test", cropped=True)[0].unsqueeze(0)

    b, n_t, d, h, w = vol.shape
    vol_t = pt_irange(0.005, 0.04, n_t).reshape(1, n_t, 1, 1 ,1).repeat(b, 1, d, h, w)
    slice_idx = 20

    #starmap = load_flatmap("./trained_models/flatmap1.pt")
    starmap = load_starmap("./trained_models/starmap1.pt")

    pipelines = [
        lambda : starmap(torch.cat((vol.real, vol.imag), 1)), 
        lambda : arlo_corrected(vol, vol_t),
    ]

    _, _, recons, _ = run_pipelines(pipelines, vol)

    starmap_recon, arlo_recon = recons

    fig, axs = plt.subplots(3, 3)
    xlabels = ["In Vivo Image", "StarMap", "ARLO"]

    for i in range(3):
        for j in range(3):
            hide_ticks(axs[i][j])
            axs[i][j].xaxis.set_label_position('top') 
            if j == 0: axs[i][j].set_ylabel(f"{round(0.005 + 0.005 * i, 3)}s", fontsize=10)
            if i == 0: axs[0][j].set_xlabel(xlabels[j], fontsize=13, position="top")
        axs[i][0].imshow(vol[0, i, slice_idx, :, :].abs().T, vmin=0, vmax=0.0001, cmap="gray")
        axs[i][1].imshow(starmap_recon[0, i, slice_idx, :, :].abs().T.detach(), cmap="gray")
        axs[i][2].imshow(arlo_recon[0, i, slice_idx, :, :].abs().T, cmap="gray")

        # axs[i][0].imshow(vol[0, i, slice_idx, :, :].abs().T, vmin=0, vmax=1, cmap="gray")
        # axs[i][1].imshow(starmap_recon[0, i, slice_idx, :, :].abs().T.detach(), vmin=0, vmax=1, cmap="gray")
        # axs[i][2].imshow(arlo_recon[0, i, slice_idx, :, :].abs().T, vmin=0, vmax=1, cmap="gray")

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
    vols = Complex_Volumes("test", cropped=True)
    n_vols = 7
    vol = vols[0].unsqueeze(0)

    for i in range(1, n_vols):
        vol = torch.cat((vol, vols[i].unsqueeze(0)), 0)

    b, n_t, d, h, w = vol.shape

    t = pt_irange(0.005, 0.04, n_t)
    vol_t = t.reshape(1, n_t, 1, 1 ,1).repeat(b, 1, d, h, w)

    starmap = load_starmap("./trained_models/StarMap.pt")

    pipelines = [
        lambda : arlo(vol.abs(), vol_t), 
        lambda : arlo_corrected(vol, vol_t),
        lambda : starmap(torch.cat((vol.real, vol.imag), 1)),
    ]

    _, durations, _, diffs = run_pipelines(pipelines, vol)

    un_arlo_duration,  arlo_duration,  starmap_duration, = durations
    un_arlo_diff,      arlo_diff,      starmap_diff      = diffs

    # rerieved from standard output of this function
    starmap_durations = torch.tensor([0.461, 0.469, 0.498, 0.524, 0.476, 0.582, 0.442, 0.564, 0.632, 0.613, 0.563])
    arlo_durations = torch.tensor([0.627, 0.638, 0.636, 0.650, 0.632, 0.968, 0.615, 0.760, 0.835, 0.849, 0.744])

    fig, ax = plt.subplots(1, 2)

    params = dict(
        x=["ARLO\n(Uncorrected)", "ARLO\n(Corrected)", "CNN"], 
        align='center', color=["#20b2aa", "#20b2aa", "#20b2aa"], ecolor='k', capsize=10, width=0.5
    )

    ax[0].bar(
        height=[un_arlo_diff.mean().abs(), arlo_diff.mean().abs(), starmap_diff.mean().abs().detach()], 
        # yerr=[0.2306 ** 2, 0.2254 ** 2, 0.2529 ** 2], 
        **params
    )

    ax[0].set_ylabel("RMSE (Arbitrary Units)")

    ax[1].bar( 
        height=[1000 * un_arlo_duration, 1000 * arlo_durations.mean(), 1000 * starmap_durations.mean()], 
        # yerr=[0.01, arlo_durations.var(), starmap_durations.var()],
        **params
    )

    ax[1].set_ylabel("Duration (ms)")

    t_tests = CompareMeans(
        DescrStatsW(starmap_diff.flatten().abs().detach()), 
        DescrStatsW(arlo_diff.flatten().abs().detach())
    ).ztest_ind(usevar="unequal")

    plt.show()

def single_pixel_reconstructions():
    vol = Complex_Volumes("train")[0].unsqueeze(0)

    b, n_t, d, h, w = vol.shape
    vol_t = pt_irange(0.005, 0.04, n_t).reshape(1, n_t, 1, 1 ,1).repeat(b, 1, d, h, w)
    slice_idx = 20

    starmap = load_starmap("./trained_models/starmap1.pt")

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

def single_pixel_reconstruction():
    vol = Complex_Volumes("train")[10].unsqueeze(0)

    b, n_t, d, h, w = vol.shape
    t = pt_irange(0.005, 0.04, n_t)
    vol_t = t.reshape(1, n_t, 1, 1 ,1).repeat(b, 1, d, h, w)
    slice_idx = 20
    x, y = 137, 82
    pipelines = [ lambda : arlo_corrected(vol, vol_t) ]

    _, _, recons, _ = run_pipelines(pipelines, vol)

    arlo_recon, = recons

    fig, ax = plt.subplots()

    arlo_recon_pt = arlo_recon[0, :, slice_idx, x, y]

    b, = ax.plot(t, arlo_recon_pt.abs(), color="#20b2aa", label="ARLO Reconstruction")
    c = ax.scatter(t, vol[0, :, slice_idx, x, y].abs(), color="#20b2aa", marker="o", label="Multi Echo MRI Signal")
    ax.set_ylim(0, 3.5)

    hide_ticks(ax)

    ax.xaxis.set_tick_params(labelbottom=True)
    ax.yaxis.set_tick_params(labelleft=True)
    ax.set_yticks([0, 0.7, 1.4, 2.1, 2.8, 3.5], ["0", "0.7", "1.4", "2.1", "2.8", "3.5"])
    ax.set_xticks([0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04])
    ax.set_xlabel("t (s)")
    ax.set_ylabel("MR Signal (AU)")
    
    # fig.legend(handles=[b, c], loc='lower center')

    plt.show()