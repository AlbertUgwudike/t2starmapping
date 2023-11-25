import matplotlib.pyplot as plt
import torch
import numpy as np

from .field_estimator import estimate_delta_omega
from utility import pt_irange, hide_ticks
from Data.DataLoader import Complex_Volumes

from StarMap.StarMap import load_starmap
from Pipeline.pipeline import run_pipelines, arlo_corrected

def phase_fits():
    vol_idx = 5
    slice_idx = 15
    xs = pt_irange(20, 170, 6).int()
    ys = pt_irange(10, 85, 6).int()
    pts = [(x, y) for x in xs for y in ys]

    t = pt_irange(0.005, 0.04, 8)
    vol = torch.cat((Complex_Volumes()[vol_idx].unsqueeze(0), Complex_Volumes()[vol_idx + 1].unsqueeze(0)), 0)
    B0, ip = estimate_delta_omega(vol)
    ip1 = ((ip + torch.pi) % (2 * torch.pi)) - torch.pi

    images = [
        (vol[1, 0, slice_idx, :, :].abs().T, "First Echo (AU)", "gray", [0, 10], ["0", "10"]),
        (B0[1, 0, slice_idx, :, :].T, "∆ω (Hz)", "twilight", [-500, 0, 500], ["-500", "0", "500"]),
        (ip1[1, 0, slice_idx, :, :].T, "Initial Phase (rad)", "twilight", [-22/7, 0, 22/7], ["-π", "0", "π"])
    ]

    fontsize=20
    
    fig = plt.figure()

    for i in range(len(images)):
        im, title, cmap, ticks, tick_labels = images[2 - i]
        imax = fig.add_subplot(3, 3, 3 * (2 - i) + 1)
        b = imax.imshow(im, cmap=cmap, vmin = ticks[0], vmax = ticks[-1])
        imax.set_title(title, fontsize=fontsize)
        hide_ticks(imax)
        cax = imax.inset_axes([-10, 0, 5, 96], transform=imax.transData)
        c_bar = plt.colorbar(b, ax=imax, cax=cax, orientation = "vertical", ticks=ticks)
        c_bar.ax.set_yticklabels(tick_labels) 
        c_bar.ax.tick_params(labelsize=15) 
        cax.yaxis.set_ticks_position("left")
        cax.yaxis.set_label_position("left")

    for i, (x, y) in enumerate(pts):
        m = B0[1, 0, slice_idx, x, y]
        c = ip[1, 0, slice_idx, x, y]
        angles = vol[1, :, slice_idx, x, y].angle()
        idx = (i // 6) * 9 + (i % 6) + 4
        ax = fig.add_subplot(6, 9, idx)

        a, = ax.plot(t, angles, c="red", label="Phase (wrapped)")
        b, = ax.plot(t, m * t + c, c="green", label="Linear fit")
        ax.plot(t,  torch.pi * torch.ones(8), c="grey", linestyle='dashed', linewidth=1)
        ax.plot(t,  torch.zeros(8), c="grey", linestyle='dashed', linewidth=1)
        ax.plot(t, -torch.pi * torch.ones(8), c="grey", linestyle='dashed', linewidth=1)

        ax.set_ylim(-5, 5)

        hide_ticks(ax)

        if idx == 54:
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.yaxis.set_tick_params(labelright=True)
            ax.set_yticks([-3.14, 0, 3.14], ["-π", "0", "π"])
            ax.set_xticks([0, 0.02, 0.04], [0, 20, 40])
            ax.tick_params(labelsize=15) 
            ax.set_xlabel("t (ms)", fontdict={"fontsize": fontsize})
            ax.set_ylabel("Phase (rad)", fontdict={"fontsize": fontsize})
            ax.yaxis.set_ticks_position("right")
            ax.yaxis.set_label_position("right")
        
        imax.plot(x, y, 'r.', markersize=2)
    
    fig.legend(handles=[a, b], loc=(0.47, 0.9), ncol = 2, fontsize=fontsize)
    
    plt.show()


def representative_d_omega_map():
    vol_idx = 0
    slice_idx = 15
    xs = pt_irange(20, 170, 6).int()
    ys = pt_irange(10, 85, 6).int()
    pts = [(x, y) for x in xs for y in ys]

    t = pt_irange(0.005, 0.04, 8)
    vol = Complex_Volumes()[vol_idx].unsqueeze(0)
    B0, ip = estimate_delta_omega(vol)
    ip = ((ip + torch.pi) % (2 * torch.pi)) - torch.pi



    starmap = load_starmap("./trained_models/StarMap.pt")

    b, n_t, d, h, w = vol.shape
    vol_t = pt_irange(0.005, 0.04, n_t).reshape(1, n_t, 1, 1 ,1).repeat(b, 1, d, h, w)

    pipelines = [
        lambda : starmap(torch.cat((vol.real, vol.imag), 1)), 
        lambda : arlo_corrected(vol, vol_t),
    ]

    _, _, recons, _ = run_pipelines(pipelines, vol)

    starmap_recon, arlo_recon = recons

    b, i = estimate_delta_omega(starmap_recon)
    i = ((i + torch.pi) % (2 * torch.pi)) - torch.pi




    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    for ax in [ax1, ax2, ax3]: hide_ticks(ax)

    a = ax1.imshow((starmap_recon[0, 5, slice_idx, :, :]).T.angle().detach(), cmap = "twilight", vmin=-torch.pi, vmax=torch.pi)
    b = ax2.imshow(starmap_recon[0, 5, slice_idx, :, :].T.angle().detach() - vol[0, 5, slice_idx, :, :].T.angle().detach(), cmap = "twilight")
    c = ax3.imshow((vol[0, 5, slice_idx, :, :]).T.angle().detach(), cmap='twilight', vmin=-torch.pi, vmax=torch.pi)

    print((starmap_recon[0, 5, slice_idx, :, :].T.angle() - (vol[0, 5, slice_idx, :, :]).T.angle()).pow(2).mean().sqrt())


    plt.colorbar(a, ax=ax1).ax.set_ylabel("MR Signal (Arbitrary Units)")
    plt.colorbar(b, ax=ax2).ax.set_ylabel("Frequency Offset - ∆ω (Hz)")
    
    c_bar = plt.colorbar(c, ax=ax3, ticks=[-torch.pi, 0, torch.pi])
    c_bar.ax.set_yticklabels(["-π", "0", "π"]) 
    c_bar.ax.set_ylabel("Phase - φ (r)")

    
    plt.show()

def B0_hist():
    dataset = Complex_Volumes("test")
    l = len(dataset)
 
    freq_map = estimate_delta_omega(dataset[0].unsqueeze(0))[0]
    for i in range(1, l):
        freq_map = torch.cat((freq_map, estimate_delta_omega(dataset[i].unsqueeze(0))[0]), 0)

    # torch.save(freq_map, "./Charts/data/freq_maps.pt")
    # freq_map = torch.load("./Charts/data/freq_maps.pt")

    gamma = 2.6752219e8 # gyromagnetic ratio of the proton
    B0_map = 1e6 * freq_map / gamma
    mean_B0 = round(B0_map.mean().data.item(), 1)
    nyquist_lim = round(1e6 * (torch.pi / 0.005) / gamma, 1)

    fig, ax = plt.subplots()
    ax.hist(B0_map.flatten(), 500)
    a = ax.axvline(-nyquist_lim, c="grey", linestyle='dashed', linewidth=1)    
    b = ax.axvline( nyquist_lim, c="grey", linestyle='dashed', linewidth=1)    
    c = ax.axvline(mean_B0, c="red", linestyle='solid', linewidth=1)
    ax.set_xlabel("Field Inhomogeneity - ∆B0 (ppm)")
    
    fig.legend([a, c], [f"Nyquist Limits: ±{nyquist_lim} ppm", f"Mean: {mean_B0} ppm"], loc=(0.6, 0.6))
    plt.show()

def single_voxel_phase():
    vol_idx = 4
    slice_idx = 15
    x = 96
    y = 48
    t = pt_irange(0.005, 0.04, 8)
    vol = Complex_Volumes()[vol_idx].unsqueeze(0)
    B0, ip = estimate_delta_omega(vol)

    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)


    # fig, axs = plt.subplots(5, 5)
    #fig, ax = plt.subplots()

    m = B0[0, 0, slice_idx, x, y]
    c = ip[0, 0, slice_idx, x, y]
    angles = vol[0, :, slice_idx, x, y].angle()

    a = ax.scatter(t, angles, c="red", label="In vivo")
    # b, = ax.plot(t, m * t + c, c="green", label="Linear fit")
    c, = ax2.plot(t, vol[0, :, slice_idx, x, y].real, c = "blue", label="Real")
    d, = ax2.plot(t, vol[0, :, slice_idx, x, y].imag, c = "purple", label="Imaginary")

    ax.axhline(torch.pi, color="gray", linestyle='dashed', linewidth=1)
    ax.axhline(0, color="gray", linestyle='dashed', linewidth=1)
    ax.axhline(-torch.pi, color="gray", linestyle='dashed', linewidth=1)

    ax.set_ylim(-7, 8)

    ax.xaxis.set_tick_params(labelbottom=True)
    ax.yaxis.set_tick_params(labelleft=True)
    ax.set_yticks([-3.14, 0, 3.14], ["-π", "0", "π"])
    ax2.set_xticks([0, 0.02, 0.04])
    ax2.set_xlabel("t (s)")
    ax.set_ylabel("Phase (r)")
    ax2.set_ylabel("MR Signal (Arbitrary Units)")

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    for l in ax.xaxis.get_major_ticks(): l.set_visible(False)
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax.xaxis.set_tick_params(color='w')
    ax2.xaxis.tick_bottom()     
    
    fig.legend(handles=[c, d], loc=(0.4, 0.4))
    fig.subplots_adjust(hspace=0)

    plt.show()


def single_voxel_phase_fit():
    vol_idx = 4
    slice_idx = 15
    x = 96
    y = 48
    t = pt_irange(0.005, 0.04, 8)
    tz = pt_irange(0, 0.04, 9)
    vol = Complex_Volumes()[vol_idx].unsqueeze(0)
    B0, ip = estimate_delta_omega(vol)


    fig, ax = plt.subplots()

    m = B0[0, 0, slice_idx, x, y]
    c = ip[0, 0, slice_idx, x, y]

    angles = vol[0, :, slice_idx, x, y].angle()
    m_, c_ = np.polyfit(t, angles, 1)

    angles_ = np.unwrap(angles, axis=0)

    a = ax.scatter(t, angles, c="red", label="Wrapped phases")
    d, = ax.plot(tz, m_ * tz + c_, c="red", label="Wrapped fit")
    e = ax.scatter(t, angles_, c="green", label="Unwrapped phases", marker="*")
    b, = ax.plot(tz, m * tz + c, c="green", label="Unwrapped fit")

    ax.axhline(torch.pi, color="gray", linestyle='dashed', linewidth=1)
    ax.axhline(0, color="gray", linestyle='dashed', linewidth=1)
    ax.axhline(-torch.pi, color="gray", linestyle='dashed', linewidth=1)

    ax.set_ylim(-7, 8)
    ax.set_xlim(0, 0.041)


    ax.set_yticks([-3.14, 0, 3.14], ["-π", "0", "π"])
    ax.set_xticks([0, 0.02, 0.04])
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Phase (r)")

    fig.legend(handles=[e, b, a, d], loc=(0.6, 0.15))

    plt.show()
