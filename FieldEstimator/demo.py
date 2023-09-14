import matplotlib.pyplot as plt
import torch

from .field_estimator import estimate_delta_omega
from utility import pt_irange, hide_ticks
from Data.DataLoader import Complex_Volumes

def phase_fits():
    vol_idx = 5
    slice_idx = 15
    xs = pt_irange(20, 170, 6).int()
    ys = pt_irange(10, 85, 6).int()
    pts = [(x, y) for x in xs for y in ys]

    t = pt_irange(0.005, 0.04, 8)
    vol = torch.cat((Complex_Volumes()[vol_idx].unsqueeze(0), Complex_Volumes()[vol_idx + 1].unsqueeze(0)), 0)
    B0, ip = estimate_delta_omega(vol)

    # fig, axs = plt.subplots(5, 5)
    fig = plt.figure()

    imax = fig.add_subplot(1, 3, 1)
    imax.imshow(vol[1, 0, slice_idx, :, :].abs(), cmap="gray")
    imax.xaxis.set_tick_params(labelbottom=False)
    imax.yaxis.set_tick_params(labelleft=False)
    imax.set_xticks([])
    imax.set_yticks([])

    for i, (x, y) in enumerate(pts):
        m = B0[1, 0, slice_idx, x, y]
        c = ip[1, 0, slice_idx, x, y]
        angles = vol[1, :, slice_idx, x, y].angle()
        idx = (i // 6) * 9 + 3 + (i % 6) + 1
        ax = fig.add_subplot(6, 9, idx)

        a, = ax.plot(t, angles, c="red", label="In vivo")
        b, = ax.plot(t, m * t + c, c="green", label="Linear fit")
        ax.plot(t,  torch.pi * torch.ones(8), c="grey", linestyle='dashed', linewidth=1)
        ax.plot(t,  torch.zeros(8), c="grey", linestyle='dashed', linewidth=1)
        ax.plot(t, -torch.pi * torch.ones(8), c="grey", linestyle='dashed', linewidth=1)

        ax.set_ylim(-5, 5)

        hide_ticks(ax)

        if idx == 49:
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.yaxis.set_tick_params(labelleft=True)
            ax.set_yticks([-3.14, 0, 3.14], ["-π", "0", "π"])
            ax.set_xticks([0, 0.02, 0.04])
            ax.set_xlabel("t (s)")
            ax.set_ylabel("Phase (r)")
        
        imax.plot(y, x, 'r.')
    
    fig.legend(handles=[a, b], loc='upper center')
    
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
    ip = ((ip + 20 * torch.pi) % (2 * torch.pi)) - torch.pi

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    for ax in [ax1, ax2, ax3]: hide_ticks(ax)

    a = ax1.imshow((vol[0, 0, slice_idx, :, :]).T.angle(), cmap = "twilight")
    b = ax2.imshow((B0[0, 0, slice_idx, :, :]).T)
    c = ax3.imshow((ip[0, 0, slice_idx, :, :]).T, cmap='twilight', vmin=-torch.pi, vmax=torch.pi)


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