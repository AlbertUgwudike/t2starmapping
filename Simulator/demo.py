import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from time import time
from functools import reduce
from scipy.stats import normaltest
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW
import numpy as np
from interpol import resize

from utility import pt_irange, hide_ticks
from .numerical_simulator import numerical_simulator, polynomial_interpolator
from .analytic_simulator import sinc_3d, sobel3D, differentiate_3d
from Data.DataLoader import Voxel_Cube
from utility import NLLS


def simulator_speeds():
    # retrieved from std output of simulator_residuals() and analytic_residuals()
    times = [0.4432, 12.88, 34.15, 81.38, 149.15, 211.535]
    voxel_counts = [22 * 8 * 192 * 96] + [500 * 8] * 5
    speeds = torch.tensor([v_c / t for t, v_c in zip(times, voxel_counts)])

    fig, ax = plt.subplots()
    x_labels = ["Analytic", "Linear", "Quadratic", "Cubic", "Quartic", "Quintic"]
    ax.bar(x_labels, speeds, color="#7bc8f6")
    l = ax.axhline(22 * 8 * 192 * 96, c="grey", linestyle='dashed', linewidth=1)
    m = ax.axhline(2 * 22 * 8 * 192 * 96, c="grey", linestyle='-.', linewidth=1)
    ax.set_yscale("log")
    ax.set_ylabel("Processing Speed (Voxels/s)")
    ax.set_xlabel("Simulation Strategy")

    fig.legend([l, m], ["1 3D-Volume per second", "2 3D-Volumes per second"], loc = (0.6, 0.6))

    plt.show()

def analytic_vs_numerical():
    n_t = 9
    t  = pt_irange(0.0001, 0.04, n_t)
    vol_t = t.reshape(1, n_t, 1, 1, 1).repeat(1, 1, 3, 3, 3)

    fig = plt.figure()


    # good = 780, 485
    for n in range(5):

        dataset = Voxel_Cube(1000)
        delta_omega = dataset[185 + 150 * n][1].unsqueeze(0).unsqueeze(0)
        grads = differentiate_3d(delta_omega)
        analytic_vsf = sinc_3d(delta_omega, *grads, vol_t)
        simulate_vsf = numerical_simulator(delta_omega_env=delta_omega, interpolator=polynomial_interpolator(1), t=t)

        for i, f in enumerate([torch.real, torch.imag]):
            ax = fig.add_subplot(5, 4, 4 * n + 2 + i + 1)
            
            if n == 4: ax.set_xlabel("Time (s)")
            else:
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.set_xticks([])
            if n == 0 and i == 0: ax.set_title("Real F(t)")
            if n == 0 and i == 1: ax.set_title("Imaginary F(t)")


            ax.set_ylim(-1.1, 1.1)
            ln1 = ax.plot(t, f(analytic_vsf[0, :, 1, 1, 1]), c="black", label="Analytic")
            ln2 = ax.plot(t, f(simulate_vsf[0, :]), c="blue", label="Simulated")
            
        for z in range(3):
            headers = ["Top", "Middle", "Lower"]
            imax = fig.add_subplot(5, 6, n * 6 + z + 1)
            if n == 0: imax.set_title(headers[z])
            a = imax.imshow(delta_omega[0, 0, z, :, :], vmin = -200, vmax = 200, cmap='RdBu_r')
            if z == 1: imax.plot(1, 1, "k*")
            hide_ticks(imax)

    
    c_bar = plt.colorbar(a, ax=fig.get_axes(), location="left")
    c_bar.ax.set_ylabel("∆ω (Hz)")

    fig.get_axes()[15].legend(loc="center")
    plt.show()



def residuals_of_interpolations():

    dfs = list(map(lambda n: pd.read_csv(f"./Charts/data/Degree_{n}_Residuals.csv"), [1, 2, 3, 4, 5]))
    comb = reduce(lambda a, b: pd.DataFrame(data=pd.concat((a["Residuals"], b["Residuals"]), axis=1)), dfs)
    mdf = pd.DataFrame(data = comb.iloc[:, :].values, columns=[f"Degree_{i}" for i in range(1, 6)])
    mdf = mdf[(mdf >= 0).all(axis=1)]
    simulated_data = -np.log(mdf.values)

    dir = "../data/simulation_residuals/"
    analytic_data = -np.log(pd.read_csv(dir + "Analytic_Residuals.csv").values[:, 1])
    analytic_data_ = -np.log(pd.read_csv(dir + "Analytic_Residuals_.csv").values[:, 1])

    data = np.hstack((
        np.expand_dims(analytic_data, 1), 
        simulated_data,
        #np.expand_dims(analytic_data_, 1), 
    ))
    
    # data pass normaility test
    results = normaltest(data)
    print(results.pvalue)
    # pvalue=array([1.55219236e-06 7.42237316e-05 9.15372330e-04 2.28935383e-04 4.07175085e-07 4.16039727e-07])

    t_tests = list(map(lambda n: CompareMeans(DescrStatsW(data[:, 0]), DescrStatsW(data[:, n])).ztest_ind()[1], [1, 2, 3, 4, 5]))
    print(t_tests)

    _, (ax, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={ "height_ratios": [20,1] })

    params = dict(
        x=np.arange(6), height=data.mean(0), yerr=data.std(0), 
        align='center', color='k', ecolor='k', capsize=10, width=0.5
    )

    ax.bar(**params)
    ax2.bar(**params)

    ax.set_ylim(3, 5.3)  # most of the data   
    ax2.set_ylim(0, 0.1)  # outliers only


    # hide the spines between ax and ax2
    ax.spines['bottom'].set_linestyle('--')
    ax2.spines['top'].set_linestyle('--')
    ax.yaxis.get_major_ticks()[0].label1.set_visible(False)
    ax2.yaxis.get_major_ticks()[-1].label1.set_visible(False)
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax.xaxis.set_tick_params(color='w')
    ax2.xaxis.tick_bottom()     
    
    ax2.set_xticklabels(["Constant", "Analytic", "Linear", "Quadratic", "Cubic", "Quartic", "Quintic", "Bogus"])
    ax2.set_xlabel("Type of simulation")
    ax.set_ylabel("Mean Transformed Residual Sum \nfollowing NLLS ( -log(AU^2) )")
    ax.axhline(data[:, 0].mean(), color="gray", linestyle="--")
    

    for i in range(6):

        # jitter time
        # x = np.random.normal(i, 0.08, size=data.shape[0])
        # y = data[:, i]
        # ax.plot(x, y, 'r.', alpha=0.04)
        
        # significance
        if i < 3: continue

        x = i - 1 / 2
        y = i / 7 + 5

        # props = {'connectionstyle':'arc', 'arrowstyle':'-', 'capstyle':'round'}
        # ax.annotate('', xy=(0,y), xytext=(i,y), arrowprops=props, zorder=10)
        ax.annotate("*", xy=(i,5), zorder=10, ha='center')
    
    plt.subplots_adjust(hspace=0.02)
    plt.show()

def single_voxel_vsf():

    n_t = 100
    t  = pt_irange(0.005, 0.04, n_t)
    vol_t = t.reshape(1, n_t, 1, 1, 1).repeat(1, 1, 3, 3, 3)
    
    delta_omega = 10 * Voxel_Cube(1000)[800][1].unsqueeze(0).unsqueeze(0)
    
    grads = differentiate_3d(delta_omega)
    analytic_vsf = sinc_3d(delta_omega, *grads, vol_t)

    simulate_vsf = numerical_simulator(delta_omega_env=delta_omega, interpolator=polynomial_interpolator(1), t=t)

    _, ax = plt.subplots()
    ax.set_ylim(0, 1)
    ax.set_ylabel("MR Signal (Arbitrary Units)")
    ax.set_xlabel("Time (s)")
    ax.plot(t, simulate_vsf[0, :].abs(), c="blue", label="Simulated")

    # _, ax = plt.subplots(1, 3)

    # for i, f in enumerate([torch.abs, torch.real, torch.imag]):
    #     ax[i].set_ylim(0, 1)
    #     ax[i].set_ylabel("MR Signal (Arbitrary Units)")
    #     ax[i].set_xlabel("Time (s)")

    #     # ax[i].plot(t, f(analytic_vsf[0, :, 1, 1, 1]), c="black", label="Analytic")
    #     ax[i].plot(t, f(simulate_vsf[0, :]), c="blue", label="Simulated")

    plt.legend()
    plt.show()

def compare_interpolations():
    cubes = Voxel_Cube(1000)
    cube = cubes[400][1].unsqueeze(0).unsqueeze(0)
    x_labels = ["Linear", "Quadratic", "Cubic", "Quartic", "Quintic"]

    fig, axs = plt.subplots(11, 6)

    axs[0][0].set_ylabel("Top")
    axs[-1][0].set_ylabel("Bottom")

    for i in range(11):
        hide_ticks(axs[i][0])
        if i % 5 == 0: 
            axs[i][0].imshow(cube[0, 0, i//5, :, :], vmin = -67, vmax = -36)
            rect = Rectangle((0.5, 0.5), width=1, height=1, facecolor='none', edgecolor='gray')
            axs[i][0].add_patch(rect)
        else: axs[i][0].imshow(torch.ones(3, 3), cmap="gray")

    for degree in range(1, 6):
        r = 20
        opt = dict(shape=[3*r, 3*r, 3*r], anchor='edges', bound='repeat', interpolation=degree)
        intravoxel = resize(cube, **opt)

        for i in range(11):
            hide_ticks(axs[i][degree])
            a = axs[i][degree].imshow(intravoxel[0, 0, min(i * 6, 59), :, :], vmin = -67, vmax = -36)
            if i == 10: axs[i][degree].set_xlabel(x_labels[degree - 1])
            rect = Rectangle((r, r), width=r, height=r, facecolor='none', edgecolor='gray')
            axs[i][degree].add_patch(rect)

    c_bar = plt.colorbar(a, ax=fig.get_axes(), location="right")
    c_bar.ax.set_ylabel("∆ω (Hz)")
    
    plt.show()


