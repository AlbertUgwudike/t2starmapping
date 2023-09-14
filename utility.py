import numpy as np
import torch
import os
import sys

from scipy.optimize import least_squares

ep = 1e-11

# inclusive range 
def irange(start, stop, n, array, arange):
    if start == stop: return array([start])
    step = (stop - start) / (n - 1)
    return arange(start, stop + ep, step)

def np_irange(start, stop, n): return irange(start, stop, n, np.array, np.arange)
def pt_irange(start, stop, n): return irange(start, stop, n, torch.tensor, torch.arange)

def hcat(a, b): return np.concatenate((a, b), axis=1)
def vcat(a, b): return np.concatenate((a, b), axis=0)

def identity(a): return a

def fst(pair): return pair[0]

def snd(pair): return pair[1]

def col_vectors(arr): 
    f = lambda cv: cv.squeeze()
    return list(map(f, torch.hsplit(arr, arr.shape[1])))

def safe_div(a, b): return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

def arlo(vol, vol_t, n = 4, M0_n = 2):
    factor = 0.005 / 3
    integrated = factor * (vol[:, :(n - 2), :, :, :] + 4 * vol[:, 1:(n-1), :, :, :] + vol[:, 2:n, :, :, :])
    differentiated = vol[:, :(n - 2), :, :, :] - vol[:, 2:n, :, :, :]
    dividend = (integrated ** 2).sum(1) + factor * (integrated * differentiated).sum(1)
    quotient = factor * (differentiated ** 2).sum(1) + (integrated * differentiated).sum(1)
    arlo_R2 = (quotient / dividend).unsqueeze(1)
    arlo_M0 = (vol[:, :M0_n, :, :, :] / torch.exp(-arlo_R2 * vol_t)[:, :M0_n, :, :, :]).mean(dim=1, keepdim=True)
    return torch.cat((arlo_M0.abs(), arlo_R2.abs()), 1)


def ms_format(seconds): return round(1000 * seconds)

def mag(img):
    real = img[:, :8, :, :]
    imag = img[:, 8:, :, :]
    return torch.sqrt(torch.pow(real, 2) + torch.pow(imag, 2))

def read_complex_volume(prefix):
    dims = list(map(int, open(prefix + "images.hdr").readlines()[1].strip().split(" ")))[:-1]
    n = 2 * np.prod(dims)
    data = np.fromfile(prefix + "images.cfl", dtype=np.single)
    real = data[np.arange(0, n, 2)].reshape(dims[::-1])[:, 5:-5, :, :]
    imag = data[np.arange(1, n, 2)].reshape(dims[::-1])[:, 5:-5, :, :]
    return np.concatenate((real, imag), axis=0)


def get_dirs(parent):
    return [os.path.abspath(parent + path) for path in os.listdir(parent) if os.path.isdir(parent + path)]

def get_volume_paths(patient_folders_path):
    patient_paths = get_dirs(patient_folders_path)
    volume_paths_per_patient = [ get_dirs(dn + "/") for dn in patient_paths ]
    return [ path for path_list in volume_paths_per_patient for path in path_list ]

def panic(msg):
    print(msg)
    exit()


def hide_ticks(ax):
    # Hide X and Y axes label marks
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)

    # Hide X and Y axes tick marks
    ax.set_xticks([])
    ax.set_yticks([])


def NLLS(vol_corr, t):

    def f(params):
        M0      = params[0].repeat(t.shape[0])
        R2_star = params[1].repeat(t.shape[0])

        return M0 * np.exp(-R2_star * np.array(t)) - np.array(vol_corr)

    ls = least_squares(f, [1.0, 10])

    return ls.x[0], ls.x[1]

def handle_args(args_dict, module_name):
    if len(sys.argv) != 2 or sys.argv[1] not in args_dict.keys(): 
        print("Usage: ")
        for key in args_dict.keys(): print(f"python3 -m {module_name} {key}")
    else: args_dict[sys.argv[1]]()

def fmap(f, arr):
    return list(map(f, arr))



