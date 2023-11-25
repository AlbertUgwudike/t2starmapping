import torch
from time import time

from FieldEstimator.field_estimator import estimate_delta_omega
from Simulator.analytic_simulator import differentiate_3d, sinc_3d, simulate_volume
from Simulator.filters import sobel3D
from utility import pt_irange, arlo, fst, snd, fmap


# a set of functions that integrate our pipeline

def arlo_corrected(vol, vol_t, arlo_n = 6, M0_n = 2):
    B0_offset_map, _ = estimate_delta_omega(vol)
    B0_x, B0_y, B0_z = differentiate_3d(B0_offset_map, sobel3D, "same")
    vsf = sinc_3d(B0_offset_map, B0_x, B0_y, B0_z, vol_t)
    vol_corr = vol / vsf   
    return arlo(vol_corr.abs(), vol_t, arlo_n, M0_n)


def run_pipelines(pipelines, vol):

    delta_omega, initial_phase = estimate_delta_omega(vol)

    def time_func(f):
        start = time()
        res = f()
        return res, time() - start
    
    outputs = fmap(time_func, pipelines)
    param_maps = fmap(fst, outputs)
    durations = fmap(snd, outputs)

    reconstruct = lambda param_map: simulate_volume(param_map, delta_omega, initial_phase)
    reconstructions = fmap(reconstruct, param_maps)

    RMSE = lambda recon: (recon - vol).abs().pow(2).mean().sqrt()
    diffs = fmap(RMSE, reconstructions)

    # print(fmap(lambda recon: (recon - vol).abs().pow(2).std(), reconstructions))
    # print(fmap(lambda recon: (recon - vol).abs().pow(2).mean(), reconstructions))

    return param_maps, durations, reconstructions, diffs
    

