import torch 

from utility import pt_irange
from .interpolator import polynomial_interpolator

# Generates simulated MRI data with user provided parameters
# Based upon the isochromat summation method:  https://doi.org/10.1002/(SICI)1099-0534(1996)8:4<253::AID-CMR2>3.0.CO;2-Y
# Ported from Julia Implementation by Dr. Andreas Wetscherek
# Our implemntaiton uniquely performs abitray interpolation of a 3D off-resonance environment
# to simulate the effects of B0 inhomogeneity


# --------- parameters -------- #

# R2_star = [50]                    Observed relaxation rate (1/T2*) in s^-1 (eg 50hz)
# delta_omega_env                   3x3x3 volume represent ambient off-resonance environment
# t                                 timepoints to simulate (s)
# M0 = [1]                          initial magnetisation
# n_isochromats = 100_000           number of isochromats to use

# --------- returns -------- #

# Simulated complex MR signal as defiend by the parameters

def numerical_simulator(
        delta_omega_env = 50 * torch.arange(3**3).reshape(1, 3, 3, 3), 
        interpolator = polynomial_interpolator(1),
        M0 = torch.tensor([1]), 
        R2_star = torch.tensor([0]), 
        t = pt_irange(0.005, 0.04, 8),
        n_isochromats = 8_000,
    ):

    # simulate cauchy-defined exponential decay
    FID = torch.exp(torch.outer(-R2_star, t))

    # simulate B0_inhomogeneity effects
    offres = interpolator(delta_omega_env, n_isochromats)
    F = torch.einsum('ij, k -> ijk', offres, 1j * t).exp().mean(0)

    # T2* decay including B0
    return (M0 * FID.T * F.T).T
