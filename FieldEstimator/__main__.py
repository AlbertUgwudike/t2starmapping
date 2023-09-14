from .demo import phase_fits, representative_d_omega_map, B0_hist
from utility import handle_args

args_dict = {
    "phase_fitting_demo": phase_fits,
    "delta_omega_maps": representative_d_omega_map,
    "B0_histogram": B0_hist,
}

if __name__ == "__main__": handle_args(args_dict, "FieldEstimator")

