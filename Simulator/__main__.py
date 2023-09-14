import sys

from .demo import single_voxel_vsf, analytic_vs_numerical, compare_interpolations
from .residuals import analytic_residuals, simulator_residuals
from .training_data import generate_mlp_data
from utility import handle_args

args_dict = {
    "analytic_vs_numerical": analytic_vs_numerical,
    "generate_analytic_residuals": analytic_residuals,
    "generate_numerical_residuals": simulator_residuals,
    "single_voxel_vsf": single_voxel_vsf,
    "compare_interpolations": compare_interpolations,
    "mlp": lambda: generate_mlp_data("./data/mlp_training_data.csv")
}

if __name__ == "__main__": handle_args(args_dict, "Simulator")