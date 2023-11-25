import sys

from .demo import single_voxel_vsf, analytic_vs_numerical, compare_interpolations, interpolation_demo, residuals_of_interpolations, vsf_comparison, numerical_comparison
from .residuals import analytic_residuals, simulator_residuals
from .training_data import generate_mlp_data
from utility import handle_args

args_dict = {
    "analytic_vs_numerical": analytic_vs_numerical,
    "generate_analytic_residuals": analytic_residuals,
    "generate_numerical_residuals": simulator_residuals,
    "single_voxel_vsf": single_voxel_vsf,
    "compare_interpolations": compare_interpolations,
    "interpolation_demo": interpolation_demo,
    "residuals_of_interpolations": residuals_of_interpolations,
    "vsf_comparison": vsf_comparison,
    "numerical_comparison": numerical_comparison,
    "mlp": lambda: generate_mlp_data("./data/mlp_training_data.csv")
}

if __name__ == "__main__": handle_args(args_dict, "Simulator")