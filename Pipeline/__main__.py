from utility import handle_args
from .demo import reconstructions, pixel_reconstructions, speed_and_residuals, param_map_demo, single_pixel_reconstructions

args_dict = {
    "reconstructions": reconstructions,
    "pixel_reconstructions": pixel_reconstructions,
    "single_pixel_reconstructions": single_pixel_reconstructions,
    "speed_and_residuals": speed_and_residuals,
    "param_map_demo": param_map_demo,
}

if __name__ == "__main__": handle_args(args_dict, "Pipeline")