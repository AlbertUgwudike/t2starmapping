from .demo import distributions, present_voxel_cubes
from utility import handle_args

args_dict = {
    "distributions": distributions, 
    "present_voxel_cubes": present_voxel_cubes
}

if __name__ == "__main__": handle_args(args_dict, "Data")