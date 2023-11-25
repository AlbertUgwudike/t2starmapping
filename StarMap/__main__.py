import sys

from .train import train
from .demo import loss_curve, pixel_demo_big
from utility import handle_args

args_dict = {
    "pixel_demo_big": pixel_demo_big,
    "loss_curve": loss_curve,
    "train": train,
}

if __name__ == "__main__": handle_args(args_dict, "StarMap")