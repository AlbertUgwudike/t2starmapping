from .demo import distributions
from utility import handle_args

args_dict = {
    "distributions": distributions
}

if __name__ == "__main__": handle_args(args_dict, "Data")