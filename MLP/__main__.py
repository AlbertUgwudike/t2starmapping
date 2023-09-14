import sys
import torch

from Data.DataLoader import SimulatedData
from .train import train
from .demo import demo_mlp
from .model import MLP
from .helpers import main

def print_usage():
    print("Usage: ")
    print("python3 -m MLP demo")
    print("python3 -m MLP train")

if __name__ == "__main__": 
    if len(sys.argv) != 2: print_usage()
    if sys.argv[1] == "train": train()
    if sys.argv[1] == "demo": 
        model = MLP()
        model.load_state_dict(torch.load("./models/mlp.pt"))
        demo_mlp(model)
    if sys.argv[1] == "helpers": main()