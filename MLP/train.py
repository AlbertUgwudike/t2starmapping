import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from .model import MLP
from .demo import demo_mlp
from Data.DataLoader import Voxel_Cube

def train():

    dataset = Voxel_Cube(1000)
    data_loader = DataLoader(dataset, batch_size=50, shuffle=True)

    cuda_available = torch.cuda.is_available()
    print("CUDA available: ", cuda_available)
    device = torch.device('cuda' if cuda_available else 'cpu')

    mlp = MLP().to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.0005)

    num_epochs = 200
    display_step = 1

    # imperatively accumulate the loss of each batch
    losses = []

    for epoch in range(num_epochs):

        mlp.train()

        for B0_env in data_loader:
            pred = mlp(B0_env.to(device))
            true = pt_signal(R2_star=torch.tensor([0]), B0_env=B0_env * 20, device=device).abs()
            loss = criterion((pred.T - 0.8) / 0.2, (true.T - 0.8) / 0.2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())

        mlp.eval()

        save_model(mlp, losses, "./models/unet1.pt")

        print(f"Epoch: {epoch}, Train Loss: {losses[-1]}")



def save_model(mlp, train_losses, filename):
    torch.save(mlp.state_dict(), filename)
    losses = torch.tensor(train_losses)
    pd.DataFrame(data = losses, columns=["Train Loss"]).to_csv("./losses/unet1_loss.csv")
    print(f"Saved current model parameters in {filename}")


if __name__ == "__main__": train()