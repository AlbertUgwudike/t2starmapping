import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

from Data.DataLoader import Complex_Volumes
from .StarMap import StarMap, load_starmap
from Simulator.analytic_simulator import simulate_volume
from FieldEstimator.field_estimator import estimate_delta_omega

def train():

    test_dataset, train_dataset = Complex_Volumes('test'), Complex_Volumes('train')
 
    data_loader = DataLoader(train_dataset, batch_size=len(train_dataset) // 2, shuffle=True, drop_last=True)  
    # data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True) 

    cuda_available = torch.cuda.is_available()
    print("CUDA available: ", cuda_available)

    device = torch.device('cuda' if cuda_available else 'cpu')

    flatmap = StarMap().to(device)

    criterion = nn.MSELoss().to(device)

    optimizer = torch.optim.Adam(flatmap.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    num_epochs = 1000

    # imperatively accumulate the loss of each batch
    train_losses = []
    validation_losses = []
    curr_loss = 0

    for epoch in range(num_epochs):
            
        flatmap.train()

        for vol in data_loader:
            vol = vol.to(device)
            s_img = torch.cat((vol.real, vol.imag), 1).to(device)
            param_map = flatmap(s_img)
            B0_offset_map, init_phase_map = estimate_delta_omega(vol)
            sim_img = simulate_volume(param_map, B0_offset_map, init_phase_map, device=device)

            diff = (vol - sim_img).abs()
            curr_loss = criterion(diff, torch.zeros(diff.shape, dtype=torch.float32).to(device)) 

            optimizer.zero_grad()
            curr_loss.backward()
            optimizer.step()
        
        train_losses.append(curr_loss.data.item())
        
        # if train_losses[-1] < 0.15 and scheduler.get_lr() == 0.05: 
        #     scheduler.step()

        flatmap.eval()

        with torch.no_grad():
            _, vol =  next(enumerate(DataLoader(test_dataset, batch_size=7)))
            vol = vol.to(device)
            s_img = torch.cat((vol.real, vol.imag), 1).to(device)
            param_map = flatmap(s_img)
            B0_offset_map, init_phase_map = estimate_delta_omega(vol)
            sim_img = simulate_volume(param_map, B0_offset_map, init_phase_map, device=device)

            diff = (vol - sim_img).abs()
            loss = criterion(diff, torch.zeros(diff.shape, dtype=torch.float32).to(device)) 

            validation_losses.append(loss.data.item())
        

        save_model(flatmap, train_losses, validation_losses, "./trained_models/starmap1.pt")

        print(f"Epoch: {epoch}, Train Loss: {train_losses[-1]}, Validation Loss: {validation_losses[-1]}")


def save_model(flatmap, train_losses, validation_losses, filename):
    torch.save(flatmap.state_dict(), filename)
    losses = torch.tensor([train_losses, validation_losses]).T
    pd.DataFrame(data = losses, columns=["Train Loss", "Validation Loss"]).to_csv("./losses/starmap1_loss.csv")
    print(f"Saved current model parameters in {filename}")


if __name__ == "__main__": train()
