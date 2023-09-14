import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

from Data.DataLoader import Complex_Volumes
from .StarMap import StarMap
from Simulator.analytic_simulator import simulate_volume
from FieldEstimator.field_estimator import estimate_delta_omega

def train():

    test_dataset, train_dataset = Complex_Volumes('test'), Complex_Volumes('train')
 
    data_loader = DataLoader(train_dataset, batch_size=len(train_dataset) // 2, shuffle=True, drop_last=True)  
    # data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True) 

    cuda_available = torch.cuda.is_available()
    print("CUDA available: ", cuda_available)

    device = torch.device('cuda' if cuda_available else 'cpu')

    starmap = StarMap().to(device)

    criterion = nn.MSELoss().to(device)

    optimizer = torch.optim.Adam(starmap.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

    num_epochs = 1000

    # imperatively accumulate the loss of each batch
    train_losses = []
    validation_losses = []
    curr_loss = 0

    for epoch in range(num_epochs):
            
        starmap.train()

        for img in data_loader:
            s_img = torch.cat((img.real, img.imag), 1).to(device)
            param_map = starmap(s_img)
            B0_offset_map, init_phase_map = estimate_delta_omega(img)
            sim_img = simulate_volume(param_map, B0_offset_map, init_phase_map, device=device)
            s_sim_img = torch.cat((sim_img.real, sim_img.imag), 1).to(device)
            curr_loss = criterion(s_img, s_sim_img) 
            optimizer.zero_grad()
            curr_loss.backward()
            optimizer.step()
        
        train_losses.append(curr_loss.data.item())
        
        if epoch % 50 == 49: scheduler.step()

        starmap.eval()

        with torch.no_grad():
            _, img =  next(enumerate(DataLoader(test_dataset, batch_size=3)))
            s_img = torch.cat((img.real, img.imag), 1).to(device)
            param_map = starmap(s_img)
            B0_offset_map, init_phase_map = estimate_delta_omega(img)
            sim_img = simulate_volume(param_map, B0_offset_map, init_phase_map, device=device)
            s_sim_img = torch.cat((sim_img.real, sim_img.imag), 1).to(device)
            loss = criterion(s_img, s_sim_img) 
            validation_losses.append(loss.data.item())
        

        save_model(starmap, train_losses, validation_losses, "./models/starmap1.pt")

        print(f"Epoch: {epoch}, Train Loss: {train_losses[-1]}, Validation Loss: {validation_losses[-1]}")


def save_model(starmap, train_losses, validation_losses, filename):
    torch.save(starmap.state_dict(), filename)
    losses = torch.tensor([train_losses, validation_losses]).T
    pd.DataFrame(data = losses, columns=["Train Loss", "Validation Loss"]).to_csv("./losses/starmap1_loss.csv")
    print(f"Saved current model parameters in {filename}")


if __name__ == "__main__": train()
