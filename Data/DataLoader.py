from torch.utils.data import Dataset 
import torch 
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt

from utility import read_complex_volume, get_volume_paths
from FieldEstimator.field_estimator import estimate_delta_omega

class SimulatedData(Dataset):
    def __init__(self, filename):
        df = pd.read_csv(filename)
        self.signal = torch.tensor(df.iloc[:, 5:].values, dtype=torch.float32)
        self.parameters = torch.tensor(df.iloc[:, 1:5].values, dtype=torch.float32)
        self.x_mean_, self.x_std_ = torch.mean(self.parameters, 0), torch.std(self.parameters, 0)

    def __len__(self):
        return self.parameters.shape[0]

    def __getitem__(self, idx):
        parameters = self.parameters[idx, :]
        signal = self.signal[idx]

        return parameters, signal
    
    def x_mean(self): return self.x_mean_
    def x_std(self): return self.x_std_


class Images(Dataset):
    def __init__(self, prefix):
        metadata = json.load(open(prefix + "refvol.json"))
        data = np.fromfile(prefix + "refvol.raw", dtype=np.int16)
        refvol = data.reshape(metadata["dims"][::-1]).astype(np.float32)
        self.data = torch.tensor(refvol, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:, idx, :, :]
    

class Complex_Volumes(Dataset):
    def __init__(self, mode = "test", cropped = "False"):
        
        volume_paths = sorted(get_volume_paths(".Data/invivo/"))
        assert len(volume_paths) == 7
        self.volume_paths = volume_paths[-5:] if mode == 'test' else volume_paths[:-5]
        self.cropped = cropped
        self.n_vols = len(self.volume_paths)

    def __len__(self):
        return self.n_vols

    def __getitem__(self, idx):
        vol_path = self.volume_paths[idx]
        vol = read_complex_volume(vol_path + "/")
        vol_tensor = torch.tensor(vol, dtype=torch.float32)
        if self.cropped: vol_tensor = vol_tensor[:, :, 32:224, 85:181]
        c_data = vol_tensor[:8, :, :, :] + 1j * vol_tensor[8:, :, :, :]
        return (c_data.real / 3.5043e-05) + 1j * (c_data.imag / 3.4269e-05)
    
class Voxel_Cube(Dataset):
    def __init__(self, size):
        vol_path = get_volume_paths("../data/invivo/")[20]
        vol = read_complex_volume(vol_path + "/")
        cropped = torch.tensor(vol, dtype=torch.float32)[:, :, 32:224, 85:181] # 192, 96 
        self.vol = cropped[:8, :, :, :] / 3.5043e-05 + 1j * cropped[8:, :, :, :] / 3.4269e-05
        self.B0 = estimate_delta_omega(self.vol.unsqueeze(0))[0]
        self.n_voxels = 190 * 94 * 20
        self.size = min(size, self.n_voxels)

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx_):
        idx = idx_ * self.n_voxels // self.size
        d = idx // (190 * 94)
        x = (idx // 94) % 190
        y = (idx % 94)
        return self.vol[:, d:(d+3), x:(x+3), y:(y+3)], self.B0[0, 0, d:(d+3), x:(x+3), y:(y+3)]
    