import torch
from torch.utils.data import Dataset

#######################
### Transformations ###
#######################
class Standardize:
    def __init__(self, mean, std, eps=1e-6):
        self.mean = torch.from_numpy(mean).float()
        self.std  = torch.from_numpy(std).float()
        self.eps  = eps

    def __call__(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def invert(self, x):
        return x * (self.std + self.eps) + self.mean

class MinMaxScale:
    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value

    def __call__(self, x):
        return (x - self.min) / (self.max - self.min)

    def invert(self, x):
        return x * (self.max - self.min) + self.min

################
### Datasets ###
################

# Dataset for ensemble forecasts
class EnsembleDataset(Dataset):
    def __init__(self, data_array, transform = None):
        """
        Parameters
        ----------
        data_array : xr.DataArray
            Input 3D xarray DataArray with dimensions (ens, lat, lon), where:
            - ens: ensemble members (e.g., different model runs or simulations)
            - lat: latitude coordinates
            - lon: longitude coordinates
        """
        self.X = data_array.values
        self.transform = transform
        
    def __len__(self):
        return self.X.shape[0]
        
    def __getitem__(self, idx):
        x = self.X[idx]
        x = torch.as_tensor(x, dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        x = x.unsqueeze(0)  # add channel dimension
        return x