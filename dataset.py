
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pyvista as pv

def read_vti_velocity(filepath, velocity_names=("u", "v", "w")):
    """Reads a .vti file and returns the velocity field as (3, D, H, W)."""
    mesh = pv.read(filepath)
    dims = mesh.dimensions  # (nx, ny, nz)
    
    components = []
    for name in velocity_names:
        if name in mesh.point_data:
            arr = mesh.point_data[name]
        elif name in mesh.cell_data:
            arr = mesh.cell_data[name]
        else:
            # Try searching for vectors if scalar names fail
            # Paper might use 'velocity' array
            potential_vectors = [k for k in mesh.point_data.keys() if 'velocity' in k.lower()]
            if potential_vectors:
                vec = mesh.point_data[potential_vectors[0]]
                # If it's a 3-component vector
                if len(vec.shape) == 2 and vec.shape[1] == 3:
                   u = vec[:, 0].reshape(dims[2], dims[1], dims[0])
                   v = vec[:, 1].reshape(dims[2], dims[1], dims[0])
                   w = vec[:, 2].reshape(dims[2], dims[1], dims[0])
                   return np.stack([u, v, w], axis=0).astype(np.float32)
            
            raise KeyError(f"Velocity component '{name}' not found in {filepath}.")
            
        # VTK Fortran order (x-fastest) -> (nz, ny, nx) -> (D, H, W)
        arr_3d = arr.reshape(dims[2], dims[1], dims[0])
        components.append(arr_3d)
    
    return np.stack(components, axis=0).astype(np.float32)

class VortexMAEDataset(Dataset):
    """
    Dataset for loading a directory of .vti files.
    Supports temporal sequence splits (7:3 ratio).
    """
    def __init__(self, data_dir, split="train", split_ratio=0.7, normalize=True):
        self.data_dir = data_dir
        self.normalize = normalize
        
        # Collect and sort all .vti files
        self.all_files = sorted(glob.glob(os.path.join(data_dir, "*.vti")))
        if not self.all_files:
            raise FileNotFoundError(f"No .vti files found in {data_dir}")
            
        num_total = len(self.all_files)
        num_train = int(num_total * split_ratio)
        
        if split == "train":
            self.files = self.all_files[:num_train]
        else:
            self.files = self.all_files[num_train:]
            
        print(f"[{split}] Loading {len(self.files)} files from {data_dir}...")
        
        # Pre-load into memory for efficiency
        self.data = []
        for f in self.files:
            self.data.append(read_vti_velocity(f))
        
        self.data = np.stack(self.data, axis=0) # (N, 3, D, H, W)
        
        if self.normalize:
            # We use global statistics for normalization
            # Note: In a real scenario, we'd use train stats for validation too
            # For simplicity here, we normalize based on the provided split
            self.mean = self.data.mean(axis=(0, 2, 3, 4), keepdims=True)
            self.std = self.data.std(axis=(0, 2, 3, 4), keepdims=True) + 1e-8
            self.data = (self.data - self.mean) / self.std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])

    @property
    def spatial_shape(self):
        return self.data.shape[2:]
