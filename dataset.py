
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
            potential_vectors = [k for k in mesh.point_data.keys() if 'velocity' in k.lower()]
            if potential_vectors:
                vec = mesh.point_data[potential_vectors[0]]
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
    Paper-consistent implementation:
      - Min-max normalization per channel (Eq. 2)
      - 128^3 random crop during training (Section IV.A.2)
      - Full grid for evaluation/inference
    """
    def __init__(self, data_dir, split="train", split_ratio=0.7, 
                 normalize=True, crop_size=128):
        self.data_dir = data_dir
        self.normalize = normalize
        self.crop_size = crop_size
        # Enable random cropping only for training splits
        self.do_crop = split in ("pretrain_train", "finetune_train", "train")
        
        # Collect and sort all .vti files
        self.all_files = sorted(glob.glob(os.path.join(data_dir, "*.vti")))
        if not self.all_files:
            raise FileNotFoundError(f"No .vti files found in {data_dir}")
            
        num_total = len(self.all_files)
        
        if split == "pretrain_train":
            end = max(int(num_total * 0.3), 1)
            self.files = self.all_files[:end]
        elif split == "pretrain_eval":
            start = max(int(num_total * 0.3), 1)
            end = max(int(num_total * 0.35), start + 1)
            self.files = self.all_files[start:end]
        elif split == "finetune_train":
            start = max(int(num_total * 0.35), 2)
            end = max(int(num_total * 0.4), start + 1)
            self.files = self.all_files[start:end]
        elif split == "inference":
            start = max(int(num_total * 0.4), 3)
            end = max(int(num_total * 0.7), start + 1)
            self.files = self.all_files[start:end]
        elif split == "train":
            num_train = int(num_total * split_ratio)
            self.files = self.all_files[:num_train]
        else:
            num_train = int(num_total * split_ratio)
            self.files = self.all_files[num_train:]
            
        print(f"[{split}] Loading {len(self.files)} files from {data_dir}...")
        
        # Pre-load raw data into memory
        self.data = []
        for f in self.files:
            self.data.append(read_vti_velocity(f))
        
        self.data = np.stack(self.data, axis=0)  # (N, 3, D, H, W)
        
        if self.normalize:
            # Paper Eq. 2: Channel-wise min-max normalization
            # v_hat_i = (v_i - min(v_i)) / (max(v_i) - min(v_i) + eps)
            eps = 1e-8
            # Compute per-channel min/max across all samples and spatial dims
            # Shape: (1, 3, 1, 1, 1)
            self.ch_min = self.data.min(axis=(0, 2, 3, 4), keepdims=True)
            self.ch_max = self.data.max(axis=(0, 2, 3, 4), keepdims=True)
            self.data = (self.data - self.ch_min) / (self.ch_max - self.ch_min + eps)
        
        _, _, D, H, W = self.data.shape
        print(f"  Grid shape: D={D}, H={H}, W={W} | Crop={self.do_crop} (size={crop_size})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]  # (3, D, H, W)
        
        if self.do_crop:
            _, D, H, W = sample.shape
            cs = self.crop_size
            
            # Clamp crop_size to actual dimension (handles grids < 128)
            cd = min(cs, D)
            ch = min(cs, H)
            cw = min(cs, W)
            
            # Random start indices
            d0 = np.random.randint(0, D - cd + 1) if D > cd else 0
            h0 = np.random.randint(0, H - ch + 1) if H > ch else 0
            w0 = np.random.randint(0, W - cw + 1) if W > cw else 0
            
            sample = sample[:, d0:d0+cd, h0:h0+ch, w0:w0+cw]
            
            # Pad to crop_size if any dim was smaller
            if cd < cs or ch < cs or cw < cs:
                padded = np.zeros((3, cs, cs, cs), dtype=np.float32)
                padded[:, :cd, :ch, :cw] = sample
                sample = padded
        
        return torch.from_numpy(sample.copy())

    @property
    def spatial_shape(self):
        return self.data.shape[2:]
