
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
        self.do_crop = split not in ("inference")
        
        # 1. Collect and sort all .vti files
        self.all_files = sorted(glob.glob(os.path.join(data_dir, "*.vti")))
        if not self.all_files:
            raise FileNotFoundError(f"No .vti files found in {data_dir}")
            
        num_total = len(self.all_files)
        
        # 2. Consistent split indices
        idx_p1 = int(num_total * 0.3)
        idx_p2 = int(num_total * 0.4)
        idx_f = int(num_total * 0.65)
        
        if split == "pretrain_train":
            self.files = self.all_files[:max(1, idx_p1)]
        elif split == "pretrain_eval":
            self.files = self.all_files[idx_p1:max(idx_p1+1, idx_p2)]
        elif split == "finetune_train":
            self.files = self.all_files[idx_p2:max(idx_p2+1, idx_f)]
        elif split == "inference":
            remaining_files = self.all_files[idx_f:]
            if len(remaining_files) > 0:
                import random
                rng = random.Random(42)
                self.files = sorted(rng.sample(remaining_files, k=min(3, len(remaining_files))))
            else:
                self.files = []
        elif split == "train":
            self.files = self.all_files[:int(num_total * split_ratio)]
        else: # test/eval/other
            self.files = self.all_files[int(num_total * split_ratio):]
            
        print(f"[{split}] Lazy loading {len(self.files)} files from {data_dir}...")
        
        # 3. Memory-efficient min-max calculation
        if self.normalize and self.files:
            from tqdm import tqdm
            print(f"  Calculating normalization stats (streaming)...")
            self.ch_min = None
            self.ch_max = None
            
            for f in tqdm(self.files, desc="Global Stats", leave=False):
                sample = read_vti_velocity(f) # (3, D, H, W)
                # Compute min/max for this file per channel
                f_min = sample.min(axis=(1, 2, 3), keepdims=True) # (3, 1, 1, 1)
                f_max = sample.max(axis=(1, 2, 3), keepdims=True)
                
                if self.ch_min is None:
                    self.ch_min = f_min
                    self.ch_max = f_max
                else:
                    self.ch_min = np.minimum(self.ch_min, f_min)
                    self.ch_max = np.maximum(self.ch_max, f_max)
            
            # Final shapes: (1, 3, 1, 1, 1) for broadcasting with (N, 3, D, H, W) in memory-based logic
            # Here we'll use (3, 1, 1, 1) for lazy loading samples
            print(f"  Normalization stats captured.")

        # 4. Get Grid Shape from first file
        if self.files:
            s0 = read_vti_velocity(self.files[0])
            self.grid_shape = s0.shape[1:]
            print(f"  Grid shape: {self.grid_shape} | Crop={self.do_crop} (size={crop_size})")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Lazy load from disk
        sample = read_vti_velocity(self.files[idx]) # (3, D, H, W)
        
        if self.normalize:
            eps = 1e-8
            # Normalize using pre-calculated global stats
            sample = (sample - self.ch_min) / (self.ch_max - self.ch_min + eps)
        
        if self.do_crop:
            _, D, H, W = sample.shape
            cs = self.crop_size
            
            cd = min(cs, D)
            ch = min(cs, H)
            cw = min(cs, W)
            
            d0 = np.random.randint(0, D - cd + 1) if D > cd else 0
            h0 = np.random.randint(0, H - ch + 1) if H > ch else 0
            w0 = np.random.randint(0, W - cw + 1) if W > cw else 0
            
            sample = sample[:, d0:d0+cd, h0:h0+ch, w0:w0+cw]
            
            if cd < cs or ch < cs or cw < cs:
                padded = np.zeros((3, cs, cs, cs), dtype=np.float32)
                padded[:, :cd, :ch, :cw] = sample
                sample = padded
        
        return torch.from_numpy(sample.copy())

    @property
    def spatial_shape(self):
        return self.grid_shape
