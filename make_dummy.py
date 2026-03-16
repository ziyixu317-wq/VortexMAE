
import os
import numpy as np
import pyvista as pv

def create_dummy_vti(save_path, shape=(16, 32, 32)):
    D, H, W = shape
    mesh = pv.ImageData()
    mesh.dimensions = (W, H, D)
    
    # Random velocity fields
    mesh.point_data["u"] = np.random.randn(D * H * W).astype(np.float32)
    mesh.point_data["v"] = np.random.randn(D * H * W).astype(np.float32)
    mesh.point_data["w"] = np.random.randn(D * H * W).astype(np.float32)
    
    mesh.save(save_path)

if __name__ == "__main__":
    dummy_dir = "dummy_data"
    os.makedirs(dummy_dir, exist_ok=True)
    print(f"Generating 10 dummy VTI files in {dummy_dir}...")
    for i in range(10):
        create_dummy_vti(os.path.join(dummy_dir, f"frame_{i:03d}.vti"))
    print("Done.")
