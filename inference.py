
import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import pyvista as pv
from tqdm import tqdm

from dataset import VortexMAEDataset
from model import VortexMAE

def main():
    parser = argparse.ArgumentParser(description="VortexMAE Inference/Visualization Script")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .vti files")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--save_dir", type=str, default="./inference_results", help="Where to save results")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="Mask ratio for visualization")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of test samples to visualize")
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Test Dataset (7:3 split consistency)
    test_dataset = VortexMAEDataset(args.data_dir, split="test", split_ratio=0.7)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    in_chans, D, H, W = test_dataset[0].shape
    
    # 2. Load Model
    model = VortexMAE(
        in_chans=in_chans,
        mask_ratio=args.mask_ratio,
        embed_dim=48,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24]
    ).to(device)
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. Running Inference
    print(f"Generating reconstructions for the first {args.num_samples} test samples...")
    
    counter = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if counter >= args.num_samples:
                break
                
            batch = batch.to(device)
            # Forward pass
            x_rec, mask = model(batch)
            
            # Extract single sample
            pred = x_rec[0].cpu().numpy()
            gt = batch[0].cpu().numpy()
            m = mask[0].cpu().numpy() # (1, D, H, W)
            
            # Un-normalize if possible
            try:
                mean = test_dataset.mean.squeeze()
                std = test_dataset.std.squeeze()
                for c in range(in_chans):
                    pred[c] = pred[c] * std[c] + mean[c]
                    gt[c] = gt[c] * std[c] + mean[c]
            except:
                pass
            
            # Save to VTI
            vis_mesh = pv.ImageData()
            vis_mesh.dimensions = (W, H, D)
            
            # Save Reconstructed vs Ground Truth for each component
            for i, name in enumerate(["u", "v", "w"]):
                vis_mesh.point_data[f"{name}_reconstructed"] = pred[i].flatten(order='C')
                vis_mesh.point_data[f"{name}_ground_truth"] = gt[i].flatten(order='C')
            
            # Also save the binary mask so we know what was hidden
            vis_mesh.point_data["mask"] = m[0].flatten(order='C')
            
            # We can also compute error
            error = np.sqrt(np.sum((pred - gt)**2, axis=0))
            vis_mesh.point_data["reconstruction_error"] = error.flatten(order='C')
            
            out_path = os.path.join(args.save_dir, f"test_sample_{counter:03d}.vti")
            vis_mesh.save(out_path)
            
            counter += 1

    print(f"\nInference complete. Results saved in {args.save_dir}")
    print("You can open these files in ParaView to compare the fields.")

if __name__ == "__main__":
    main()
