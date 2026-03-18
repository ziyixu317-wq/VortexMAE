
import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import pyvista as pv
from tqdm import tqdm

from dataset import VortexMAEDataset
from model import VortexMAE
from vortex_utils import calculate_ivd

def main():
    parser = argparse.ArgumentParser(description="VortexMAE Inference/Visualization Script (Vortex Mask)")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .vti files")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to fine-tuned .pth checkpoint")
    parser.add_argument("--save_dir", type=str, default="./vortex_results", help="Where to save results")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of test samples to visualize")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary threshold for vortex mask")
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Test Dataset
    test_dataset = VortexMAEDataset(args.data_dir, split="inference")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    in_chans, D, H, W = test_dataset[0].shape
    
    # 2. Load Fine-tuned Model
    model = VortexMAE(
        in_chans=in_chans,
        out_chans=1,
        mode='segmentation'
    ).to(device)
    
    print(f"Loading fine-tuned checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 3. Running Inference
    print(f"Generating vortex identification masks for {args.num_samples} samples...")
    
    counter = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if counter >= args.num_samples:
                break
                
            batch = batch.to(device)
            # Pred output is sigmoid probability
            pred_prob = model(batch)
            
            # Extract GT IVD on the fly
            gt_ivd = calculate_ivd(batch)
            gt_mask = (gt_ivd > 0).float()
            
            # Prepare data for VTI
            p_mask = pred_prob[0, 0].cpu().numpy()
            g_mask = gt_mask[0].cpu().numpy()
            b_mask = (p_mask > args.threshold).astype(np.float32)
            
            # Save to VTI
            mesh = pv.ImageData()
            mesh.dimensions = (W, H, D)
            
            mesh.point_data["GT_IVD_Mask"] = g_mask.flatten(order='C')
            mesh.point_data["Pred_Prob_Map"] = p_mask.flatten(order='C')
            mesh.point_data["Binary_Selection"] = b_mask.flatten(order='C')
            
            out_path = os.path.join(args.save_dir, f"vortex_id_{counter:03d}.vti")
            mesh.save(out_path)
            
            counter += 1

    print(f"\nInference complete. Visual comparison VTIs saved in {args.save_dir}")
    print("Use ParaView to visualize GT_IVD_Mask vs Binary_Selection.")

if __name__ == "__main__":
    main()
