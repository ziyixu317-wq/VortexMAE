
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pyvista as pv
from tqdm import tqdm
try:
    from scipy.ndimage import label as ccl_label
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not found. 3D CCL post-processing will be skipped.")

from dataset import VortexMAEDataset
from model import VortexMAE
from vortex_utils import calculate_ivd

def main():
    parser = argparse.ArgumentParser(description="VortexMAE Inference/Visualization Script (Vortex Mask)")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .vti files")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to fine-tuned .pth checkpoint")
    parser.add_argument("--save_dir", type=str, default="./vortex_results", help="Where to save results")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of test samples to visualize")
    parser.add_argument("--threshold", type=float, default=0.5, help="Segmentation threshold")
    parser.add_argument("--no_ccl", action="store_true", help="Disable 3D Connected Component Analysis")
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    # Device detection: TPU (via torch_xla) > CUDA > CPU
    use_tpu = False
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        use_tpu = True
        print(f"Using device: TPU ({device})")
    except ImportError:
        xm = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    
    # 1. Load Test Dataset
    test_dataset = VortexMAEDataset(args.data_dir, split="inference")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    in_chans, D, H, W = test_dataset[0].shape
    
    # 2. Load Fine-tuned Model
    model = VortexMAE(
        in_chans=in_chans,
        out_chans=1,
        mode='segmentation'
    )
    
    print(f"Loading fine-tuned checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 3. Running Inference
    print(f"Generating vortex identification masks for samples...")
    
    # Sliding window parameters
    d_win, h_win, w_win = 128, 128, 128
    stride = 96 # Overlap of 32
    
    counter = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            if counter >= args.num_samples:
                break
            
            # batch: (1, 3, D, H, W)
            _, _, D, H, W = batch.shape
            batch = batch.to(device)
            
            # Predict using sliding window
            full_logits = torch.zeros((1, 1, D, H, W), device=device)
            full_count = torch.zeros((1, 1, D, H, W), device=device)
            
            # Iterate over the grid
            # Using min(i, max(0, D-d_win)) Ensures we cover the last part of each dimension
            d_steps = list(range(0, max(1, D - d_win + 1), stride))
            if d_steps[-1] < D - d_win: d_steps.append(D - d_win)
            
            h_steps = list(range(0, max(1, H - h_win + 1), stride))
            if h_steps[-1] < H - h_win: h_steps.append(H - h_win)
            
            w_steps = list(range(0, max(1, W - w_win + 1), stride))
            if w_steps[-1] < W - w_win: w_steps.append(W - w_win)

            for d in d_steps:
                for h in h_steps:
                    for w in w_steps:
                        # Extract crop
                        d_s, d_e = d, d + d_win
                        h_s, h_e = h, h + h_win
                        w_s, w_e = w, w + w_win
                        
                        # Handle cases where grid is smaller than window
                        d_e, h_e, w_e = min(d_e, D), min(h_e, H), min(w_e, W)
                        crop = batch[:, :, d_s:d_e, h_s:h_e, w_s:w_e]
                        
                        # Pad if crop is smaller than 128 (e.g. if D=80)
                        pd = d_win - (d_e - d_s)
                        ph = h_win - (h_e - h_s)
                        pw = w_win - (w_e - w_s)
                        if pd > 0 or ph > 0 or pw > 0:
                            crop = F.pad(crop, (0, pw, 0, ph, 0, pd))
                        
                        # Forward (Now returns logits for stability)
                        logits = model(crop) 
                        
                        # Unpad and accumulate
                        logits = logits[:, :, :d_e-d_s, :h_e-h_s, :w_e-w_s]
                        full_logits[:, :, d_s:d_e, h_s:h_e, w_s:w_e] += logits
                        full_count[:, :, d_s:d_e, h_s:h_e, w_s:w_e] += 1
            
            # Final prediction mask
            full_prob = (full_logits / torch.clamp(full_count, min=1.0)).sigmoid()
            pred_mask = (full_prob > args.threshold).float().cpu().numpy()
            
            # Post-processing: 3D Connected Component Analysis (Eq. 23)
            if HAS_SCIPY and not args.no_ccl:
                labeled_array, num_features = ccl_label(pred_mask[0])
                if num_features > 0:
                    # Logic: Retain components larger than a tiny noise threshold (e.g., 10 voxels)
                    component_sizes = np.bincount(labeled_array.ravel())
                    # The paper says "Largest connected vortex regions"
                    # We'll keep components > 5% of max component size or a flat threshold
                    min_size = max(10, int(0.01 * component_sizes[1:].max()))
                    mask_ccl = np.zeros_like(pred_mask[0])
                    for i in range(1, num_features + 1):
                        if component_sizes[i] >= min_size:
                            mask_ccl[labeled_array == i] = 1
                    pred_mask[0] = mask_ccl
            
            # Extract GT IVD on the fly
            gt_ivd = calculate_ivd(batch)
            gt_mask = (gt_ivd > 0).float().cpu().numpy()
            
            # Prepare data for VTI (Squeeze batch/channel if needed)
            p_mask = pred_mask[0, 0] if pred_mask.ndim == 5 else pred_mask[0]
            g_mask = gt_mask[0]
            b_mask = (p_mask > args.threshold).astype(np.float32)
            
            # IMPORTANT: Remove padding artifacts (D=80 vs 128)
            # Find the actual data range in D dimension by looking for non-zero slices in Ground Truth
            valid_d = D
            # If D is already correct (e.g. 80), this should be fine. 
            # But let's be double sure for consistency with train.py
            for d_idx in range(D - 1, -1, -1):
                if np.abs(g_mask[d_idx]).sum() > 1e-6:
                    valid_d = d_idx + 1
                    break
            
            g_mask = g_mask[:valid_d, :, :]
            p_mask = p_mask[:valid_d, :, :]
            b_mask = b_mask[:valid_d, :, :]
            
            # Final VTI Dimensions (W, H, D)
            mesh = pv.ImageData()
            mesh.dimensions = (W, H, valid_d)
            
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
