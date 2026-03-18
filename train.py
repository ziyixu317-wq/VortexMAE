
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import pyvista as pv

from dataset import VortexMAEDataset
from model import VortexMAE, vortex_mae_pretrain_loss

def main():
    parser = argparse.ArgumentParser(description="VortexMAE Pre-training Script (Paper Consistent)")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .vti files")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="MAE masking ratio")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_vortexmae", help="Save directory")
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Datasets & Loaders
    train_dataset = VortexMAEDataset(args.data_dir, split="pretrain_train")
    test_dataset = VortexMAEDataset(args.data_dir, split="pretrain_eval")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Get spatial dimensions and channels
    sample = train_dataset[0]
    in_chans, D, H, W = sample.shape
    print(f"Input Shape: {in_chans}x{D}x{H}x{W}")
    
    # 2. Model Initialization
    model = VortexMAE(
        in_chans=in_chans,
        mask_ratio=args.mask_ratio,
        embed_dim=48, # Scale based on VRAM
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24]
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 3. Training Loop
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            x_rec, mask = model(batch)
            loss = vortex_mae_pretrain_loss(x_rec, batch, mask)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 4. Evaluation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                x_rec, mask = model(batch)
                loss = vortex_mae_pretrain_loss(x_rec, batch, mask)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        scheduler.step()
        
        print(f"Epoch {epoch} | Train MSE: {avg_train_loss:.6f} | Test MSE: {avg_test_loss:.6f}")
        
        # 5. Checkpointing
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            ckpt_path = os.path.join(args.save_dir, "vortexmae_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': best_loss
            }, ckpt_path)
            print(f" -> Saved best checkpoint: {ckpt_path}")
            
        # 6. Visualization (Save first sample of test set reconstruction)
        if epoch % 10 == 0 or epoch == args.epochs:
            with torch.no_grad():
                # Re-run first batch of test set
                sample_batch = next(iter(test_loader)).to(device)
                x_rec, mask = model(sample_batch)
                
                # Take first one
                pred = x_rec[0].cpu().numpy()
                gt = sample_batch[0].cpu().numpy()
                
                # Un-normalize
                try:
                    mean = test_dataset.mean.squeeze() # (3,)
                    std = test_dataset.std.squeeze()
                    for c in range(in_chans):
                        pred[c] = pred[c] * std[c] + mean[c]
                        gt[c] = gt[c] * std[c] + mean[c]
                except:
                    pass
                
                # Save as VTI
                # VTK uses (x, y, z)
                vis_mesh = pv.ImageData()
                vis_mesh.dimensions = (W, H, D)
                
                # Components u,v,w
                for i, name in enumerate(["u", "v", "w"]):
                    vis_mesh.point_data[f"{name}_rec"] = pred[i].flatten(order='C')
                    vis_mesh.point_data[f"{name}_gt"] = gt[i].flatten(order='C')
                
                out_path = os.path.join(args.save_dir, f"recon_epoch_{epoch}.vti")
                vis_mesh.save(out_path)
                print(f" -> Saved reconstruction to {out_path}")

    print("\nPre-training Complete.")

if __name__ == "__main__":
    main()
