
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

from dataset import VortexMAEDataset
from model import VortexMAE
from vortex_utils import vortex_mae_finetune_loss, calculate_iou, calculate_ivd

def main():
    parser = argparse.ArgumentParser(description="VortexMAE Fine-tuning Script (Paper Consistent)")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .vti files")
    parser.add_argument("--pretrained_ckpt", type=str, required=True, help="Path to pre-trained checkpoint")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_finetune", help="Save directory")
    
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
    
    # 1. Dataset & Loader
    train_dataset = VortexMAEDataset(args.data_dir, split="finetune_train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    in_chans, D, H, W = train_dataset[0].shape
    
    # 2. Model Initialization
    model = VortexMAE(
        in_chans=in_chans,
        out_chans=1,
        mode='segmentation' # Switch to segmentation mode
    ).to(device)
    
    # Load pre-trained encoder weights
    print(f"Loading pre-trained weights from {args.pretrained_ckpt}...")
    checkpoint = torch.load(args.pretrained_ckpt, map_location=device)
    # Filter state dict to only load encoder and shared decoder parts if possible
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 3. Fine-tuning Loop
    best_iou = 0.0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_iou = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Fine-tune]"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Generate IVD-based Ground Truth on the fly for demonstration
            # In a real scenario, these would be pre-calculated and stored in the dataset
            with torch.no_grad():
                gt_ivd = calculate_ivd(batch) # Returns (B, D, H, W)
                # Normalize IVD to [0, 1] for BCE
                gt_mask = (gt_ivd > 0).float().unsqueeze(1) # B, 1, D, H, W
            
            pred_prob = model(batch) # B, 1, D, H, W
            
            loss = vortex_mae_finetune_loss(pred_prob, gt_mask)
            loss.backward()
            optimizer.step()
            if use_tpu:
                xm.mark_step()
            
            epoch_loss += loss.item()
            epoch_iou += calculate_iou(pred_prob, gt_mask).item()
            
        avg_loss = epoch_loss / len(train_loader)
        avg_iou = epoch_iou / len(train_loader)
        scheduler.step()
        
        print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Mean IoU: {avg_iou:.4f}")
        
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'iou': best_iou
            }, os.path.join(args.save_dir, "vortexmae_finetuned_best.pth"))

    print("\nFine-tuning Complete.")

if __name__ == "__main__":
    main()
