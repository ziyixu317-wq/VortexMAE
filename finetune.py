import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np

from dataset import VortexMAEDataset
from model import VortexMAE
from vortex_utils import vortex_mae_paper_loss, calculate_iou, calculate_ivd

# TPU Support
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_backend
    import torch_xla.distributed.parallel_loader as pl
    IS_TPU = True
except ImportError:
    IS_TPU = False

def setup_ddp():
    """Initialize DDP environment for torchrun (handles GPU/TPU)."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        backend = "xla" if IS_TPU else "nccl"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
        
        if IS_TPU:
            device = xm.xla_device()
        else:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
    else:
        # Single device fallback
        rank = 0
        world_size = 1
        local_rank = 0
        if IS_TPU:
            device = xm.xla_device()
            if not dist.is_initialized():
                 dist.init_process_group(backend="xla", init_method="tcp://127.0.0.1:23457", world_size=1, rank=0)
        elif torch.cuda.is_available():
            device = torch.device("cuda:0")
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:23457", world_size=1, rank=0)
        else:
            device = torch.device("cpu")
            if not dist.is_initialized():
                dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:23457", world_size=1, rank=0)
                
    return rank, world_size, local_rank, device

def main():
    parser = argparse.ArgumentParser(description="VortexMAE Fine-tuning (DDP T4*2 Optimized)")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .vti files")
    parser.add_argument("--pretrained_ckpt", type=str, required=True, help="Path to pre-trained checkpoint")
    parser.add_argument("--batch_size", type=int, default=8, help="TOTAL batch size across all GPUs")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_finetune", help="Save directory")
    parser.add_argument("--pos_weight", type=float, default=2.0, help="Positive class weight for paper loss")
    
    args = parser.parse_args()
    
    # 1. DDP Setup
    rank, world_size, local_rank, device = setup_ddp()
    
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"DDP Initialized: Rank {rank}/{world_size} on {device}")
        print(f"Config: Total Batch Size={args.batch_size}, LR={args.lr}")

    # 2. Dataset & Sampler
    train_dataset = VortexMAEDataset(args.data_dir, split="finetune_train")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    batch_size_per_gpu = args.batch_size // world_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size_per_gpu, sampler=train_sampler, num_workers=4)
    
    in_chans, D, H, W = train_dataset[0].shape
    
    # 3. Model Initialization
    model = VortexMAE(
        in_chans=in_chans,
        out_chans=1,
        mode='segmentation'
    )
    
    # Load pre-trained encoder weights
    if rank == 0:
        print(f"Loading pre-trained weights from {args.pretrained_ckpt}...")
    checkpoint = torch.load(args.pretrained_ckpt, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Strip 'module.' prefix if present (e.g., if checkpoint was saved from a DataParallel or DDP model)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    
    # SyncBatchNorm and DDP
    if not IS_TPU:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        model = DDP(model, gradient_as_bucket_view=True)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.99))
    scheduler = StepLR(optimizer, step_size=100, gamma=0.8)
    
    # 4. Fine-tuning Loop
    best_iou = 0.0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch) # Important for shuffling to be different across epochs
        epoch_loss = torch.tensor(0.0).to(device)
        epoch_iou = torch.tensor(0.0).to(device)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Fine-tune]", disable=(rank != 0))
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Ground Truth Mask calculation
            with torch.no_grad():
                gt_ivd = calculate_ivd(batch)
                gt_mask = (gt_ivd > 0).float().unsqueeze(1)
            
            # Forward
            pred_logits = model(batch)
            loss = vortex_mae_paper_loss(pred_logits, gt_mask, pos_weight=args.pos_weight)
            
            loss.backward()
            
            if IS_TPU:
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()
            
            epoch_loss += loss.detach()
            pred_prob = torch.sigmoid(pred_logits)
            epoch_iou += calculate_iou(pred_prob, gt_mask).detach()
            
        # Aggregate metrics from all GPUs
        dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_iou, op=dist.ReduceOp.SUM)
        
        # Calculate average over all samples processed by all GPUs
        avg_loss = epoch_loss.item() / (len(train_loader) * world_size)
        avg_iou = epoch_iou.item() / (len(train_loader) * world_size)
        scheduler.step()
        
        if rank == 0:
            print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Mean IoU: {avg_iou:.4f}")
            if avg_iou > best_iou:
                best_iou = avg_iou
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(), # Save unwrapped model state_dict
                    'iou': best_iou
                }, os.path.join(args.save_dir, "vortexmae_finetuned_best.pth"))
                print(f" -> Saved best fine-tuned checkpoint (IoU: {best_iou:.4f})")

    if rank == 0:
        print("\nFine-tuning Complete.")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
