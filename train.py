
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
import pyvista as pv

from dataset import VortexMAEDataset
from model import VortexMAE, vortex_mae_pretrain_loss
from vortex_utils import calculate_psnr

# TPU Support Detection (Environment-based to avoid top-level hardware race conditions)
IS_TPU = (os.environ.get('TPU_NAME') is not None or 
          os.environ.get('TPU_ACCELERATOR_TYPE') is not None or
          os.environ.get('KAGGLE_TPU_VERSION') is not None)

if IS_TPU:
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.xla_backend # Register XLA backend at top level
    except ImportError:
        IS_TPU = False

def setup_ddp():
    """Initialize DDP environment for torchrun (handles GPU/TPU)."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        if IS_TPU:
            # Mandatory PjRt env vars for torchrun parity
            os.environ['PJRT_DEVICE'] = 'TPU'
            os.environ['PJRT_LOCAL_PROCESS_RANK'] = str(local_rank)
            os.environ['PJRT_LOCAL_DEVICE_COUNT'] = '8'
            
            # 1. Initialize XLA backend
            if not dist.is_initialized():
                dist.init_process_group(backend="xla", init_method="env://")
            # 2. Get device
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
        else:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="env://")
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
    else:
        # Single device fallback
        rank = 0
        world_size = 1
        local_rank = 0
        if IS_TPU:
            os.environ['PJRT_DEVICE'] = 'TPU'
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            if not dist.is_initialized():
                 dist.init_process_group(backend="xla", init_method="tcp://127.0.0.1:23456", world_size=1, rank=0)
        elif torch.cuda.is_available():
            device = torch.device("cuda:0")
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:23456", world_size=1, rank=0)
        else:
            device = torch.device("cpu")
            if not dist.is_initialized():
                dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:23456", world_size=1, rank=0)
                
    return rank, world_size, local_rank, device

def main():
    parser = argparse.ArgumentParser(description="VortexMAE Pre-training (DDP T4*2 Optimized)")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .vti files")
    parser.add_argument("--batch_size", type=int, default=16, help="TOTAL batch size across all GPUs")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (SR-scaled for Batch 16)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_pretrain", help="Save directory")
    
    args = parser.parse_args()
    
    # 1. DDP Setup
    rank, world_size, local_rank, device = setup_ddp()
    
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"DDP Initialized: Rank {rank}/{world_size} on {device}")
        print(f"Config: Total Batch Size={args.batch_size}, LR={args.lr}")

    # 2. Datasets & Samplers
    train_dataset = VortexMAEDataset(args.data_dir, split="pretrain_train")
    test_dataset = VortexMAEDataset(args.data_dir, split="pretrain_eval")
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # Per-GPU batch size
    batch_size_per_gpu = args.batch_size // world_size
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size_per_gpu, sampler=train_sampler, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_per_gpu, sampler=test_sampler, num_workers=4)
    
    # Get spatial dimensions
    sample = train_dataset[0]
    in_chans, D, H, W = sample.shape
    
    # 3. Model Initialization
    model = VortexMAE(
        in_chans=in_chans,
        mask_ratio=args.mask_ratio if hasattr(args, 'mask_ratio') else 0.25,
        embed_dim=48,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24]
    ).to(device)
    
    # SyncBatchNorm
    if not IS_TPU:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    else:
        # Wrap for TPU DDP if using torch.distributed backend
        model = DDP(model, gradient_as_bucket_view=True)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.99))
    scheduler = StepLR(optimizer, step_size=100, gamma=0.8)
    
    # 4. Training Loop
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss = torch.tensor(0.0).to(device)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", disable=(rank != 0))
        
        # TPU Parallel Loader
        if IS_TPU:
            import torch_xla.distributed.parallel_loader as pl
            effective_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
        else:
            effective_loader = train_loader

        for batch in effective_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            x_rec, mask = model(batch)
            loss = vortex_mae_pretrain_loss(x_rec, batch, mask)
            
            loss.backward()
            
            if IS_TPU:
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()
                
            train_loss += loss.detach()
            
        # Synchronize loss across all GPUs
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        avg_train_loss = train_loss.item() / (len(train_loader) * world_size)
        
        # Evaluation
        model.eval()
        test_loss = torch.tensor(0.0).to(device)
        test_psnr = torch.tensor(0.0).to(device)
        
        if IS_TPU:
            import torch_xla.distributed.parallel_loader as pl
            effective_test_loader = pl.ParallelLoader(test_loader, [device]).per_device_loader(device)
        else:
            effective_test_loader = test_loader

        with torch.no_grad():
            for batch in effective_test_loader:
                batch = batch.to(device)
                x_rec, mask = model(batch)
                loss = vortex_mae_pretrain_loss(x_rec, batch, mask)
                test_loss += loss.detach()
                test_psnr += calculate_psnr(x_rec, batch).detach()
        
        dist.all_reduce(test_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(test_psnr, op=dist.ReduceOp.SUM)
        
        avg_test_loss = test_loss.item() / (len(test_loader) * world_size)
        avg_test_psnr = test_psnr.item() / (len(test_loader) * world_size)
        
        scheduler.step()
        
        if rank == 0:
            print(f"Epoch {epoch} | Train MSE: {avg_train_loss:.6f} | Test MSE: {avg_test_loss:.6f} | Test PSNR: {avg_test_psnr:.2f} dB")
            
            # Checkpointing
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                ckpt_path = os.path.join(args.save_dir, "vortexmae_best.pth")
                # Save unwrapped model
                checkpoint_model = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': checkpoint_model.state_dict(),
                    'loss': best_loss
                }, ckpt_path)
                print(f" -> Saved best checkpoint: {ckpt_path}")
            
            # Visualization (Rank 0 only)
            if epoch % 10 == 0 or epoch == args.epochs:
                with torch.no_grad():
                    sample_batch = next(iter(test_loader)).to(device)
                    x_rec, mask = model(sample_batch)
                    pred = x_rec[0].cpu().numpy()
                    gt = sample_batch[0].cpu().numpy()
                    
                    _, vD, vH, vW = pred.shape
                    try:
                        ch_min = test_dataset.ch_min.squeeze()
                        ch_max = test_dataset.ch_max.squeeze()
                        for c in range(in_chans):
                            pred[c] = pred[c] * (ch_max[c] - ch_min[c]) + ch_min[c]
                            gt[c] = gt[c] * (ch_max[c] - ch_min[c]) + ch_min[c]
                    except: pass
                    
                    # Remove padding
                    valid_d = vD
                    for d in range(vD - 1, -1, -1):
                        if np.abs(gt[:, d, :, :]).sum() > 1e-6:
                            valid_d = d + 1
                            break
                    pred = pred[:, :valid_d, :, :]
                    gt = gt[:, :valid_d, :, :]
                    
                    vis_mesh = pv.ImageData()
                    vis_mesh.dimensions = (vW, vH, valid_d)
                    for i, name in enumerate(["u", "v", "w"]):
                        vis_mesh.point_data[f"{name}_rec"] = pred[i].flatten(order='C')
                        vis_mesh.point_data[f"{name}_gt"] = gt[i].flatten(order='C')
                    
                    out_path = os.path.join(args.save_dir, f"recon_epoch_{epoch}.vti")
                    vis_mesh.save(out_path)
                    print(f" -> Saved reconstruction to {out_path}")

    if rank == 0:
        print("\nPre-training Complete.")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
