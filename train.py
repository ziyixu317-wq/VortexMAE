
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

# 1. Hardware Detection
IS_TPU = (os.environ.get('TPU_NAME') is not None or 
          os.environ.get('TPU_ACCELERATOR_TYPE') is not None or
          os.environ.get('KAGGLE_TPU_VERSION') is not None)

if IS_TPU:
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        import torch_xla.distributed.spmd as xs
        # CRITICAL: Enable SPMD before any XLA operations
        xr.use_spmd()
    except ImportError:
        IS_TPU = False

def parse_args():
    parser = argparse.ArgumentParser(description="VortexMAE Pre-training (GPU/TPU)")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .vti files")
    parser.add_argument("--batch_size", type=int, default=8, help="TOTAL batch size")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_pretrain", help="Save directory")
    parser.add_argument("--mask_ratio", type=float, default=0.25, help="MAE mask ratio")
    parser.add_argument("--use_checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    return parser.parse_args()

def print_tpu_memory():
    """Print memory usage for all 8 TPU cores."""
    if IS_TPU:
        # PjRt: get_memory_info(device) provides HBM usage
        device = xm.xla_device()
        info = xm.get_memory_info(device)
        # Convert to GB
        kb = 1024.0
        used = (info['kb_total'] - info['kb_free']) / (kb * 1024.0)
        total = info['kb_total'] / (kb * 1024.0)
        print(f" [TPU Memory] Core 0: {used:.2f}GB / {total:.2f}GB")
        # To see all 8 cores in SPMD, you can just assume they are similar 
        # as the workload is sharded equally.

def setup_gpu_ddp():
    """Initialize DDP for GPU."""
    if 'RANK' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return rank, world_size, local_rank, device
    else:
        # Single GPU fallback
        return 0, 1, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    args = parse_args()
    
    # 2. Strategy Setup
    if IS_TPU:
        device = xm.xla_device()
        rank = 0  # Single process in SPMD
        world_size = 1 
        local_rank = 0
        num_cores = xr.global_runtime_device_count()
        # Define 1D mesh for data parallelism
        mesh = xs.Mesh(np.arange(num_cores), (num_cores,), ('data',))
        if rank == 0:
            print(f"TPU SPMD Mode: Controlling {num_cores} cores from 1 process.")
    else:
        rank, world_size, local_rank, device = setup_gpu_ddp()
        if rank == 0:
            print(f"GPU DDP Mode: Rank {rank}/{world_size} on {device}")

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    # 3. Datasets
    train_dataset = VortexMAEDataset(args.data_dir, split="pretrain_train")
    test_dataset = VortexMAEDataset(args.data_dir, split="pretrain_eval")
    
    # SPMD uses single sampler, DDP uses DistributedSampler
    if IS_TPU:
        train_sampler = None
        test_sampler = None
        shuffle_train = True
    else:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle_train = False

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size if IS_TPU else args.batch_size // world_size, 
                              sampler=train_sampler, shuffle=shuffle_train, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size if IS_TPU else args.batch_size // world_size, 
                             sampler=test_sampler, num_workers=4, drop_last=True)
    
    in_chans, D, H, W = train_dataset[0].shape
    
    # 4. Model
    model = VortexMAE(
        in_chans=in_chans,
        mask_ratio=args.mask_ratio,
        embed_dim=64,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        use_checkpoint=args.use_checkpoint if not IS_TPU else True # Auto-enable on TPU
    ).to(device)
    
    if IS_TPU:
        model = model.to(torch.bfloat16)
    
    if not IS_TPU and world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.8)
    
    # 5. Loop
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        train_loss_val = 0.0
        num_batches = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            # SPMD: Shard batch across cores
            if IS_TPU:
                xs.mark_sharding(batch, mesh, ('data', None, None, None, None))
            
            # Use autocast for TPU to save memory (HBM)
            if IS_TPU:
                with torch.autocast(device_type='xla', dtype=torch.bfloat16):
                    x_rec, mask = model(batch)
                    loss = vortex_mae_pretrain_loss(x_rec, batch, mask)
                    loss = loss / args.accumulation_steps
                loss.backward()
                if (num_batches + 1) % args.accumulation_steps == 0:
                    xm.optimizer_step(optimizer)
                    optimizer.zero_grad()
            else:
                x_rec, mask = model(batch)
                loss = vortex_mae_pretrain_loss(x_rec, batch, mask)
                loss = loss / args.accumulation_steps
                loss.backward()
                if (num_batches + 1) % args.accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
            train_loss_val += loss.item()
            num_batches += 1
            
        avg_train_loss = train_loss_val / max(num_batches, 1)
        
        # Test Loop
        model.eval()
        test_loss_val = 0.0
        test_psnr_val = 0.0
        num_test_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                if IS_TPU:
                    xs.mark_sharding(batch, mesh, ('data', None, None, None, None))
                    
                x_rec, mask = model(batch)
                loss = vortex_mae_pretrain_loss(x_rec, batch, mask)
                psnr = calculate_psnr(x_rec, batch)
                
                test_loss_val += loss.item()
                test_psnr_val += psnr.item()
                num_test_batches += 1
        
        avg_test_loss = test_loss_val / max(num_test_batches, 1)
        avg_test_psnr = test_psnr_val / max(num_test_batches, 1)
        
        scheduler.step()
        
        if rank == 0:
            print(f"Epoch {epoch} | Train: {avg_train_loss:.6f} | Test: {avg_test_loss:.6f} | PSNR: {avg_test_psnr:.2f}dB")
            if IS_TPU:
                print_tpu_memory()
            
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                ckpt_path = os.path.join(args.save_dir, "vortexmae_best.pth")
                raw_model = model.module if hasattr(model, 'module') else model
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': raw_model.state_dict(),
                    'loss': best_loss
                }
                if IS_TPU:
                    xm.save(save_dict, ckpt_path)
                else:
                    torch.save(save_dict, ckpt_path)
                print(f" -> Best Checkpoint Saved.")

    if not IS_TPU and world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
