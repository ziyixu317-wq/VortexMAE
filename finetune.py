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

# TPU Support Detection (Environment-based)
IS_TPU = (os.environ.get('TPU_NAME') is not None or 
          os.environ.get('TPU_ACCELERATOR_TYPE') is not None or
          os.environ.get('KAGGLE_TPU_VERSION') is not None)

if IS_TPU:
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.xla_multiprocessing as xmp
    except ImportError:
        IS_TPU = False

def parse_args():
    parser = argparse.ArgumentParser(description="VortexMAE Fine-tuning (GPU/TPU)")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .vti files")
    parser.add_argument("--pretrained_ckpt", type=str, required=True, help="Path to pre-trained checkpoint")
    parser.add_argument("--batch_size", type=int, default=8, help="TOTAL batch size across all devices")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_finetune", help="Save directory")
    parser.add_argument("--pos_weight", type=float, default=2.0, help="Positive class weight")
    return parser.parse_args()

# ============================================================
# GPU Path: launched via torchrun
# ============================================================
def setup_ddp_gpu():
    """Initialize DDP for GPU via torchrun."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, local_rank, device

def main_gpu():
    """GPU entry point (called by torchrun)."""
    args = parse_args()
    rank, world_size, local_rank, device = setup_ddp_gpu()
    _finetune_loop(args, rank, world_size, local_rank, device, is_tpu=False)

# ============================================================
# TPU Path: launched via xmp.spawn
# ============================================================
def _mp_fn(index, args):
    """TPU worker function. index = 0..7, assigned by xmp.spawn."""
    device = xm.xla_device()
    rank = xm.get_ordinal()
    world_size = xm.xrt_world_size()
    _finetune_loop(args, rank, world_size, rank, device, is_tpu=True)

# ============================================================
# Shared Fine-tuning Loop
# ============================================================
def _finetune_loop(args, rank, world_size, local_rank, device, is_tpu):
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Fine-tuning on {'TPU' if is_tpu else 'GPU'}: Rank {rank}/{world_size} on {device}")
        print(f"Config: Total Batch Size={args.batch_size}, LR={args.lr}")

    # 1. Dataset & Sampler
    train_dataset = VortexMAEDataset(args.data_dir, split="finetune_train")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    batch_size_per_device = args.batch_size // world_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size_per_device, sampler=train_sampler, num_workers=4, drop_last=True)
    
    in_chans, D, H, W = train_dataset[0].shape
    
    # 2. Model
    model = VortexMAE(in_chans=in_chans, out_chans=1, mode='segmentation')
    
    # Load pre-trained encoder weights
    if rank == 0:
        print(f"Loading pre-trained weights from {args.pretrained_ckpt}...")
    checkpoint = torch.load(args.pretrained_ckpt, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    
    if is_tpu:
        pass  # xm.optimizer_step handles gradient sync
    else:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.99))
    scheduler = StepLR(optimizer, step_size=100, gamma=0.8)
    
    # 3. Fine-tuning Loop
    best_iou = 0.0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_loss = torch.tensor(0.0, device=device)
        epoch_iou = torch.tensor(0.0, device=device)
        num_batches = 0
        
        if is_tpu:
            effective_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
        else:
            effective_loader = train_loader
        
        for batch in effective_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                gt_ivd = calculate_ivd(batch)
                gt_mask = (gt_ivd > 0).float().unsqueeze(1)
            
            pred_logits = model(batch)
            loss = vortex_mae_paper_loss(pred_logits, gt_mask, pos_weight=args.pos_weight)
            
            loss.backward()
            
            if is_tpu:
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()
            
            epoch_loss += loss.detach()
            pred_prob = torch.sigmoid(pred_logits)
            epoch_iou += calculate_iou(pred_prob, gt_mask).detach()
            num_batches += 1
            
        # Aggregate metrics
        if is_tpu:
            loss_reduced = xm.mesh_reduce('ft_loss', epoch_loss.item(), lambda x: sum(x))
            iou_reduced = xm.mesh_reduce('ft_iou', epoch_iou.item(), lambda x: sum(x))
            total_batches = xm.mesh_reduce('ft_batches', num_batches, lambda x: sum(x))
            avg_loss = loss_reduced / max(total_batches, 1)
            avg_iou = iou_reduced / max(total_batches, 1)
        else:
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_iou, op=dist.ReduceOp.SUM)
            avg_loss = epoch_loss.item() / (len(train_loader) * world_size)
            avg_iou = epoch_iou.item() / (len(train_loader) * world_size)
        
        scheduler.step()
        
        is_master = xm.is_master_ordinal() if is_tpu else (rank == 0)
        
        if is_master:
            print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Mean IoU: {avg_iou:.4f}")
            if avg_iou > best_iou:
                best_iou = avg_iou
                checkpoint_model = model.module if hasattr(model, 'module') else model
                ckpt_data = {
                    'epoch': epoch,
                    'model_state_dict': checkpoint_model.state_dict(),
                    'iou': best_iou
                }
                ckpt_path = os.path.join(args.save_dir, "vortexmae_finetuned_best.pth")
                if is_tpu:
                    xm.save(ckpt_data, ckpt_path)
                else:
                    torch.save(ckpt_data, ckpt_path)
                print(f" -> Saved best fine-tuned checkpoint (IoU: {best_iou:.4f})")

    if is_tpu:
        if xm.is_master_ordinal():
            print("\nFine-tuning Complete.")
    else:
        if rank == 0:
            print("\nFine-tuning Complete.")
        dist.destroy_process_group()

# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    if IS_TPU:
        args = parse_args()
        xmp.spawn(_mp_fn, args=(args,))
    else:
        main_gpu()
