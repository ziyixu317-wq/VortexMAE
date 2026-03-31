
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
    parser = argparse.ArgumentParser(description="VortexMAE Fine-tuning (GPU/TPU)")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .vti files")
    parser.add_argument("--pretrained_ckpt", type=str, required=True, help="Path to pre-trained checkpoint")
    parser.add_argument("--batch_size", type=int, default=8, help="TOTAL batch size")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_finetune", help="Save directory")
    parser.add_argument("--pos_weight", type=float, default=2.0, help="Positive class weight")
    parser.add_argument("--use_checkpoint", action="store_true", help="Use gradient checkpointing")
    return parser.parse_args()

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
        return 0, 1, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    args = parse_args()
    
    # 2. Strategy Setup
    if IS_TPU:
        device = xm.xla_device()
        rank = 0
        world_size = 1
        local_rank = 0
        num_cores = xr.global_runtime_device_count()
        mesh = xs.Mesh(np.arange(num_cores), (num_cores,), ('data',))
        if rank == 0:
            print(f"TPU SPMD Mode: Controlling {num_cores} cores from 1 process.")
    else:
        rank, world_size, local_rank, device = setup_gpu_ddp()
        if rank == 0:
            print(f"GPU DDP Mode: Rank {rank}/{world_size} on {device}")

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    # 3. Dataset & Sampler
    train_dataset = VortexMAEDataset(args.data_dir, split="finetune_train")
    if IS_TPU:
        train_sampler = None
        shuffle_train = True
    else:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle_train = False

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size if IS_TPU else args.batch_size // world_size, 
                              sampler=train_sampler, shuffle=shuffle_train, num_workers=4, drop_last=True)
    
    in_chans, D, H, W = train_dataset[0].shape
    
    # 4. Model Initialization
    model = VortexMAE(in_chans=in_chans, out_chans=1, mode='segmentation', 
                      use_checkpoint=args.use_checkpoint if not IS_TPU else True).to(device)
    
    if IS_TPU:
        model = model.to(torch.bfloat16)
    
    # Load pre-trained weights
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
    
    if not IS_TPU and world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.8)
    
    # 5. Fine-tuning Loop
    best_iou = 0.0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        epoch_iou = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(rank != 0))
        for batch in pbar:
            batch = batch.to(device)
            if IS_TPU:
                xs.mark_sharding(batch, mesh, ('data', None, None, None, None))
                
            optimizer.zero_grad()
            
            with torch.no_grad():
                gt_ivd = calculate_ivd(batch)
                gt_mask = (gt_ivd > 0).float().unsqueeze(1)
            
            if IS_TPU:
                with torch.autocast(device_type='xla', dtype=torch.bfloat16):
                    pred_logits = model(batch)
                    # Cast gt_mask to bfloat16 for TPU compatibility
                    loss = vortex_mae_paper_loss(pred_logits, gt_mask.to(torch.bfloat16), pos_weight=args.pos_weight)
                loss.backward()
                xm.optimizer_step(optimizer)
            else:
                pred_logits = model(batch)
                loss = vortex_mae_paper_loss(pred_logits, gt_mask, pos_weight=args.pos_weight)
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            pred_prob = torch.sigmoid(pred_logits)
            epoch_iou += calculate_iou(pred_prob, gt_mask).item()
            num_batches += 1
            
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_iou = epoch_iou / max(num_batches, 1)
        scheduler.step()
        
        if rank == 0:
            print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | IoU: {avg_iou:.4f}")
            if avg_iou > best_iou:
                best_iou = avg_iou
                ckpt_path = os.path.join(args.save_dir, "vortexmae_finetuned_best.pth")
                raw_model = model.module if hasattr(model, 'module') else model
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': raw_model.state_dict(),
                    'iou': best_iou
                }
                if IS_TPU:
                    xm.save(save_dict, ckpt_path)
                else:
                    torch.save(save_dict, ckpt_path)
                print(f" -> Best IoU: {best_iou:.4f}, Saved.")

    if not IS_TPU and world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
