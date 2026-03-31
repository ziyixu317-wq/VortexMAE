
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from swin3d import SwinTransformer3D

class VortexMAE(nn.Module):
    """
    VortexMAE implementation consistent with the 2025 paper.
    Architecture: Swin Transformer Encoder + U-Net Transposed Conv Decoder.
    """
    def __init__(self, patch_size=(4, 4, 4), in_chans=3, out_chans=1,
                 embed_dim=64, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                 window_size=(4, 4, 4), mask_ratio=0.25, mode='pretrain',
                 use_checkpoint=False):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.mask_ratio = mask_ratio
        self.mode = mode
        self.use_checkpoint = use_checkpoint
        
        # 1. Swin-ViT Encoder
        self.encoder = SwinTransformer3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, 
            depths=depths, num_heads=num_heads, window_size=window_size
        )
        if self.use_checkpoint:
            for layer in self.encoder.layers:
                layer.use_checkpoint = True
        
        # 2. Mask Token (only used in pre-training)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 1, embed_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        # 3. U-Net Expansive Path (Decoder)
        # Use Upsample + Conv3d instead of ConvTranspose3d to avoid checkerboard artifacts
        d1 = embed_dim * 2
        d2 = embed_dim * 4
        d3 = embed_dim * 8
        
        self.up_stage3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(d3, d2, kernel_size=3, padding=1)
        )
        self.up_stage2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(d2, d1, kernel_size=3, padding=1)
        )
        self.up_stage1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(d1, embed_dim, kernel_size=3, padding=1)
        )
        
        # Final reconstruction layer for pre-training (3-channel velocity)
        self.up_final_rec = nn.Sequential(
            nn.Upsample(scale_factor=patch_size, mode='trilinear', align_corners=False),
            nn.Conv3d(embed_dim, in_chans, kernel_size=3, padding=1)
        )
        
        # Final segmentation head for fine-tuning (out_chans-channel mask, usually 1)
        self.up_final_seg = nn.Sequential(
            nn.Upsample(scale_factor=patch_size, mode='trilinear', align_corners=False),
            nn.Conv3d(embed_dim, out_chans, kernel_size=3, padding=1)
        )

    def forward(self, x):
        """
        x: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        
        # --- 1. Patch Embedding & Masking ---
        x_embed = self.encoder.patch_embed(x)
        Bp, Dp, Hp, Wp, Cp = x_embed.shape
        N = Dp * Hp * Wp
        
        if self.mode == 'pretrain':
            # Create random mask for pre-training only
            noise = torch.rand(B, N, device=x.device)
            mask = (noise < self.mask_ratio).float().view(B, Dp, Hp, Wp, 1) # 1 is masked
            x_masked = x_embed * (1 - mask) + self.mask_token * mask
            x_input = self.encoder.pos_drop(x_masked)
        else:
            # No masking in fine-tuning/inference
            x_input = self.encoder.pos_drop(x_embed)
            mask = None
        
        # --- 2. Encoder Forward ---
        outs = []
        curr_x = x_input
        for layer in self.encoder.layers:
            # Note: BasicLayer3D now handles its own internal block-level checkpointing
            x_layer_out, curr_x = layer(curr_x)
            outs.append(x_layer_out)
        
        # --- 3. U-Net Decoder (Expansive Path) ---
        # Helper for decoder checkpointing
        def upsample_add(z, skip, up_op):
            z = up_op(z)
            sh = skip.shape
            # Handle possible rounding differences in upsampling
            z = z[:, :, :sh[1], :sh[2], :sh[3]]
            return z + skip.permute(0, 4, 1, 2, 3)

        # Stage 3 -> 2
        z = outs[3].permute(0, 4, 1, 2, 3) 
        if self.use_checkpoint:
            from torch.utils.checkpoint import checkpoint
            # Workaround for torch.utils.checkpoint not finding torch.xla in some envs
            if not hasattr(torch, 'xla'):
                try: import torch_xla; torch.xla = torch_xla
                except ImportError: pass
            z = checkpoint(upsample_add, z, outs[2], self.up_stage3, use_reentrant=False)
        else:
            z = upsample_add(z, outs[2], self.up_stage3)
        
        # Stage 2 -> 1
        if self.use_checkpoint:
            z = checkpoint(upsample_add, z, outs[1], self.up_stage2, use_reentrant=False)
        else:
            z = upsample_add(z, outs[1], self.up_stage2)
        
        # Stage 1 -> 0
        if self.use_checkpoint:
            z = checkpoint(upsample_add, z, outs[0], self.up_stage1, use_reentrant=False)
        else:
            z = upsample_add(z, outs[0], self.up_stage1)
        
        # Final layer based on mode
        if self.mode == 'pretrain':
            out = self.up_final_rec(z)
            # Crop to exact input size (Upsample may overshoot)
            out = out[:, :, :D, :H, :W]
            mask_pixel = F.interpolate(mask.permute(0, 4, 1, 2, 3), 
                                       size=(D, H, W), mode='nearest')
            return out, mask_pixel
        else:
            out = self.up_final_seg(z)
            out = out[:, :, :D, :H, :W]
            # Return raw logits for more stable BCEWithLogitsLoss
            return out

def vortex_mae_pretrain_loss(pred, target, mask):
    """
    Strict paper-consistent loss: MSE on masked regions ONLY.
    pred, target: (B, 3, D, H, W)
    mask: (B, 1, D, H, W), 1 means masked
    """
    # Filter only masked pixels
    # Eq. (16): L_pre = 1/|X| * sum_{i in X} ||V_i - V_i_hat||^2
    
    loss = F.mse_loss(pred * mask, target * mask, reduction='sum')
    return loss / (mask.sum() * target.shape[1] + 1e-8)
