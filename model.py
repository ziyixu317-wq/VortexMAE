
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
    def __init__(self, patch_size=(2, 4, 4), in_chans=3, 
                 embed_dim=48, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], 
                 window_size=(4, 4, 4), mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        
        # 1. Swin-ViT Encoder
        self.encoder = SwinTransformer3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, 
            depths=depths, num_heads=num_heads, window_size=window_size
        )
        
        # 2. Mask Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 1, embed_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        # 3. U-Net Expansive Path (Decoder)
        # The encoder produces 4 hierarchical feature maps (outs).
        # We use transposed convolutions to upscale and combine them.
        
        # Dimensions at each stage:
        # Stage 0: embed_dim
        # Stage 1: embed_dim * 2
        # Stage 2: embed_dim * 4
        # Stage 3: embed_dim * 8 (final encoder output)
        
        d1 = embed_dim * 2
        d2 = embed_dim * 4
        d3 = embed_dim * 8
        
        self.up_stage3 = nn.ConvTranspose3d(d3, d2, kernel_size=2, stride=2)
        self.up_stage2 = nn.ConvTranspose3d(d2, d1, kernel_size=2, stride=2)
        self.up_stage1 = nn.ConvTranspose3d(d1, embed_dim, kernel_size=2, stride=2)
        
        # Final reconstruction layer (transposed patch embedding)
        self.up_final = nn.ConvTranspose3d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        x: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        
        # --- 1. Patch Embedding & Masking ---
        x_embed = self.encoder.patch_embed(x)
        Bp, Dp, Hp, Wp, Cp = x_embed.shape
        N = Dp * Hp * Wp
        
        # Create random mask
        noise = torch.rand(B, N, device=x.device)
        mask = (noise < self.mask_ratio).float().view(B, Dp, Hp, Wp, 1) # 1 is masked
        
        # Replace masked tokens with mask_token
        x_masked = x_embed * (1 - mask) + self.mask_token * mask
        x_input = self.encoder.pos_drop(x_masked)
        
        # --- 2. Encoder Forward ---
        # Get hierarchical features for skip connections
        outs = []
        curr_x = x_input
        for layer in self.encoder.layers:
            x_layer_out, curr_x = layer(curr_x)
            outs.append(x_layer_out)
        
        # --- 3. U-Net Decoder (Expansive Path) ---
        # outs[3] is Stage 3 output (deepest)
        # outs[2] is Stage 2 ...
        
        # Stage 3 -> 2
        z = outs[3].permute(0, 4, 1, 2, 3) # B, C, D, H, W
        z = self.up_stage3(z)
        z = z + outs[2].permute(0, 4, 1, 2, 3) # Simple skip add (or concat if paper specifies)
        
        # Stage 2 -> 1
        z = self.up_stage2(z)
        z = z + outs[1].permute(0, 4, 1, 2, 3)
        
        # Stage 1 -> 0
        z = self.up_stage1(z)
        z = z + outs[0].permute(0, 4, 1, 2, 3)
        
        # Final Reconstruction
        x_rec = self.up_final(z)
        
        # Return reconstructed volume and the binary mask (expanded to pixel level)
        mask_pixel = F.interpolate(mask.permute(0, 4, 1, 2, 3), 
                                   size=(D, H, W), mode='nearest')
        
        return x_rec, mask_pixel

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
