import torch
import torch.nn.functional as F

def get_velocity_gradient(u_tensor, dx=1.0, dy=1.0, dz=1.0):
    """
    u_tensor: (B, 3, D, H, W)
    Returns grad_u: (B, 3, 3, D, H, W)
    grad_u[:, i, j] = \partial u_i / \partial x_j
    """
    B, _, D, H, W = u_tensor.shape
    device = u_tensor.device
    
    u, v, w = u_tensor[:, 0], u_tensor[:, 1], u_tensor[:, 2]
    
    def central_diff(f, axis, h):
        if axis == 'x': # W dimension (last)
            pad = F.pad(f.unsqueeze(1), (1, 1, 0, 0, 0, 0), mode='replicate').squeeze(1)
            return (pad[..., 2:] - pad[..., :-2]) / (2 * h)
        elif axis == 'y': # H dimension
            pad = F.pad(f.unsqueeze(1), (0, 0, 1, 1, 0, 0), mode='replicate').squeeze(1)
            return (pad[..., 2:, :] - pad[..., :-2, :]) / (2 * h)
        elif axis == 'z': # D dimension
            pad = F.pad(f.unsqueeze(1), (0, 0, 0, 0, 1, 1), mode='replicate').squeeze(1)
            return (pad[:, 2:, :, :] - pad[:, :-2, :, :]) / (2 * h)
            
    grad = torch.zeros((B, 3, 3, D, H, W), device=device)
    
    # \nabla u
    grad[:, 0, 0] = central_diff(u, 'x', dx)
    grad[:, 0, 1] = central_diff(u, 'y', dy)
    grad[:, 0, 2] = central_diff(u, 'z', dz)
    
    # \nabla v
    grad[:, 1, 0] = central_diff(v, 'x', dx)
    grad[:, 1, 1] = central_diff(v, 'y', dy)
    grad[:, 1, 2] = central_diff(v, 'z', dz)
    
    # \nabla w
    grad[:, 2, 0] = central_diff(w, 'x', dx)
    grad[:, 2, 1] = central_diff(w, 'y', dy)
    grad[:, 2, 2] = central_diff(w, 'z', dz)
    
    return grad

def calculate_ivd(u_tensor, dx=1.0, dy=1.0, dz=1.0):
    """
    Isolation by Vorticity Deviation (IVD) - Haller et al. 2016
    Simplified version: Local vorticity deviation from mean.
    """
    grad_u = get_velocity_gradient(u_tensor, dx, dy, dz)
    
    omega_x = grad_u[:, 2, 1] - grad_u[:, 1, 2]
    omega_y = grad_u[:, 0, 2] - grad_u[:, 2, 0]
    omega_z = grad_u[:, 1, 0] - grad_u[:, 0, 1]
    
    vorticity_mag = torch.sqrt(omega_x**2 + omega_y**2 + omega_z**2 + 1e-8)
    
    # IVD: deviation from spatial mean
    mean_vort = torch.mean(vorticity_mag, dim=(1, 2, 3), keepdim=True)
    ivd = vorticity_mag - mean_vort
    
    return ivd

def vortex_mae_finetune_loss(pred_logits, target_mask, pos_weight=10.0):
    """
    Combined Loss for Vortex Segmentation: Weighted BCE + Dice Loss
    """
    # 1. Weighted BCE with Logits
    # pos_weight helps with the extreme sparsity of vortex regions
    weight = torch.tensor([pos_weight], device=pred_logits.device)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target_mask, pos_weight=weight)
    
    # 2. Dice Loss (ignores background dominance)
    pred_prob = torch.sigmoid(pred_logits)
    smooth = 1e-6
    intersection = (pred_prob * target_mask).sum()
    dice_loss = 1 - (2. * intersection + smooth) / (pred_prob.sum() + target_mask.sum() + smooth)
    
    return 0.5 * bce + 0.5 * dice_loss

def calculate_iou(pred_mask, gt_mask, threshold=0.5):
    """
    Calculate Intersection over Union for vortex masks.
    """
    pred = (pred_mask > threshold).float()
    intersection = (pred * gt_mask).sum()
    union = pred.sum() + gt_mask.sum() - intersection
    return (intersection + 1e-8) / (union + 1e-8)
