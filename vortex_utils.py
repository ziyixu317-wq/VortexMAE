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
    ivd_val = vorticity_mag - mean_vort
    
    return ivd_val

def calculate_q_criterion(u_tensor, dx=1.0, dy=1.0, dz=1.0):
    """
    Q-criterion: Second invariant of the velocity gradient tensor.
    Q = 0.5 * (||Omega||^2 - ||S||^2) > 0 defines a vortex region.
    """
    grad_u = get_velocity_gradient(u_tensor, dx, dy, dz) # (B, 3, 3, D, H, W)
    
    # Calculate Frobenius norms squared of S and Omega
    s_norm_sq = 0
    omega_norm_sq = 0
    
    for i in range(3):
        for j in range(3):
            s_ij = 0.5 * (grad_u[:, i, j] + grad_u[:, j, i])
            omega_ij = 0.5 * (grad_u[:, i, j] - grad_u[:, j, i])
            s_norm_sq += s_ij**2
            omega_norm_sq += omega_ij**2
            
    q = 0.5 * (omega_norm_sq - s_norm_sq)
    return q

def vortex_mae_finetune_loss(pred_logits, target_mask, pos_weight=1.5):
    """
    Current implementation: Weighted BCE + Dice Loss.
    Note: See vortex_mae_paper_loss for the exact Eq. 20-22 version.
    """
    weight = torch.tensor([pos_weight], device=pred_logits.device)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target_mask, pos_weight=weight)
    
    pred_prob = torch.sigmoid(pred_logits)
    smooth = 1e-6
    intersection = (pred_prob * target_mask).sum()
    dice_loss = 1 - (2. * intersection + smooth) / (pred_prob.sum() + target_mask.sum() + smooth)
    
    return 0.5 * bce + 0.5 * dice_loss

def vortex_mae_paper_loss(pred_logits, target_mask, alpha=1.0, beta=1.0):
    """
    Paper-Consistent Loss (Eq. 20-22): alpha * L_BCE + beta * L_L2 (MSE).
    """
    # L_BCE (Eq. 20)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target_mask)
    
    # L_L2 (Eq. 21) - Per-voxel L2 is equivalent to Mean Squared Error 
    pred_prob = torch.sigmoid(pred_logits)
    mse = F.mse_loss(pred_prob, target_mask)
    
    return alpha * bce + beta * mse

def calculate_iou(pred_mask, gt_mask, threshold=0.5):
    """
    Calculate Intersection over Union for vortex masks.
    """
    pred = (pred_mask > threshold).float()
    intersection = (pred * gt_mask).sum()
    union = pred.sum() + gt_mask.sum() - intersection
    return (intersection + 1e-8) / (union + 1e-8)
