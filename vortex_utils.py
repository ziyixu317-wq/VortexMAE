def calculate_ivd(u_tensor, dx=1.0, dy=1.0, dz=1.0):
    """
    Isolation by Vorticity Deviation (IVD) - Haller et al. 2016
    Simplified version: Local vorticity deviation from mean.
    """
    grad_u = get_velocity_gradient(u_tensor, dx, dy, dz)
    # Vorticity vector omega = curl(u)
    # omega_x = dw/dy - dv/dz
    # omega_y = du/dz - dw/dx
    # omega_z = dv/dx - du/dy
    
    omega_x = grad_u[:, 2, 1] - grad_u[:, 1, 2]
    omega_y = grad_u[:, 0, 2] - grad_u[:, 2, 0]
    omega_z = grad_u[:, 1, 0] - grad_u[:, 0, 1]
    
    vorticity_mag = torch.sqrt(omega_x**2 + omega_y**2 + omega_z**2 + 1e-8)
    
    # IVD: deviation from spatial mean
    # IVD(x) = |omega(x)| - mean(|omega|)
    mean_vort = torch.mean(vorticity_mag, dim=(1, 2, 3), keepdim=True)
    ivd = vorticity_mag - mean_vort
    
    # Return as a binary mask or soft mask
    return ivd

def vortex_mae_finetune_loss(pred_prob, target_mask):
    """
    Eq. (22) in paper: L = alpha * L_BCE + beta * L_L2
    Alpha and Beta are usually 0.5.
    pred_prob: sigmoid output (B, 1, D, H, W)
    target_mask: binary ground truth (B, 1, D, H, W)
    """
    bce = F.binary_cross_entropy(pred_prob, target_mask)
    l2 = F.mse_loss(pred_prob, target_mask)
    return 0.5 * bce + 0.5 * l2

def calculate_iou(pred_mask, gt_mask, threshold=0.5):
    """
    Calculate Intersection over Union for vortex masks.
    """
    pred = (pred_mask > threshold).float()
    intersection = (pred * gt_mask).sum()
    union = pred.sum() + gt_mask.sum() - intersection
    return (intersection + 1e-8) / (union + 1e-8)
