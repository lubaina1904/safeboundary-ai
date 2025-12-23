"""
Custom Loss Functions for Bladder Segmentation
MPS-compatible for Apple Silicon (M1/M2/M4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# MPS-COMPATIBLE FOCAL LOSS
# =====================================================================
class MPSFocalLoss(nn.Module):
    """
    Custom Focal Loss implementation that works on MPS devices
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        """
        pred: logits (before sigmoid), shape (B, C, H, W)
        target: binary mask, shape (B, C, H, W)
        """
        pred = torch.sigmoid(pred)
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
        
        # Binary cross entropy
        ce_loss = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        
        # Focal weight
        p_t = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha balancing
        if self.alpha >= 0:
            alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
            
        return focal_loss.mean()


# =====================================================================
# MPS-COMPATIBLE DICE LOSS
# =====================================================================
class MPSDiceLoss(nn.Module):
    """
    Custom Dice Loss that works on MPS devices
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        pred: logits (before sigmoid), shape (B, C, H, W)
        target: binary mask, shape (B, C, H, W)
        """
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


# =====================================================================
# BOUNDARY LOSS
# =====================================================================
class BoundaryLoss(nn.Module):
    """
    Loss that heavily penalizes errors at the bladder boundary.
    Critical for clinical safety.
    """
    def __init__(self, theta=5):
        super().__init__()
        self.theta = theta

    def get_boundary(self, mask):
        mask = mask.float()

        kernel_size = self.theta * 2 + 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)

        # Dilate
        dilated = F.conv2d(mask, kernel, padding=self.theta)
        dilated = (dilated > 0).float()

        # Erode
        eroded = F.conv2d(mask, kernel, padding=self.theta)
        eroded = (eroded == kernel.sum()).float()

        boundary = (dilated - eroded).clamp(0, 1)
        return boundary

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()

        boundary = self.get_boundary(target)
        pred_prob = torch.sigmoid(pred)

        bce = F.binary_cross_entropy(pred_prob, target, reduction="none")

        # boundary pixels receive 3× weight
        weights = 1.0 + 2.0 * boundary
        weighted_bce = bce * weights

        return weighted_bce.mean()


# =====================================================================
# COMBINED LOSS
# =====================================================================
class CombinedLoss(nn.Module):
    """
    Combines Dice + Focal + Boundary losses, MPS-safe
    """
    def __init__(self, dice_weight=0.3, focal_weight=0.3, boundary_weight=0.4):
        super().__init__()

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight

        # Use custom MPS-compatible losses
        self.dice_loss = MPSDiceLoss(smooth=1.0)
        self.focal_loss = MPSFocalLoss(alpha=0.25, gamma=2.0)
        self.boundary_loss = BoundaryLoss(theta=5)

    def forward(self, pred, target):
        # Convert to float32 to ensure compatibility
        pred = pred.float()
        target = target.float()

        # All losses computed on the same device
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        boundary = self.boundary_loss(pred, target)

        # Combine losses
        total_loss = (
            self.dice_weight * dice +
            self.focal_weight * focal +
            self.boundary_weight * boundary
        )

        return total_loss, {
            "dice": float(dice.item()),
            "focal": float(focal.item()),
            "boundary": float(boundary.item()),
            "total": float(total_loss.item())
        }


# =====================================================================
# TVERSKY LOSS
# =====================================================================
class TverskyLoss(nn.Module):
    """
    Tversky loss for imbalanced segmentation
    """
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred.float())
        target = target.float()

        TP = (pred * target).sum(dim=(2, 3))
        FP = ((1 - target) * pred).sum(dim=(2, 3))
        FN = (target * (1 - pred)).sum(dim=(2, 3))

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky.mean()


# =====================================================================
# EDGE-AWARE LOSS
# =====================================================================
class EdgeAwareLoss(nn.Module):
    """
    L1 loss on edges using Sobel operator
    """
    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.sobel_x = sobel_x.view(1, 1, 3, 3)
        self.sobel_y = sobel_y.view(1, 1, 3, 3)

    def get_edges(self, x):
        self.sobel_x = self.sobel_x.to(x.device)
        self.sobel_y = self.sobel_y.to(x.device)

        edge_x = F.conv2d(x, self.sobel_x, padding=1)
        edge_y = F.conv2d(x, self.sobel_y, padding=1)

        return torch.sqrt(edge_x ** 2 + edge_y ** 2)

    def forward(self, pred, target):
        pred = torch.sigmoid(pred.float())
        target = target.float()

        pred_edges = self.get_edges(pred)
        target_edges = self.get_edges(target)

        return F.l1_loss(pred_edges, target_edges)


# =====================================================================
# FACTORY FUNCTION
# =====================================================================
def get_loss_function(config):
    loss_type = config.get("loss_type", "combined")

    if loss_type == "combined":
        return CombinedLoss(
            dice_weight=config.get("dice_weight", 0.3),
            focal_weight=config.get("focal_weight", 0.3),
            boundary_weight=config.get("boundary_weight", 0.4)
        )
    elif loss_type == "boundary":
        return BoundaryLoss(theta=config.get("boundary_theta", 5))
    elif loss_type == "tversky":
        return TverskyLoss(
            alpha=config.get("tversky_alpha", 0.7),
            beta=config.get("tversky_beta", 0.3)
        )
    elif loss_type == "edge":
        return EdgeAwareLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
