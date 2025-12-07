"""
Custom Loss Functions for Bladder Segmentation
Emphasizes boundary accuracy for clinical safety
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class BoundaryLoss(nn.Module):
    """
    Loss that heavily penalizes errors at the bladder boundary
    This is CRITICAL for clinical safety - boundary errors = danger!
    """
    def __init__(self, theta=5):
        super().__init__()
        self.theta = theta  # Boundary thickness in pixels
    
    def get_boundary(self, mask):
        """
        Extract boundary region from mask
        """
        # Erosion and dilation to get boundary
        kernel_size = self.theta * 2 + 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size).to(mask.device)
        
        # Dilate
        dilated = F.conv2d(mask, kernel, padding=self.theta)
        dilated = (dilated > 0).float()
        
        # Erode
        eroded = F.conv2d(mask, kernel, padding=self.theta)
        eroded = (eroded == kernel.sum()).float()
        
        # Boundary is difference
        boundary = dilated - eroded
        boundary = boundary.clamp(0, 1)
        
        return boundary
    
    def forward(self, pred, target):
        """
        Calculate boundary loss
        
        Args:
            pred: Predicted mask (logits) [B, 1, H, W]
            target: Ground truth mask [B, 1, H, W]
        """
        # Get boundary region
        boundary = self.get_boundary(target)
        
        # Apply sigmoid to predictions
        pred_prob = torch.sigmoid(pred)
        
        # Calculate BCE on boundary region
        bce = F.binary_cross_entropy(pred_prob, target, reduction='none')
        
        # Weight by boundary
        # Errors at boundary are 3x more important
        weights = 1.0 + 2.0 * boundary
        weighted_bce = bce * weights
        
        return weighted_bce.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function for bladder segmentation
    Balances multiple objectives:
    - Overall segmentation quality (Dice)
    - Class imbalance (Focal)
    - Boundary accuracy (Boundary)
    """
    def __init__(self, 
                 dice_weight=0.3,
                 focal_weight=0.3,
                 boundary_weight=0.4):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        
        # Individual loss functions
        self.dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.focal_loss = smp.losses.FocalLoss(mode='binary')
        self.boundary_loss = BoundaryLoss(theta=5)
    
    def forward(self, pred, target):
        """
        Calculate combined loss
        
        Args:
            pred: Predicted mask (logits) [B, 1, H, W]
            target: Ground truth mask [B, 1, H, W]
        """
        # Calculate individual losses
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(torch.sigmoid(pred), target)
        boundary = self.boundary_loss(pred, target)
        
        # Weighted combination
        total_loss = (
            self.dice_weight * dice +
            self.focal_weight * focal +
            self.boundary_weight * boundary
        )
        
        # Return total and individual losses for logging
        return total_loss, {
            'dice': dice.item(),
            'focal': focal.item(),
            'boundary': boundary.item(),
            'total': total_loss.item()
        }


class TverskyLoss(nn.Module):
    """
    Tversky loss - good for handling class imbalance
    Can tune alpha/beta to penalize FP or FN more
    """
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # True Positives, False Positives, False Negatives
        TP = (pred * target).sum(dim=(2, 3))
        FP = ((1 - target) * pred).sum(dim=(2, 3))
        FN = (target * (1 - pred)).sum(dim=(2, 3))
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky.mean()


class EdgeAwareLoss(nn.Module):
    """
    Loss that emphasizes edges using Sobel operator
    """
    def __init__(self):
        super().__init__()
        
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.sobel_x = sobel_x.view(1, 1, 3, 3)
        self.sobel_y = sobel_y.view(1, 1, 3, 3)
    
    def get_edges(self, x):
        """Extract edges using Sobel operator"""
        self.sobel_x = self.sobel_x.to(x.device)
        self.sobel_y = self.sobel_y.to(x.device)
        
        edge_x = F.conv2d(x, self.sobel_x, padding=1)
        edge_y = F.conv2d(x, self.sobel_y, padding=1)
        
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        return edges
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Get edges
        pred_edges = self.get_edges(pred)
        target_edges = self.get_edges(target)
        
        # L1 loss on edges
        edge_loss = F.l1_loss(pred_edges, target_edges)
        
        return edge_loss


def get_loss_function(config):
    """
    Factory function to get loss from config
    """
    loss_type = config.get('loss_type', 'combined')
    
    if loss_type == 'combined':
        return CombinedLoss(
            dice_weight=config.get('dice_weight', 0.3),
            focal_weight=config.get('focal_weight', 0.3),
            boundary_weight=config.get('boundary_weight', 0.4)
        )
    elif loss_type == 'boundary':
        return BoundaryLoss(theta=config.get('boundary_theta', 5))
    elif loss_type == 'tversky':
        return TverskyLoss(
            alpha=config.get('tversky_alpha', 0.7),
            beta=config.get('tversky_beta', 0.3)
        )
    elif loss_type == 'edge':
        return EdgeAwareLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == '__main__':
    # Test loss functions
    pred = torch.randn(2, 1, 256, 256)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    
    print("Testing loss functions...")
    
    # Combined loss
    combined_loss = CombinedLoss()
    loss, losses_dict = combined_loss(pred, target)
    print(f"\nCombined Loss: {loss.item():.4f}")
    print(f"  Dice: {losses_dict['dice']:.4f}")
    print(f"  Focal: {losses_dict['focal']:.4f}")
    print(f"  Boundary: {losses_dict['boundary']:.4f}")
    
    # Boundary loss
    boundary_loss = BoundaryLoss()
    loss = boundary_loss(pred, target)
    print(f"\nBoundary Loss: {loss.item():.4f}")
    
    # Tversky loss
    tversky_loss = TverskyLoss()
    loss = tversky_loss(pred, target)
    print(f"Tversky Loss: {loss.item():.4f}")
    
    # Edge loss
    edge_loss = EdgeAwareLoss()
    loss = edge_loss(pred, target)
    print(f"Edge Loss: {loss.item():.4f}")