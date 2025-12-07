"""
U-Net++ Model with Boundary Attention for Bladder Segmentation
Optimized for laparoscopic surgery imagery
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class BoundaryAttentionModule(nn.Module):
    """
    Attention module that focuses on bladder boundaries
    Critical for accurate danger zone calculation
    """
    def __init__(self, in_channels):
        super().__init__()
        
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
        )
        
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract edge features
        edges = self.edge_conv(x)
        
        # Generate attention map
        attention = self.attention_conv(edges)
        
        # Apply attention
        out = x * attention
        
        return out, attention


class BladderSegmentationModel(nn.Module):
    """
    Complete model for bladder segmentation
    Based on U-Net++ with boundary attention
    """
    def __init__(self, 
                 encoder_name='efficientnet-b3',
                 encoder_weights='imagenet',
                 classes=1,
                 activation=None):
        super().__init__()
        
        # Base U-Net++ model
        self.backbone = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
            activation=activation,
        )
        
        # Boundary attention module
        # U-Net++ decoder output has 16 channels
        self.boundary_attention = BoundaryAttentionModule(16)
        
        # Final segmentation head
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, classes, 1)
        )
    
    def forward(self, x):
        # Encoder
        features = self.backbone.encoder(x)
        
        # Decoder
        decoder_output = self.backbone.decoder(*features)
        
        # Apply boundary attention
        attended_features, attention_map = self.boundary_attention(decoder_output)
        
        # Final prediction
        mask = self.final_conv(attended_features)
        
        return mask, attention_map
    
    def predict(self, x):
        """Simple prediction without attention map"""
        mask, _ = self.forward(x)
        return torch.sigmoid(mask)


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for better robustness
    """
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x):
        predictions = []
        for model in self.models:
            pred = model.predict(x)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        return ensemble_pred


def create_model(config):
    """
    Factory function to create model from config
    
    Args:
        config: Dictionary with model configuration
            - encoder_name: e.g., 'efficientnet-b3', 'resnet50'
            - encoder_weights: 'imagenet' or None
            - classes: number of output classes (1 for binary)
    """
    encoder_name = config.get('encoder_name', 'efficientnet-b3')
    encoder_weights = config.get('encoder_weights', 'imagenet')
    classes = config.get('classes', 1)
    
    model = BladderSegmentationModel(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=classes,
        activation=None  # We'll apply sigmoid separately
    )
    
    return model


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model creation
    config = {
        'encoder_name': 'efficientnet-b3',
        'encoder_weights': 'imagenet',
        'classes': 1
    }
    
    model = create_model(config)
    print(f"Model created: {model.__class__.__name__}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        output, attention = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Attention shape: {attention.shape}")