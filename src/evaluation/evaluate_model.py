"""
Quick Model Evaluation - Visualize Predictions
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from src.models.unet_plus import BladderSegmentationModel


def load_model(model_path, device='mps'):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model = BladderSegmentationModel(
        encoder_name='efficientnet-b3',
        encoder_weights=None,
        classes=1
    )
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model


def preprocess(image, size=256):
    """Preprocess image for model"""
    img_resized = cv2.resize(image, (size, size))
    img_norm = img_resized.astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_norm - mean) / std
    
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float()
    return img_tensor


def predict(model, image, device='mps'):
    """Run prediction"""
    input_tensor = preprocess(image).to(device)
    
    with torch.no_grad():
        output, _ = model(input_tensor)
        mask = torch.sigmoid(output).cpu().numpy()[0, 0]
    
    return mask


def visualize_prediction(image, mask, ground_truth=None):
    """Visualize prediction vs ground truth"""
    fig, axes = plt.subplots(1, 3 if ground_truth is not None else 2, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Predicted mask
    axes[1].imshow(mask, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Predicted Mask (Heatmap)')
    axes[1].axis('off')
    
    # Ground truth if available
    if ground_truth is not None:
        axes[2].imshow(ground_truth, cmap='gray')
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def evaluate_samples(model, data_dir, num_samples=5, device='mps'):
    """Evaluate on sample images"""
    image_dir = Path(data_dir) / 'images'
    mask_dir = Path(data_dir) / 'masks'
    
    image_files = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')))[:num_samples]
    
    print(f"Evaluating {len(image_files)} samples...")
    
    for i, img_path in enumerate(image_files):
        print(f"\nSample {i+1}/{len(image_files)}: {img_path.name}")
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ⚠️  Could not load image")
            continue
        
        # Load ground truth if exists
        mask_path = mask_dir / img_path.name
        ground_truth = None
        if mask_path.exists():
            ground_truth = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Predict
        pred_mask = predict(model, image, device)
        pred_mask_resized = cv2.resize(pred_mask, (image.shape[1], image.shape[0]))
        
        # Calculate statistics
        pred_binary = (pred_mask_resized > 0.5).astype(np.uint8)
        pred_pixels = pred_binary.sum()
        pred_percent = (pred_pixels / pred_binary.size) * 100
        
        print(f"  Prediction stats:")
        print(f"    - Mean confidence: {pred_mask_resized.mean():.3f}")
        print(f"    - Max confidence: {pred_mask_resized.max():.3f}")
        print(f"    - Predicted pixels: {pred_pixels} ({pred_percent:.1f}%)")
        
        if ground_truth is not None:
            gt_pixels = (ground_truth > 127).sum()
            gt_percent = (gt_pixels / ground_truth.size) * 100
            print(f"  Ground truth stats:")
            print(f"    - GT pixels: {gt_pixels} ({gt_percent:.1f}%)")
            
            # Calculate Dice score
            intersection = (pred_binary & (ground_truth > 127)).sum()
            dice = (2 * intersection) / (pred_binary.sum() + (ground_truth > 127).sum() + 1e-7)
            print(f"    - Dice score: {dice:.3f}")
        
        # Visualize
        fig = visualize_prediction(image, pred_mask_resized, ground_truth)
        output_path = f"outputs/evaluation/sample_{i+1}.png"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to evaluate')
    parser.add_argument('--device', type=str, default='mps', choices=['mps', 'cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Check device
    device = args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, using CPU")
        device = 'cpu'
    
    print(f"Loading model from {args.model}")
    model = load_model(args.model, device)
    print(f"✓ Model loaded on {device}\n")
    
    evaluate_samples(model, args.data_dir, args.num_samples, device)
    
    print("\n" + "="*70)
    print("Evaluation complete! Check outputs/evaluation/ for visualizations")
    print("="*70)


if __name__ == '__main__':
    main()