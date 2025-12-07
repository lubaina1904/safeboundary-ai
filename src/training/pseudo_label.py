"""
Pseudo-Label Generation for SafeBoundary AI
Expands training set by labeling unlabeled frames with trained model
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.unet_plus import BladderSegmentationModel


class PseudoLabelGenerator:
    """
    Generate pseudo-labels using trained model
    """
    def __init__(self, model_path, confidence_threshold=0.85, device='mps'):
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Load model
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create model
        config = checkpoint.get('config', {})
        self.model = BladderSegmentationModel(
            encoder_name=config.get('model', {}).get('encoder_name', 'efficientnet-b3'),
            encoder_weights=None  # Don't load pretrained weights
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        
        print(f"✓ Model loaded (Best IoU: {checkpoint.get('best_iou', 'N/A')})")
    
    def preprocess(self, image, size=512):
        """Preprocess image for inference"""
        # Resize
        image_resized = cv2.resize(image, (size, size))
        
        # Normalize
        image_norm = image_resized.astype(np.float32) / 255.0
        image_norm = (image_norm - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # To tensor
        image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def predict(self, image):
        """
        Predict mask for image
        
        Returns:
            mask: Binary mask (H, W)
            confidence: Confidence score
        """
        with torch.no_grad():
            # Preprocess
            input_tensor = self.preprocess(image)
            
            # Predict
            output, _ = self.model(input_tensor)
            
            # Get probability
            prob = torch.sigmoid(output).cpu().numpy()[0, 0]
            
            # Resize to original size
            prob = cv2.resize(prob, (image.shape[1], image.shape[0]))
            
            # Calculate confidence (inverse of entropy)
            entropy = -(prob * np.log(prob + 1e-10) + 
                       (1 - prob) * np.log(1 - prob + 1e-10))
            confidence = 1 - entropy.mean()
            
            # Threshold
            mask = (prob > 0.5).astype(np.uint8) * 255
            
            return mask, confidence
    
    def is_plausible(self, mask, image_shape):
        """
        Check if mask is anatomically plausible for bladder
        """
        h, w = image_shape[:2]
        mask_area = (mask > 127).sum()
        image_area = h * w
        
        # Size check
        ratio = mask_area / image_area
        if not (0.03 < ratio < 0.25):
            return False, f"Size implausible: {ratio:.1%}"
        
        # Location check (should be in lower part)
        mask_binary = (mask > 127).astype(np.uint8)
        y_coords, x_coords = np.where(mask_binary)
        
        if len(y_coords) == 0:
            return False, "Empty mask"
        
        centroid_y = np.mean(y_coords)
        if centroid_y < h * 0.33:
            return False, f"Too high (centroid at {centroid_y/h:.1%})"
        
        # Compactness check
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            area = cv2.contourArea(contours[0])
            perimeter = cv2.arcLength(contours[0], True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity < 0.2:
                    return False, f"Too elongated (circularity={circularity:.2f})"
        
        return True, "Plausible"
    
    def refine_mask(self, mask):
        """Post-process mask"""
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
        
        # Keep largest component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_cleaned, connectivity=8)
        
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_cleaned = (labels == largest_label).astype(np.uint8) * 255
        
        # Smooth
        mask_cleaned = cv2.GaussianBlur(mask_cleaned, (5, 5), 0)
        mask_cleaned = (mask_cleaned > 127).astype(np.uint8) * 255
        
        return mask_cleaned
    
    def generate_pseudo_labels(self, unlabeled_dir, labeled_dir, output_dir):
        """
        Generate pseudo-labels for all unlabeled frames
        
        Args:
            unlabeled_dir: Directory with all frames
            labeled_dir: Directory with manually labeled frames
            output_dir: Where to save pseudo-labels
        """
        unlabeled_dir = Path(unlabeled_dir)
        labeled_dir = Path(labeled_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all frames
        all_frames = set(unlabeled_dir.glob('*.jpg')) | set(unlabeled_dir.glob('*.png'))
        all_frames = {f for f in all_frames if not f.stem.endswith('_mask')}
        
        # Get labeled frames
        labeled_frames = set(labeled_dir.glob('*.jpg')) | set(labeled_dir.glob('*.png'))
        labeled_frames = {f.stem for f in labeled_frames if not f.stem.endswith('_mask')}
        
        # Get unlabeled frames
        unlabeled_frames = [f for f in all_frames if f.stem not in labeled_frames]
        
        print(f"\nPseudo-Label Generation")
        print("=" * 70)
        print(f"Total frames: {len(all_frames)}")
        print(f"Labeled: {len(labeled_frames)}")
        print(f"Unlabeled: {len(unlabeled_frames)}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print("=" * 70 + "\n")
        
        results = []
        accepted = 0
        rejected = 0
        
        for frame_path in tqdm(unlabeled_frames, desc="Generating pseudo-labels"):
            # Load image
            image = cv2.imread(str(frame_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Predict
            mask, confidence = self.predict(image_rgb)
            
            # Refine
            mask_refined = self.refine_mask(mask)
            
            # Check plausibility
            plausible, reason = self.is_plausible(mask_refined, image.shape)
            
            # Check confidence
            accept = confidence >= self.confidence_threshold and plausible
            
            if accept:
                # Save pseudo-label
                output_image = output_dir / frame_path.name
                output_mask = output_dir / f"{frame_path.stem}_mask.png"
                
                cv2.imwrite(str(output_image), image)
                cv2.imwrite(str(output_mask), mask_refined)
                
                accepted += 1
            else:
                rejected += 1
            
            results.append({
                'frame': frame_path.name,
                'confidence': float(confidence),
                'plausible': plausible,
                'reason': reason,
                'accepted': accept
            })
        
        # Save results
        results_path = output_dir / 'pseudo_label_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Summary
        print(f"\n{'=' * 70}")
        print("Pseudo-Labeling Complete!")
        print("=" * 70)
        print(f"Accepted: {accepted} / {len(unlabeled_frames)} ({accepted/len(unlabeled_frames)*100:.1f}%)")
        print(f"Rejected: {rejected} / {len(unlabeled_frames)} ({rejected/len(unlabeled_frames)*100:.1f}%)")
        print(f"Avg confidence (accepted): {np.mean([r['confidence'] for r in results if r['accepted']]):.3f}")
        print(f"Pseudo-labels saved to: {output_dir}")
        print(f"Results saved to: {results_path}")
        print("=" * 70 + "\n")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Generate pseudo-labels')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--unlabeled_frames', type=str, required=True,
                       help='Directory with all frames')
    parser.add_argument('--labeled_frames', type=str, required=True,
                       help='Directory with manually labeled frames')
    parser.add_argument('--output', type=str, default='data/pseudo_labels',
                       help='Output directory for pseudo-labels')
    parser.add_argument('--confidence_threshold', type=float, default=0.85,
                       help='Minimum confidence to accept pseudo-label')
    parser.add_argument('--device', type=str, default='mps',
                       choices=['mps', 'cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Check device
    device = args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, using CPU")
        device = 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = 'cpu'
    
    # Create generator
    generator = PseudoLabelGenerator(
        model_path=args.model,
        confidence_threshold=args.confidence_threshold,
        device=device
    )
    
    # Generate pseudo-labels
    results = generator.generate_pseudo_labels(
        unlabeled_dir=args.unlabeled_frames,
        labeled_dir=args.labeled_frames,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()