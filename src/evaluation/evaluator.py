"""
Model Evaluation Script
Calculates IoU, Dice, BPE, FPS and other metrics
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
from scipy.spatial.distance import directed_hausdorff
import time

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.unet_plus import BladderSegmentationModel


class Evaluator:
    """Evaluate model performance"""
    
    def __init__(self, model_path, device='mps'):
        self.device = device
        
        # Load model
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        config = checkpoint.get('config', {})
        self.model = BladderSegmentationModel(
            encoder_name=config.get('model', {}).get('encoder_name', 'efficientnet-b3'),
            encoder_weights=None
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        
        print(f"✓ Model loaded")
        
        # Metrics storage
        self.metrics = {
            'iou': [],
            'dice': [],
            'precision': [],
            'recall': [],
            'bpe': [],
            'fps': []
        }
    
    def preprocess(self, image, size=512):
        """Preprocess image"""
        image_resized = cv2.resize(image, (size, size))
        image_norm = image_resized.astype(np.float32) / 255.0
        image_norm = (image_norm - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def predict(self, image):
        """Predict mask"""
        with torch.no_grad():
            input_tensor = self.preprocess(image)
            output, _ = self.model(input_tensor)
            mask = torch.sigmoid(output).cpu().numpy()[0, 0]
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            return mask
    
    def calculate_iou(self, pred, gt):
        """Calculate IoU"""
        pred_binary = (pred > 0.5).astype(np.uint8)
        gt_binary = (gt > 127).astype(np.uint8)
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        iou = intersection / (union + 1e-7)
        return iou
    
    def calculate_dice(self, pred, gt):
        """Calculate Dice coefficient"""
        pred_binary = (pred > 0.5).astype(np.uint8)
        gt_binary = (gt > 127).astype(np.uint8)
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        
        dice = 2 * intersection / (pred_binary.sum() + gt_binary.sum() + 1e-7)
        return dice
    
    def calculate_precision_recall(self, pred, gt):
        """Calculate precision and recall"""
        pred_binary = (pred > 0.5).astype(np.uint8)
        gt_binary = (gt > 127).astype(np.uint8)
        
        tp = np.logical_and(pred_binary, gt_binary).sum()
        fp = np.logical_and(pred_binary, 1 - gt_binary).sum()
        fn = np.logical_and(1 - pred_binary, gt_binary).sum()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        
        return precision, recall
    
    def extract_boundary(self, mask):
        """Extract boundary from mask"""
        mask_binary = (mask > 0.5 if mask.max() <= 1 else mask > 127).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask_binary, kernel, iterations=1)
        boundary = mask_binary - eroded
        return boundary
    
    def calculate_bpe(self, pred, gt):
        """
        Calculate Boundary Proximity Error
        THE KEY CLINICAL METRIC
        """
        # Extract boundaries
        pred_boundary = self.extract_boundary(pred)
        gt_boundary = self.extract_boundary(gt)
        
        if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
            return float('inf')
        
        # Get coordinates
        pred_coords = np.column_stack(np.where(pred_boundary))
        gt_coords = np.column_stack(np.where(gt_boundary))
        
        if len(pred_coords) == 0 or len(gt_coords) == 0:
            return float('inf')
        
        # Calculate Hausdorff distance
        try:
            dist1, _, _ = directed_hausdorff(pred_coords, gt_coords)
            dist2, _, _ = directed_hausdorff(gt_coords, pred_coords)
            bpe = (dist1 + dist2) / 2
        except:
            bpe = float('inf')
        
        return bpe
    
    def evaluate_dataset(self, test_dir):
        """Evaluate on test dataset"""
        test_dir = Path(test_dir)
        
        # Get test images
        image_files = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
        image_files = [f for f in image_files if not f.stem.endswith('_mask')]
        
        # Filter to those with masks
        test_pairs = []
        for img_file in image_files:
            mask_file = test_dir / f"{img_file.stem}_mask.png"
            if mask_file.exists():
                test_pairs.append((img_file, mask_file))
        
        print(f"\nEvaluating on {len(test_pairs)} test samples\n")
        
        for img_path, mask_path in tqdm(test_pairs, desc="Evaluating"):
            # Load image and ground truth
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            # Predict
            start_time = time.time()
            pred_mask = self.predict(image)
            inference_time = time.time() - start_time
            
            # Calculate metrics
            iou = self.calculate_iou(pred_mask, gt_mask)
            dice = self.calculate_dice(pred_mask, gt_mask)
            precision, recall = self.calculate_precision_recall(pred_mask, gt_mask)
            bpe = self.calculate_bpe(pred_mask, gt_mask)
            fps = 1.0 / inference_time if inference_time > 0 else 0
            
            # Store
            self.metrics['iou'].append(iou)
            self.metrics['dice'].append(dice)
            self.metrics['precision'].append(precision)
            self.metrics['recall'].append(recall)
            self.metrics['bpe'].append(bpe if bpe != float('inf') else 0)
            self.metrics['fps'].append(fps)
        
        return self.get_summary()
    
    def get_summary(self):
        """Get summary statistics"""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                valid_values = [v for v in values if v != float('inf')]
                if valid_values:
                    summary[f'{metric_name}_mean'] = float(np.mean(valid_values))
                    summary[f'{metric_name}_std'] = float(np.std(valid_values))
                    summary[f'{metric_name}_median'] = float(np.median(valid_values))
                    summary[f'{metric_name}_min'] = float(np.min(valid_values))
                    summary[f'{metric_name}_max'] = float(np.max(valid_values))
        
        return summary
    
    def generate_report(self, summary, output_dir):
        """Generate evaluation report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Text report
        report = "=" * 70 + "\n"
        report += "        SAFEBOUNDARY AI - EVALUATION REPORT\n"
        report += "=" * 70 + "\n\n"
        
        report += "SEGMENTATION QUALITY\n"
        report += "-" * 70 + "\n"
        report += f"Mean IoU:              {summary.get('iou_mean', 0):.4f} ± {summary.get('iou_std', 0):.4f}\n"
        report += f"  Median:              {summary.get('iou_median', 0):.4f}\n"
        report += f"  Range:               [{summary.get('iou_min', 0):.4f}, {summary.get('iou_max', 0):.4f}]\n\n"
        
        report += f"Mean Dice:             {summary.get('dice_mean', 0):.4f} ± {summary.get('dice_std', 0):.4f}\n"
        report += f"  Median:              {summary.get('dice_median', 0):.4f}\n"
        report += f"  Range:               [{summary.get('dice_min', 0):.4f}, {summary.get('dice_max', 0):.4f}]\n\n"
        
        report += f"Mean Precision:        {summary.get('precision_mean', 0):.4f} ± {summary.get('precision_std', 0):.4f}\n"
        report += f"Mean Recall:           {summary.get('recall_mean', 0):.4f} ± {summary.get('recall_std', 0):.4f}\n\n"
        
        report += "CLINICAL SAFETY (KEY METRIC)\n"
        report += "-" * 70 + "\n"
        report += f"Mean BPE:              {summary.get('bpe_mean', 0):.2f} pixels\n"
        report += f"  Median BPE:          {summary.get('bpe_median', 0):.2f} pixels\n"
        report += f"  Best BPE:            {summary.get('bpe_min', 0):.2f} pixels\n"
        report += f"  Worst BPE:           {summary.get('bpe_max', 0):.2f} pixels\n"
        report += f"  (<3px is excellent for clinical use)\n\n"
        
        report += "REAL-TIME PERFORMANCE\n"
        report += "-" * 70 + "\n"
        report += f"Mean FPS:              {summary.get('fps_mean', 0):.1f}\n"
        report += f"  Median FPS:          {summary.get('fps_median', 0):.1f}\n"
        report += f"  Min FPS:             {summary.get('fps_min', 0):.1f}\n"
        report += f"  Max FPS:             {summary.get('fps_max', 0):.1f}\n"
        report += f"  (>25 FPS required for real-time)\n\n"
        
        report += "OVERALL ASSESSMENT\n"
        report += "-" * 70 + "\n"
        
        # Assessment
        iou_mean = summary.get('iou_mean', 0)
        bpe_mean = summary.get('bpe_mean', 0)
        fps_mean = summary.get('fps_mean', 0)
        
        if iou_mean > 0.90 and bpe_mean < 3 and fps_mean > 30:
            assessment = "EXCELLENT - Production Ready! 🏆"
        elif iou_mean > 0.85 and bpe_mean < 5 and fps_mean > 25:
            assessment = "GOOD - Meets clinical requirements ✓"
        elif iou_mean > 0.80 and bpe_mean < 7 and fps_mean > 20:
            assessment = "ACCEPTABLE - Needs improvement"
        else:
            assessment = "NEEDS WORK - Below clinical standards"
        
        report += f"Status: {assessment}\n\n"
        report += "=" * 70 + "\n"
        
        # Save reports
        with open(output_dir / 'evaluation_report.txt', 'w') as f:
            f.write(report)
        
        with open(output_dir / 'evaluation_report.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(report)
        print(f"Reports saved to: {output_dir}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate SafeBoundary AI model')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--test_data', type=str, required=True, help='Test data directory')
    parser.add_argument('--output', type=str, default='outputs/reports', help='Output directory')
    parser.add_argument('--device', type=str, default='mps', choices=['mps', 'cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Check device
    device = args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, using CPU")
        device = 'cpu'
    
    # Evaluate
    evaluator = Evaluator(args.model, device=device)
    summary = evaluator.evaluate_dataset(args.test_data)
    evaluator.generate_report(summary, args.output)


if __name__ == '__main__':
    main()