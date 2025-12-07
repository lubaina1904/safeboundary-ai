"""
Semi-Automatic Annotation Tool using Segment Anything Model (SAM)
For bladder segmentation in laparoscopic surgery frames
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from segment_anything import sam_model_registry, SamPredictor


class BladderAnnotator:
    """
    Interactive annotation tool using SAM
    """
    def __init__(self, sam_checkpoint, model_type='vit_h', device='mps'):
        """
        Initialize SAM model
        
        Args:
            sam_checkpoint: Path to SAM checkpoint
            model_type: 'vit_h', 'vit_l', or 'vit_b'
            device: 'mps' for Mac M1/M2, 'cuda' for NVIDIA, 'cpu' otherwise
        """
        print(f"🔧 Loading SAM model ({model_type})...")
        
        # Check device availability
        if device == 'mps' and not torch.backends.mps.is_available():
            print("⚠️  MPS not available, falling back to CPU")
            device = 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            device = 'cpu'
        
        self.device = device
        print(f"   Using device: {device}")
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        
        self.predictor = SamPredictor(sam)
        self.current_image = None
        self.current_mask = None
        
        print("✅ SAM model loaded successfully")
    
    def predict_bladder(self, image, point_coords=None, point_labels=None):
        """
        Predict bladder mask from image
        
        Args:
            image: RGB image (H, W, 3)
            point_coords: Optional click points [[x, y], ...]
            point_labels: Optional labels [1=foreground, 0=background]
        """
        # Set image
        self.predictor.set_image(image)
        
        # Use default anatomical prior if no points provided
        if point_coords is None:
            h, w = image.shape[:2]
            # Bladder typically in lower-center of frame
            point_coords = np.array([
                [w // 2, int(h * 0.65)],  # Center-low
            ])
            point_labels = np.array([1])  # Foreground
        
        # Predict
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        
        # Select best mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        return best_mask, scores[best_idx]
    
    def refine_mask(self, mask):
        """
        Post-process mask to remove noise and ensure anatomical plausibility
        """
        # Convert to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
        
        # Keep only largest connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_cleaned, connectivity=8)
        
        if num_labels > 1:
            # Find largest component (excluding background)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_cleaned = (labels == largest_label).astype(np.uint8) * 255
        
        # Smooth boundaries
        mask_cleaned = cv2.GaussianBlur(mask_cleaned, (5, 5), 0)
        mask_cleaned = (mask_cleaned > 127).astype(np.uint8) * 255
        
        return mask_cleaned
    
    def is_plausible_bladder(self, mask, image_shape):
        """
        Check if mask is anatomically plausible for bladder
        """
        h, w = image_shape[:2]
        mask_area = mask.sum() / 255
        image_area = h * w
        
        # 1. Size check: bladder should be 3-20% of frame
        ratio = mask_area / image_area
        if not (0.03 < ratio < 0.25):
            return False, f"Size implausible: {ratio:.1%} of frame"
        
        # 2. Location check: bladder should be in lower 2/3
        mask_binary = (mask > 127).astype(np.uint8)
        y_coords, x_coords = np.where(mask_binary)
        
        if len(y_coords) == 0:
            return False, "Empty mask"
        
        centroid_y = np.mean(y_coords)
        
        if centroid_y < h * 0.33:
            return False, f"Too high in frame (centroid at {centroid_y/h:.1%})"
        
        # 3. Compactness check: should be relatively round
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            area = cv2.contourArea(contours[0])
            perimeter = cv2.arcLength(contours[0], True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity < 0.2:  # Too elongated
                    return False, f"Too elongated (circularity={circularity:.2f})"
        
        return True, "Plausible"


def interactive_annotation(annotator, frames_dir, output_dir, auto_mode=True):
    """
    Interactive annotation with SAM assistance
    
    Args:
        annotator: BladderAnnotator instance
        frames_dir: Directory containing frames
        output_dir: Directory to save annotations
        auto_mode: If True, use automatic prediction; if False, require manual points
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all frame files
    frame_files = sorted(frames_dir.glob('*.jpg')) + sorted(frames_dir.glob('*.png'))
    
    if not frame_files:
        raise ValueError(f"No frames found in {frames_dir}")
    
    print(f"\n📋 Found {len(frame_files)} frames to annotate")
    print(f"💾 Annotations will be saved to: {output_dir}")
    
    if auto_mode:
        print("\n🤖 AUTO MODE: Using automatic bladder detection")
        print("   Review each annotation and press:")
        print("     'y' = Accept")
        print("     'n' = Reject (skip)")
        print("     'q' = Quit")
    else:
        print("\n✋ MANUAL MODE: Click on bladder to segment")
        print("   Click to add point, then press:")
        print("     'y' = Accept")
        print("     'r' = Reset points")
        print("     'n' = Skip frame")
        print("     'q' = Quit")
    
    annotation_log = []
    click_points = []
    
    def mouse_callback(event, x, y, flags, param):
        """Handle mouse clicks for manual mode"""
        nonlocal click_points
        if event == cv2.EVENT_LBUTTONDOWN:
            click_points.append([x, y])
            print(f"   Point added: ({x}, {y})")
    
    for idx, frame_path in enumerate(tqdm(frame_files, desc="Annotating")):
        # Load image
        image = cv2.imread(str(frame_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if auto_mode:
            # Automatic prediction
            mask, confidence = annotator.predict_bladder(image_rgb)
            
            # Refine mask
            mask_refined = annotator.refine_mask(mask)
            
            # Check plausibility
            plausible, reason = annotator.is_plausible_bladder(mask_refined, image.shape)
            
            # Visualize
            vis = visualize_annotation(image, mask_refined, confidence, plausible, reason)
            
            cv2.imshow('Annotation', vis)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('y'):  # Accept
                # Save annotation
                mask_filename = frame_path.stem + '_mask.png'
                mask_path = output_dir / mask_filename
                cv2.imwrite(str(mask_path), mask_refined)
                
                annotation_log.append({
                    'frame': frame_path.name,
                    'mask': mask_filename,
                    'confidence': float(confidence),
                    'plausible': plausible,
                    'reason': reason,
                    'accepted': True
                })
            
            elif key == ord('n'):  # Reject
                annotation_log.append({
                    'frame': frame_path.name,
                    'accepted': False,
                    'reason': 'Manually rejected'
                })
                continue
            
            elif key == ord('q'):  # Quit
                break
        
        else:  # Manual mode
            click_points = []
            cv2.namedWindow('Annotation')
            cv2.setMouseCallback('Annotation', mouse_callback)
            
            while True:
                # Show image with click points
                vis = image.copy()
                for pt in click_points:
                    cv2.circle(vis, tuple(pt), 5, (0, 255, 0), -1)
                
                cv2.imshow('Annotation', vis)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('y') and click_points:  # Accept with points
                    # Predict with manual points
                    point_coords = np.array(click_points)
                    point_labels = np.ones(len(click_points))
                    
                    mask, confidence = annotator.predict_bladder(
                        image_rgb, point_coords, point_labels
                    )
                    
                    mask_refined = annotator.refine_mask(mask)
                    
                    # Save
                    mask_filename = frame_path.stem + '_mask.png'
                    mask_path = output_dir / mask_filename
                    cv2.imwrite(str(mask_path), mask_refined)
                    
                    annotation_log.append({
                        'frame': frame_path.name,
                        'mask': mask_filename,
                        'confidence': float(confidence),
                        'points': click_points,
                        'accepted': True
                    })
                    break
                
                elif key == ord('r'):  # Reset points
                    click_points = []
                
                elif key == ord('n'):  # Skip
                    annotation_log.append({
                        'frame': frame_path.name,
                        'accepted': False,
                        'reason': 'Skipped'
                    })
                    break
                
                elif key == ord('q'):  # Quit
                    cv2.destroyAllWindows()
                    save_annotation_log(annotation_log, output_dir)
                    return
    
    cv2.destroyAllWindows()
    save_annotation_log(annotation_log, output_dir)
    
    # Summary
    accepted = sum(1 for a in annotation_log if a.get('accepted', False))
    print(f"\n✅ Annotation complete!")
    print(f"   Accepted: {accepted}/{len(frame_files)}")
    print(f"   Annotations saved to: {output_dir}")


def visualize_annotation(image, mask, confidence, plausible, reason):
    """
    Create visualization of annotation
    """
    vis = image.copy()
    
    # Overlay mask
    mask_colored = np.zeros_like(image)
    mask_colored[:, :, 1] = mask  # Green channel
    vis = cv2.addWeighted(vis, 0.7, mask_colored, 0.3, 0)
    
    # Draw boundary
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(vis, contours, -1, (255, 0, 255), 2)
    
    # Add info text
    color = (0, 255, 0) if plausible else (0, 0, 255)
    status = "✓ PLAUSIBLE" if plausible else "✗ IMPLAUSIBLE"
    
    cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(vis, f"Confidence: {confidence:.3f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, f"{reason}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Instructions
    cv2.putText(vis, "Y=Accept | N=Reject | Q=Quit", (10, vis.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    return vis


def save_annotation_log(annotation_log, output_dir):
    """Save annotation log to JSON"""
    log_path = output_dir / 'annotation_log.json'
    with open(log_path, 'w') as f:
        json.dump(annotation_log, f, indent=2)
    print(f"📝 Annotation log saved to: {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Semi-automatic bladder annotation using SAM'
    )
    parser.add_argument(
        '--frames',
        type=str,
        required=True,
        help='Directory containing extracted frames'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/annotations',
        help='Output directory for annotations'
    )
    parser.add_argument(
        '--sam_checkpoint',
        type=str,
        default='models/sam/sam_vit_h_4b8939.pth',
        help='Path to SAM checkpoint'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='vit_h',
        choices=['vit_h', 'vit_l', 'vit_b'],
        help='SAM model type'
    )
    parser.add_argument(
        '--manual',
        action='store_true',
        help='Use manual mode (click on bladder)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        choices=['mps', 'cuda', 'cpu'],
        help='Device to use (mps for Mac M1/M2)'
    )
    
    args = parser.parse_args()
    
    # Initialize annotator
    annotator = BladderAnnotator(
        sam_checkpoint=args.sam_checkpoint,
        model_type=args.model_type,
        device=args.device
    )
    
    # Run annotation
    interactive_annotation(
        annotator=annotator,
        frames_dir=args.frames,
        output_dir=args.output,
        auto_mode=not args.manual
    )


if __name__ == '__main__':
    main()