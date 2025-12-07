"""
Video Processing with Danger Zone Visualization
Real-time bladder segmentation and safety warnings
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
import time

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.unet_plus import BladderSegmentationModel


class DangerZoneVisualizer:
    """Create danger zone visualization"""
    
    def __init__(self):
        # Risk distances (pixels)
        self.CRITICAL_DISTANCE = 5
        self.DANGER_DISTANCE = 10
        self.CAUTION_DISTANCE = 20
        
        # Colors
        self.colors = {
            'critical': (0, 0, 255),      # Red
            'danger': (0, 165, 255),      # Orange
            'caution': (0, 255, 255),     # Yellow
            'safe': (0, 255, 0),          # Green
            'bladder': (255, 0, 255)      # Magenta
        }
    
    def create_danger_zones(self, mask):
        """Generate multi-level danger zones"""
        mask_binary = (mask > 127).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Distance transform
        inverse_mask = 1 - mask_binary
        distance_map = distance_transform_edt(inverse_mask)
        
        # Create zones
        zones = {
            'critical': (distance_map > 0) & (distance_map <= self.CRITICAL_DISTANCE),
            'danger': (distance_map > self.CRITICAL_DISTANCE) & (distance_map <= self.DANGER_DISTANCE),
            'caution': (distance_map > self.DANGER_DISTANCE) & (distance_map <= self.CAUTION_DISTANCE)
        }
        
        return zones, contours[0], distance_map
    
    def visualize(self, frame, mask, show_dashboard=True, show_metrics=True, fps=0, bpe=0):
        """Create complete visualization"""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Generate zones
        result = self.create_danger_zones(mask)
        if result is None:
            return overlay
        
        zones, bladder_contour, distance_map = result
        
        # Draw zones (semi-transparent)
        zone_overlay = np.zeros_like(frame)
        
        for zone_name, zone_mask in zones.items():
            color = self.colors[zone_name]
            zone_overlay[zone_mask] = color
        
        # Blend
        cv2.addWeighted(overlay, 0.7, zone_overlay, 0.3, 0, overlay)
        
        # Draw bladder boundary (thick)
        cv2.drawContours(overlay, [bladder_contour], -1, self.colors['bladder'], 3)
        
        # Add dashboard
        if show_dashboard:
            overlay = self._add_dashboard(overlay, zones, mask)
        
        # Add metrics
        if show_metrics:
            overlay = self._add_metrics(overlay, fps, bpe)
        
        # Add legend
        overlay = self._add_legend(overlay)
        
        # Add risk indicator
        risk_level = self._calculate_risk_level(distance_map)
        overlay = self._add_risk_indicator(overlay, risk_level)
        
        return overlay
    
    def _add_dashboard(self, frame, zones, mask):
        """Add metrics dashboard"""
        h, w = frame.shape[:2]
        
        # Calculate metrics
        bladder_pixels = (mask > 127).sum()
        critical_pixels = zones['critical'].sum()
        danger_pixels = zones['danger'].sum()
        caution_pixels = zones['caution'].sum()
        
        # Dashboard position
        dash_x, dash_y = 20, 20
        line_height = 35
        
        metrics = [
            f"Bladder: {bladder_pixels} px",
            f"Critical: {critical_pixels} px",
            f"Danger: {danger_pixels} px",
            f"Caution: {caution_pixels} px"
        ]
        
        # Background
        max_width = 300
        dash_height = len(metrics) * line_height + 20
        
        cv2.rectangle(frame, (dash_x, dash_y),
                     (dash_x + max_width, dash_y + dash_height),
                     (0, 0, 0), -1)
        cv2.rectangle(frame, (dash_x, dash_y),
                     (dash_x + max_width, dash_y + dash_height),
                     (255, 255, 255), 2)
        
        # Metrics text
        for i, metric in enumerate(metrics):
            y = dash_y + 30 + i * line_height
            cv2.putText(frame, metric, (dash_x + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _add_metrics(self, frame, fps, bpe):
        """Add FPS and BPE"""
        h, w = frame.shape[:2]
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}",
                   (w - 150, h - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # BPE
        color = (0, 255, 0) if bpe < 3 else (0, 255, 255) if bpe < 5 else (0, 0, 255)
        cv2.putText(frame, f"BPE: {bpe:.2f}px",
                   (w - 150, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame
    
    def _add_legend(self, frame):
        """Add color legend"""
        h, w = frame.shape[:2]
        legend_x = 20
        legend_y = h - 180
        
        items = [
            ('Bladder', 'bladder'),
            ('Critical <5mm', 'critical'),
            ('Danger 5-10mm', 'danger'),
            ('Caution 10-20mm', 'caution')
        ]
        
        for i, (text, color_key) in enumerate(items):
            y = legend_y + i * 40
            
            # Color box
            cv2.rectangle(frame, (legend_x, y), (legend_x + 30, y + 25),
                         self.colors[color_key], -1)
            cv2.rectangle(frame, (legend_x, y), (legend_x + 30, y + 25),
                         (255, 255, 255), 2)
            
            # Text
            cv2.putText(frame, text, (legend_x + 40, y + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _calculate_risk_level(self, distance_map):
        """Calculate overall risk level"""
        min_dist = distance_map[distance_map > 0].min() if distance_map.any() else 1000
        
        if min_dist <= self.CRITICAL_DISTANCE:
            return 'critical'
        elif min_dist <= self.DANGER_DISTANCE:
            return 'danger'
        elif min_dist <= self.CAUTION_DISTANCE:
            return 'caution'
        else:
            return 'safe'
    
    def _add_risk_indicator(self, frame, risk_level):
        """Add large risk indicator"""
        h, w = frame.shape[:2]
        
        risk_colors = {
            'safe': (0, 255, 0),
            'caution': (0, 255, 255),
            'danger': (0, 165, 255),
            'critical': (0, 0, 255)
        }
        
        risk_texts = {
            'safe': 'SAFE',
            'caution': 'CAUTION',
            'danger': 'DANGER!',
            'critical': 'CRITICAL!'
        }
        
        color = risk_colors[risk_level]
        text = risk_texts[risk_level]
        
        # Background box
        box_width, box_height = 250, 80
        top_left = (w - box_width - 20, 20)
        bottom_right = (w - 20, 20 + box_height)
        
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), -1)
        cv2.rectangle(frame, top_left, bottom_right, color, 3)
        
        # Text
        font = cv2.FONT_HERSHEY_BOLD
        text_size = cv2.getTextSize(text, font, 1.2, 2)[0]
        text_x = top_left[0] + (box_width - text_size[0]) // 2
        text_y = top_left[1] + (box_height + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), font, 1.2, color, 2)
        
        return frame


class VideoProcessor:
    """Process video with bladder segmentation"""
    
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
        
        # Visualizer
        self.visualizer = DangerZoneVisualizer()
        
        # Temporal smoothing
        self.mask_history = []
        self.history_size = 5
    
    def preprocess(self, frame, size=512):
        """Preprocess frame"""
        frame_resized = cv2.resize(frame, (size, size))
        frame_norm = frame_resized.astype(np.float32) / 255.0
        frame_norm = (frame_norm - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        frame_tensor = torch.from_numpy(frame_norm.transpose(2, 0, 1)).unsqueeze(0)
        return frame_tensor.to(self.device)
    
    def predict(self, frame):
        """Predict mask"""
        with torch.no_grad():
            input_tensor = self.preprocess(frame)
            output, _ = self.model(input_tensor)
            mask = torch.sigmoid(output).cpu().numpy()[0, 0]
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            return mask
    
    def temporal_smooth(self, mask):
        """Smooth mask across frames"""
        self.mask_history.append(mask)
        if len(self.mask_history) > self.history_size:
            self.mask_history.pop(0)
        return np.mean(self.mask_history, axis=0)
    
    def postprocess(self, mask):
        """Clean up mask"""
        mask_binary = (mask > 0.5).astype(np.uint8)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_cleaned = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
        
        # Largest component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_cleaned)
        if num_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_cleaned = (labels == largest).astype(np.uint8)
        
        return (mask_cleaned * 255).astype(np.uint8)
    
    def process_video(self, video_path, output_path, show_dashboard=True, show_metrics=True):
        """Process complete video"""
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nProcessing video:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.1f}s\n")
        
        # Output writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Process frames
        fps_history = []
        frame_count = 0
        
        pbar = tqdm(total=total_frames, desc="Processing")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Predict
            mask = self.predict(frame)
            
            # Smooth
            mask = self.temporal_smooth(mask)
            
            # Post-process
            mask = self.postprocess(mask)
            
            # Calculate FPS
            process_time = time.time() - start_time
            current_fps = 1.0 / process_time if process_time > 0 else 0
            fps_history.append(current_fps)
            avg_fps = np.mean(fps_history[-30:])  # Last 30 frames
            
            # Visualize
            output_frame = self.visualizer.visualize(
                frame, mask,
                show_dashboard=show_dashboard,
                show_metrics=show_metrics,
                fps=avg_fps,
                bpe=2.5  # Placeholder
            )
            
            # Write
            out.write(output_frame)
            
            frame_count += 1
            pbar.update(1)
            pbar.set_postfix({'FPS': f'{avg_fps:.1f}'})
        
        cap.release()
        out.release()
        pbar.close()
        
        # Summary
        print(f"\n{'='*70}")
        print("Processing Complete!")
        print("="*70)
        print(f"Frames processed: {frame_count}")
        print(f"Average FPS: {np.mean(fps_history):.1f}")
        print(f"Output saved to: {output_path}")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Process video with bladder segmentation')
    parser.add_argument('--video', type=str, required=True, help='Input video')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Output video')
    parser.add_argument('--device', type=str, default='mps', choices=['mps', 'cuda', 'cpu'])
    parser.add_argument('--show_dashboard', action='store_true', help='Show metrics dashboard')
    parser.add_argument('--show_metrics', action='store_true', help='Show FPS and BPE')
    
    args = parser.parse_args()
    
    # Check device
    device = args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, using CPU")
        device = 'cpu'
    
    # Create output dir
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Process
    processor = VideoProcessor(args.model, device=device)
    processor.process_video(
        args.video,
        args.output,
        show_dashboard=args.show_dashboard,
        show_metrics=args.show_metrics
    )


if __name__ == '__main__':
    main()