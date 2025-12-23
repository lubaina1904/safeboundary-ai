"""
Frame Extraction for SafeBoundary AI
Intelligently extracts high-quality frames from surgical video
"""

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import json


class FrameExtractor:
    """
    Intelligent frame extraction with quality filtering
    """
    def __init__(self, 
                 quality_threshold=0.3,
                 min_frame_diff=0.05,
                 blur_threshold=100):
        """
        Args:
            quality_threshold: Minimum quality score (0-1)
            min_frame_diff: Minimum difference between frames (avoid duplicates)
            blur_threshold: Minimum Laplacian variance (higher = sharper)
        """
        self.quality_threshold = quality_threshold
        self.min_frame_diff = min_frame_diff
        self.blur_threshold = blur_threshold
        
        self.prev_frame = None
    
    def calculate_blur(self, image):
        """
        Calculate blur using Laplacian variance
        Higher values = sharper image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    
    def calculate_brightness(self, image):
        """
        Calculate average brightness
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray.mean()
    
    def calculate_frame_difference(self, frame1, frame2):
        """
        Calculate difference between two frames
        Returns normalized difference (0-1)
        """
        if frame1 is None or frame2 is None:
            return 1.0
        
        # Resize for faster comparison
        frame1_small = cv2.resize(frame1, (160, 120))
        frame2_small = cv2.resize(frame2, (160, 120))
        
        # Calculate MSE
        diff = cv2.absdiff(frame1_small, frame2_small)
        diff = diff.astype(np.float32) / 255.0
        mse = np.mean(diff ** 2)
        
        return mse
    
    def is_too_dark(self, image, min_brightness=30):
        """Check if frame is too dark"""
        brightness = self.calculate_brightness(image)
        return brightness < min_brightness
    
    def is_too_bright(self, image, max_brightness=225):
        """Check if frame is overexposed"""
        brightness = self.calculate_brightness(image)
        return brightness > max_brightness
    
    def calculate_quality_score(self, image):
        """
        Calculate overall quality score (0-1)
        Considers: sharpness, brightness, contrast
        """
        # Sharpness (blur)
        blur = self.calculate_blur(image)
        blur_score = min(blur / 500.0, 1.0)  # Normalize
        
        # Brightness (0 and 255 are bad, 127 is good)
        brightness = self.calculate_brightness(image)
        brightness_score = 1.0 - abs(brightness - 127) / 127.0
        
        # Contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()
        contrast_score = min(contrast / 50.0, 1.0)
        
        # Combined score
        quality = (blur_score * 0.5 + 
                  brightness_score * 0.3 + 
                  contrast_score * 0.2)
        
        return quality
    
    def is_good_frame(self, image):
        """
        Determine if frame is good quality
        """
        # Check blur
        blur = self.calculate_blur(image)
        if blur < self.blur_threshold:
            return False, f"Too blurry (blur={blur:.1f})"
        
        # Check if too dark or bright
        if self.is_too_dark(image):
            return False, "Too dark"
        
        if self.is_too_bright(image):
            return False, "Too bright"
        
        # Check quality score
        quality = self.calculate_quality_score(image)
        if quality < self.quality_threshold:
            return False, f"Low quality (score={quality:.2f})"
        
        # Check if too similar to previous frame
        if self.prev_frame is not None:
            diff = self.calculate_frame_difference(image, self.prev_frame)
            if diff < self.min_frame_diff:
                return False, f"Too similar to previous (diff={diff:.3f})"
        
        return True, f"Good quality (score={quality:.2f})"
    
    def extract_frames(self, video_path, output_dir, num_frames=200, visualize=False):
        """
        Extract high-quality frames from video
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save frames
            num_frames: Target number of frames to extract
            visualize: Show extracted frames
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        
        print(f"\n{'='*70}")
        print("Video Information")
        print("="*70)
        print(f"File: {video_path.name}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")
        print(f"Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
        print(f"Target frames: {num_frames}")
        print("="*70 + "\n")
        
        # Calculate sampling interval
        # We'll check more frames than needed to ensure we get enough good ones
        check_interval = max(1, total_frames // (num_frames * 3))
        
        extracted_frames = []
        frame_metadata = []
        
        frame_idx = 0
        checked = 0
        accepted = 0
        
        pbar = tqdm(total=num_frames, desc="Extracting frames")
        
        while accepted < num_frames and frame_idx < total_frames:
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret, frame = cap.read()
            if not ret:
                break
            
            checked += 1
            
            # Check quality
            is_good, reason = self.is_good_frame(frame)
            
            if is_good:
                # Save frame
                timestamp = frame_idx / fps
                frame_filename = f"frame_{accepted:04d}_t{timestamp:.2f}s.jpg"
                frame_path = output_dir / frame_filename
                
                cv2.imwrite(str(frame_path), frame, 
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Store metadata
                quality_score = self.calculate_quality_score(frame)
                blur_score = self.calculate_blur(frame)
                
                frame_metadata.append({
                    'filename': frame_filename,
                    'frame_index': frame_idx,
                    'timestamp': timestamp,
                    'quality_score': float(quality_score),
                    'blur_score': float(blur_score),
                    'brightness': float(self.calculate_brightness(frame))
                })
                
                extracted_frames.append(frame)
                self.prev_frame = frame.copy()
                
                accepted += 1
                pbar.update(1)
                
                # Visualize if requested
                if visualize:
                    vis = frame.copy()
                    cv2.putText(vis, f"Frame {accepted}/{num_frames}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2)
                    cv2.putText(vis, f"Quality: {quality_score:.2f}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Extracted Frame', vis)
                    cv2.waitKey(1)
            
            # Move to next frame to check
            frame_idx += check_interval
        
        cap.release()
        pbar.close()
        
        if visualize:
            cv2.destroyAllWindows()
        
        # Save metadata
        metadata_path = output_dir / 'extraction_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump({
                'video_file': str(video_path),
                'total_frames': total_frames,
                'fps': fps,
                'duration': duration,
                'frames_checked': checked,
                'frames_extracted': accepted,
                'quality_threshold': self.quality_threshold,
                'blur_threshold': self.blur_threshold,
                'frames': frame_metadata
            }, f, indent=2)
        
        # Summary
        print(f"\n{'='*70}")
        print("Extraction Complete!")
        print("="*70)
        print(f"Frames checked: {checked}")
        print(f"Frames extracted: {accepted}")
        print(f"Acceptance rate: {accepted/checked*100:.1f}%")
        print(f"Output directory: {output_dir}")
        print(f"Metadata saved: {metadata_path}")
        
        if accepted < num_frames:
            print(f"\n⚠️  Warning: Only extracted {accepted}/{num_frames} frames")
            print(f"   Try lowering quality_threshold or blur_threshold")
        
        print("="*70 + "\n")
        
        # Create quality report
        self.create_quality_report(frame_metadata, output_dir)
        
        return extracted_frames, frame_metadata
    
    def create_quality_report(self, metadata, output_dir):
        """Create visual quality report"""
        if not metadata:
            return
        
        quality_scores = [f['quality_score'] for f in metadata]
        blur_scores = [f['blur_score'] for f in metadata]
        brightness = [f['brightness'] for f in metadata]
        
        report = "\n" + "="*70 + "\n"
        report += "QUALITY REPORT\n"
        report += "="*70 + "\n\n"
        
        report += f"Quality Scores:\n"
        report += f"  Mean: {np.mean(quality_scores):.3f}\n"
        report += f"  Min:  {np.min(quality_scores):.3f}\n"
        report += f"  Max:  {np.max(quality_scores):.3f}\n\n"
        
        report += f"Blur Scores:\n"
        report += f"  Mean: {np.mean(blur_scores):.1f}\n"
        report += f"  Min:  {np.min(blur_scores):.1f}\n"
        report += f"  Max:  {np.max(blur_scores):.1f}\n\n"
        
        report += f"Brightness:\n"
        report += f"  Mean: {np.mean(brightness):.1f}\n"
        report += f"  Min:  {np.min(brightness):.1f}\n"
        report += f"  Max:  {np.max(brightness):.1f}\n"
        
        report += "="*70 + "\n"
        
        print(report)
        
        # Save report
        report_path = output_dir / 'quality_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)


def main():
    parser = argparse.ArgumentParser(
        description='Extract high-quality frames from surgical video'
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path to input video file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/frames',
        help='Output directory for extracted frames'
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=200,
        help='Number of frames to extract'
    )
    parser.add_argument(
        '--quality_threshold',
        type=float,
        default=0.3,
        help='Minimum quality score (0-1)'
    )
    parser.add_argument(
        '--blur_threshold',
        type=float,
        default=100,
        help='Minimum blur score (higher = sharper)'
    )
    parser.add_argument(
        '--min_frame_diff',
        type=float,
        default=0.05,
        help='Minimum difference between frames (avoid duplicates)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show extracted frames in real-time'
    )
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = FrameExtractor(
        quality_threshold=args.quality_threshold,
        min_frame_diff=args.min_frame_diff,
        blur_threshold=args.blur_threshold
    )
    
    # Extract frames
    frames, metadata = extractor.extract_frames(
        video_path=args.video,
        output_dir=args.output,
        num_frames=args.num_frames,
        visualize=args.visualize
    )
    
    print("✅ Frame extraction complete!")
    print(f"   Next step: Annotate frames in {args.output}")


if __name__ == '__main__':
    main()