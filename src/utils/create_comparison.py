"""
Create side-by-side comparison video
Original vs SafeBoundary AI processed
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm


def create_comparison_video(original_path, processed_path, output_path):
    """
    Create side-by-side comparison video
    
    Args:
        original_path: Path to original video
        processed_path: Path to processed video with annotations
        output_path: Path to save comparison video
    """
    # Open videos
    cap_orig = cv2.VideoCapture(str(original_path))
    cap_proc = cv2.VideoCapture(str(processed_path))
    
    if not cap_orig.isOpened():
        raise ValueError(f"Cannot open original video: {original_path}")
    if not cap_proc.isOpened():
        raise ValueError(f"Cannot open processed video: {processed_path}")
    
    # Get video properties
    fps = int(cap_orig.get(cv2.CAP_PROP_FPS))
    width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output video (double width for side-by-side)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width * 2, height)
    )
    
    print(f"\nCreating comparison video:")
    print(f"  Original: {original_path}")
    print(f"  Processed: {processed_path}")
    print(f"  Output: {output_path}")
    print(f"  Resolution: {width * 2}x{height}")
    print(f"  Frames: {total_frames}\n")
    
    for i in tqdm(range(total_frames), desc="Creating comparison"):
        ret_orig, frame_orig = cap_orig.read()
        ret_proc, frame_proc = cap_proc.read()
        
        if not ret_orig or not ret_proc:
            break
        
        # Add labels
        frame_orig_labeled = frame_orig.copy()
        frame_proc_labeled = frame_proc.copy()
        
        # Add "ORIGINAL" label
        cv2.rectangle(frame_orig_labeled, (10, 10), (250, 70), (0, 0, 0), -1)
        cv2.rectangle(frame_orig_labeled, (10, 10), (250, 70), (255, 255, 255), 2)
        cv2.putText(
            frame_orig_labeled,
            "ORIGINAL",
            (30, 50),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        
        # Add "SAFEBOUNDARY AI" label
        cv2.rectangle(frame_proc_labeled, (10, 10), (350, 70), (0, 0, 0), -1)
        cv2.rectangle(frame_proc_labeled, (10, 10), (350, 70), (0, 255, 0), 2)
        cv2.putText(
            frame_proc_labeled,
            "SAFEBOUNDARY AI",
            (30, 50),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        
        # Add divider line in the middle
        combined = np.hstack([frame_orig_labeled, frame_proc_labeled])
        
        # Draw vertical divider
        cv2.line(combined, (width, 0), (width, height), (255, 255, 255), 3)
        
        # Write frame
        out.write(combined)
    
    cap_orig.release()
    cap_proc.release()
    out.release()
    
    print(f"\n✓ Comparison video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create side-by-side comparison video'
    )
    parser.add_argument('--original', type=str, required=True,
                       help='Path to original video')
    parser.add_argument('--processed', type=str, required=True,
                       help='Path to processed video')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for comparison video')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Create comparison
    create_comparison_video(args.original, args.processed, args.output)


if __name__ == '__main__':
    main()
