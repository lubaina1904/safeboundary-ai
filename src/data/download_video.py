"""
YouTube Video Downloader for SafeBoundary AI
Downloads the laparoscopic surgery video
"""

import os
import sys
import argparse
from pathlib import Path
import yt_dlp


def download_video(url, output_path, quality='best'):
    """
    Download video from YouTube
    
    Args:
        url: YouTube video URL
        output_path: Where to save the video
        quality: Video quality ('best', 'worst', or specific height like '720')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    ydl_opts = {
        'format': f'{quality}[ext=mp4]/best[ext=mp4]/best',
        'outtmpl': str(output_path),
        'quiet': False,
        'no_warnings': False,
        'progress_hooks': [download_progress_hook],
    }
    
    print(f"📥 Downloading video from: {url}")
    print(f"📁 Saving to: {output_path}")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            duration = info.get('duration', 0)
            title = info.get('title', 'Unknown')
            
            print(f"\n✅ Download complete!")
            print(f"📹 Title: {title}")
            print(f"⏱️  Duration: {duration // 60} minutes {duration % 60} seconds")
            print(f"💾 File: {output_path}")
            
            return output_path
            
    except Exception as e:
        print(f"\n❌ Error downloading video: {e}")
        sys.exit(1)


def download_progress_hook(d):
    """Display download progress"""
    if d['status'] == 'downloading':
        percent = d.get('_percent_str', 'N/A')
        speed = d.get('_speed_str', 'N/A')
        eta = d.get('_eta_str', 'N/A')
        
        print(f"\r⬇️  Progress: {percent} | Speed: {speed} | ETA: {eta}", end='', flush=True)
    
    elif d['status'] == 'finished':
        print("\n🔄 Processing video...")


def verify_video(video_path):
    """Verify that video file is valid"""
    import cv2
    
    if not os.path.exists(video_path):
        return False, "File does not exist"
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "Cannot open video file"
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    print(f"\n📊 Video Information:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Frames: {frame_count}")
    print(f"   Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    
    return True, "Video is valid"


def main():
    parser = argparse.ArgumentParser(
        description='Download laparoscopic surgery video from YouTube'
    )
    parser.add_argument(
        '--url',
        type=str,
        default='https://youtu.be/TUKr2C5E8jA',
        help='YouTube video URL'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/surgery_video.mp4',
        help='Output video file path'
    )
    parser.add_argument(
        '--quality',
        type=str,
        default='best',
        choices=['best', 'worst', '720', '1080'],
        help='Video quality'
    )
    
    args = parser.parse_args()
    
    # Download video
    video_path = download_video(args.url, args.output, args.quality)
    
    # Verify video
    valid, message = verify_video(video_path)
    if not valid:
        print(f"❌ {message}")
        sys.exit(1)
    
    print(f"\n✅ All done! Video ready for processing.")


if __name__ == '__main__':
    main()