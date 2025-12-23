"""
Plot evaluation metrics for presentation
Creates visualizations of model performance
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path


def plot_metrics(report_path, output_path):
    """
    Create comprehensive metrics visualization
    
    Args:
        report_path: Path to evaluation report JSON
        output_path: Path to save plot
    """
    # Load report
    with open(report_path, 'r') as f:
        data = json.load(f)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 10)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SafeBoundary AI - Performance Metrics', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Segmentation Quality (Bar Chart)
    ax1 = axes[0, 0]
    metrics = ['IoU', 'Dice', 'Precision', 'Recall']
    values = [
        data.get('iou_mean', 0),
        data.get('dice_mean', 0),
        data.get('precision_mean', 0),
        data.get('recall_mean', 0)
    ]
    errors = [
        data.get('iou_std', 0),
        data.get('dice_std', 0),
        data.get('precision_std', 0),
        data.get('recall_std', 0)
    ]
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = ax1.bar(metrics, values, yerr=errors, capsize=5, 
                   color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Segmentation Quality Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.1])
    ax1.axhline(y=0.9, color='green', linestyle='--', 
                label='Target (0.90)', alpha=0.7)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Clinical Safety - BPE (Gauge-style)
    ax2 = axes[0, 1]
    bpe_mean = data.get('bpe_mean', 0)
    bpe_max = 10  # Max scale
    
    # Create colored zones
    zones = [
        (0, 3, '#2ecc71', 'Excellent'),
        (3, 5, '#f39c12', 'Good'),
        (5, 7, '#e67e22', 'Acceptable'),
        (7, 10, '#e74c3c', 'Poor')
    ]
    
    for start, end, color, label in zones:
        ax2.barh(0, end - start, left=start, height=0.5, 
                color=color, alpha=0.6, label=label)
    
    # Add BPE marker
    ax2.scatter(bpe_mean, 0, s=300, c='black', marker='|', 
               linewidths=5, zorder=10, label=f'BPE: {bpe_mean:.2f}px')
    
    ax2.set_xlim([0, bpe_max])
    ax2.set_ylim([-0.5, 0.5])
    ax2.set_xlabel('Boundary Proximity Error (pixels)', 
                  fontsize=12, fontweight='bold')
    ax2.set_title('Clinical Safety (BPE)', fontsize=14, fontweight='bold')
    ax2.set_yticks([])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Real-time Performance - FPS
    ax3 = axes[0, 2]
    fps_mean = data.get('fps_mean', 0)
    fps_target = 30
    
    # Speedometer-style
    categories = ['Current FPS', 'Target FPS']
    values_fps = [fps_mean, fps_target]
    colors_fps = ['#3498db' if fps_mean >= fps_target else '#e74c3c', '#2ecc71']
    
    bars = ax3.barh(categories, values_fps, color=colors_fps, 
                    alpha=0.8, edgecolor='black')
    
    for bar, value in zip(bars, values_fps):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2.,
                f'{value:.1f}',
                ha='left', va='center', fontweight='bold', fontsize=12)
    
    ax3.set_xlabel('Frames Per Second', fontsize=12, fontweight='bold')
    ax3.set_title('Real-time Performance', fontsize=14, fontweight='bold')
    ax3.set_xlim([0, max(fps_mean, fps_target) * 1.2])
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Metric Ranges (Box Plot style visualization)
    ax4 = axes[1, 0]
    
    metric_names = ['IoU', 'Dice', 'BPE']
    means = [
        data.get('iou_mean', 0),
        data.get('dice_mean', 0),
        data.get('bpe_mean', 0) / 10  # Normalize to 0-1
    ]
    mins = [
        data.get('iou_min', 0),
        data.get('dice_min', 0),
        data.get('bpe_min', 0) / 10
    ]
    maxs = [
        data.get('iou_max', 0),
        data.get('dice_max', 0),
        data.get('bpe_max', 0) / 10
    ]
    
    x_pos = np.arange(len(metric_names))
    ax4.errorbar(x_pos, means, 
                yerr=[np.array(means) - np.array(mins), 
                      np.array(maxs) - np.array(means)],
                fmt='o', markersize=10, capsize=8, capthick=2,
                color='#3498db', ecolor='#34495e', linewidth=2)
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(metric_names)
    ax4.set_ylabel('Score (normalized)', fontsize=12, fontweight='bold')
    ax4.set_title('Metric Variability', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.1])
    
    # 5. Overall Assessment (Status Card)
    ax5 = axes[1, 1]
    ax5.axis('off')
    
    # Determine status
    iou = data.get('iou_mean', 0)
    bpe = data.get('bpe_mean', 0)
    fps = data.get('fps_mean', 0)
    
    if iou > 0.90 and bpe < 3 and fps > 30:
        status = "EXCELLENT"
        status_color = '#2ecc71'
        emoji = "🏆"
        message = "Production Ready!"
    elif iou > 0.85 and bpe < 5 and fps > 25:
        status = "GOOD"
        status_color = '#3498db'
        emoji = "✓"
        message = "Meets Clinical Requirements"
    elif iou > 0.80 and bpe < 7 and fps > 20:
        status = "ACCEPTABLE"
        status_color = '#f39c12'
        emoji = "⚠"
        message = "Needs Improvement"
    else:
        status = "NEEDS WORK"
        status_color = '#e74c3c'
        emoji = "✗"
        message = "Below Clinical Standards"
    
    # Create status card
    rect = plt.Rectangle((0.1, 0.2), 0.8, 0.6, 
                         facecolor=status_color, alpha=0.2, 
                         edgecolor=status_color, linewidth=3)
    ax5.add_patch(rect)
    
    ax5.text(0.5, 0.7, f"{emoji} {status}", 
            ha='center', va='center', fontsize=28, 
            fontweight='bold', color=status_color)
    ax5.text(0.5, 0.4, message, 
            ha='center', va='center', fontsize=16,
            style='italic', color=status_color)
    
    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, 1])
    
    # 6. Key Statistics Summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    stats_text = f"""
    KEY STATISTICS
    ═══════════════════════════════
    
    Segmentation Quality:
    • Mean IoU:      {data.get('iou_mean', 0):.4f}
    • Mean Dice:     {data.get('dice_mean', 0):.4f}
    
    Clinical Safety:
    • Mean BPE:      {data.get('bpe_mean', 0):.2f} px
    • Median BPE:    {data.get('bpe_median', 0):.2f} px
    
    Performance:
    • Mean FPS:      {data.get('fps_mean', 0):.1f}
    • Median FPS:    {data.get('fps_median', 0):.1f}
    
    Reliability:
    • Precision:     {data.get('precision_mean', 0):.4f}
    • Recall:        {data.get('recall_mean', 0):.4f}
    """
    
    ax6.text(0.1, 0.9, stats_text, 
            ha='left', va='top', fontsize=11,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    ax6.set_xlim([0, 1])
    ax6.set_ylim([0, 1])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Metrics visualization saved to: {output_path}")
    
    # Also save as high-res PNG
    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ High-res PNG saved to: {png_path}")
    
    plt.close()


def create_training_curves(history_path, output_path):
    """
    Create training curves from training history
    
    Args:
        history_path: Path to training_history.json
        output_path: Path to save plot
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_losses'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # IoU
    axes[1].plot(epochs, history['val_ious'], 'g-', linewidth=2)
    axes[1].axhline(y=0.9, color='r', linestyle='--', label='Target (0.90)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].set_title('Validation IoU')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Dice
    axes[2].plot(epochs, history['val_dices'], 'orange', linewidth=2)
    axes[2].axhline(y=0.9, color='r', linestyle='--', label='Target (0.90)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Dice')
    axes[2].set_title('Validation Dice')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot evaluation metrics')
    parser.add_argument('--report', type=str, required=True,
                       help='Path to evaluation report JSON')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for plot')
    parser.add_argument('--history', type=str, default=None,
                       help='Optional: Path to training history JSON')
    
    args = parser.parse_args()
    
    # Plot metrics
    plot_metrics(args.report, args.output)
    
    # Plot training curves if history provided
    if args.history and Path(args.history).exists():
        history_output = Path(args.output).parent / 'training_curves.png'
        create_training_curves(args.history, history_output)


if __name__ == '__main__':
    main()