#!/usr/bin/env python3
"""Create a bar chart showing the effect of ensembling and TTA on leaderboard scores."""

import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def create_ensemble_tta_chart(output_path):
    """Create a bar chart comparing ensemble results with and without TTA."""
    # Data from results.md where we have both with and without TTA comparisons
    # Group 1: EffB4 + Res50d (clean) - with/without TTA
    # Group 2: EffB4 + Res50d + Res200d (baseline) - with/without TTA
    
    models = [
        'EffB4 + Res50d\n(clean)',
        'EffB4 + Res50d +\nRes200d (baseline)'
    ]
    
    # Public LB scores (from results.md)
    # EffB4 + Res50d (clean): no TTA 0.9879, with TTA 0.9879
    # EffB4 + Res50d + Res200d: no TTA 0.9875, with TTA 0.9877
    public_no_tta = [0.9879, 0.9875]
    public_with_tta = [0.9879, 0.9877]
    
    # Private LB scores (from results.md)
    # EffB4 + Res50d (clean): no TTA 0.9843, with TTA 0.9845
    # EffB4 + Res50d + Res200d: no TTA 0.9841, with TTA 0.9843
    private_no_tta = [0.9843, 0.9841]
    private_with_tta = [0.9845, 0.9843]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Effect of Ensembling and TTA on Leaderboard Scores', fontsize=14, fontweight='bold')
    
    # Public LB comparison
    bars1 = ax1.bar(x - width/2, public_no_tta, width, label='No TTA', color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar(x + width/2, public_with_tta, width, label='With TTA', color='#A23B72', alpha=0.8)
    
    ax1.set_ylabel('Public Leaderboard Score', fontsize=11)
    ax1.set_title('Public Leaderboard', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0.9870, 0.9882])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=8)
    
    # Private LB comparison
    bars3 = ax2.bar(x - width/2, private_no_tta, width, label='No TTA', color='#2E86AB', alpha=0.8)
    bars4 = ax2.bar(x + width/2, private_with_tta, width, label='With TTA', color='#A23B72', alpha=0.8)
    
    ax2.set_ylabel('Private Leaderboard Score', fontsize=11)
    ax2.set_title('Private Leaderboard', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0.9835, 0.9850])
    
    # Add value labels on bars
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved chart to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Create ensemble and TTA comparison chart')
    parser.add_argument('--output', type=str, default='report/ensemble_tta_chart.png', 
                       help='Output path for the chart')
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_ensemble_tta_chart(output_path)

if __name__ == '__main__':
    main()

