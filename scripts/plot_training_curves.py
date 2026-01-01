#!/usr/bin/env python3
"""Plot training/validation accuracy and loss curves from history JSON files."""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_history(json_path: Path):
    """Load history from JSON file and compute average across folds per epoch."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    history = data['history']
    
    # Group by epoch
    epoch_data = defaultdict(lambda: {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []})
    
    for entry in history:
        epoch = entry['epoch']
        epoch_data[epoch]['train_loss'].append(entry['train_loss'])
        epoch_data[epoch]['train_acc'].append(entry['train_acc'])
        epoch_data[epoch]['val_loss'].append(entry['val_loss'])
        epoch_data[epoch]['val_acc'].append(entry['val_acc'])
    
    # Compute averages
    epochs = sorted(epoch_data.keys())
    train_loss = [np.mean(epoch_data[e]['train_loss']) for e in epochs]
    train_acc = [np.mean(epoch_data[e]['train_acc']) for e in epochs]
    val_loss = [np.mean(epoch_data[e]['val_loss']) for e in epochs]
    val_acc = [np.mean(epoch_data[e]['val_acc']) for e in epochs]
    
    return epochs, train_loss, train_acc, val_loss, val_acc

def plot_curves(baseline_path: Path, finetune_path: Path, output_path: Path):
    """Plot training curves for baseline and fine-tuning."""
    # Load data
    baseline_epochs, baseline_train_loss, baseline_train_acc, baseline_val_loss, baseline_val_acc = load_history(baseline_path)
    finetune_epochs, finetune_train_loss, finetune_train_acc, finetune_val_loss, finetune_val_acc = load_history(finetune_path)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training/Validation Accuracy and Loss Curves', fontsize=14, fontweight='bold')
    
    # Plot accuracy curves
    ax1 = axes[0, 0]
    ax1.plot(baseline_epochs, baseline_train_acc, 'b-', label='Baseline Train', linewidth=2)
    ax1.plot(baseline_epochs, baseline_val_acc, 'b--', label='Baseline Val', linewidth=2)
    ax1.plot(finetune_epochs, finetune_train_acc, 'r-', label='Fine-tune Train', linewidth=2)
    ax1.plot(finetune_epochs, finetune_val_acc, 'r--', label='Fine-tune Val', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss curves
    ax2 = axes[0, 1]
    ax2.plot(baseline_epochs, baseline_train_loss, 'b-', label='Baseline Train', linewidth=2)
    ax2.plot(baseline_epochs, baseline_val_loss, 'b--', label='Baseline Val', linewidth=2)
    ax2.plot(finetune_epochs, finetune_train_loss, 'r-', label='Fine-tune Train', linewidth=2)
    ax2.plot(finetune_epochs, finetune_val_loss, 'r--', label='Fine-tune Val', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot accuracy comparison (zoomed)
    ax3 = axes[1, 0]
    ax3.plot(baseline_epochs, baseline_val_acc, 'b--', label='Baseline Val', linewidth=2)
    ax3.plot(finetune_epochs, finetune_val_acc, 'r--', label='Fine-tune Val', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Validation Accuracy')
    ax3.set_title('Validation Accuracy Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot loss comparison (zoomed)
    ax4 = axes[1, 1]
    ax4.plot(baseline_epochs, baseline_val_loss, 'b--', label='Baseline Val', linewidth=2)
    ax4.plot(finetune_epochs, finetune_val_loss, 'r--', label='Fine-tune Val', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Loss')
    ax4.set_title('Validation Loss Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot training curves from history JSON files')
    parser.add_argument('--baseline', type=str, required=True, help='Path to baseline history JSON')
    parser.add_argument('--finetune', type=str, required=True, help='Path to fine-tuning history JSON')
    parser.add_argument('--output', type=str, default='report/training_curves.png', help='Output path for the plot')
    
    args = parser.parse_args()
    
    baseline_path = Path(args.baseline)
    finetune_path = Path(args.finetune)
    output_path = Path(args.output)
    
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline history file not found: {baseline_path}")
    if not finetune_path.exists():
        raise FileNotFoundError(f"Fine-tuning history file not found: {finetune_path}")
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plot_curves(baseline_path, finetune_path, output_path)

if __name__ == '__main__':
    main()

