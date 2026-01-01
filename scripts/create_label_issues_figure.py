#!/usr/bin/env python3
"""Create a figure showing representative label issue samples."""

import matplotlib.pyplot as plt
from PIL import Image
import argparse
from pathlib import Path

def create_label_issues_figure(image_paths, labels, predicted_labels, output_path):
    """Create a figure with 4 images in a row showing label issues."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    fig.suptitle('Representative Label Issue Samples', fontsize=12, fontweight='bold')
    
    for idx, (img_path, label, pred_label) in enumerate(zip(image_paths, labels, predicted_labels)):
        ax = axes[idx]
        
        # Load and display image
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
        
        # Add title with label info
        title = f'True: {label}\nPred: {pred_label}'
        ax.set_title(title, fontsize=9, pad=5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Create label issues figure')
    parser.add_argument('--output', type=str, default='report/label_issues_samples.png', 
                       help='Output path for the figure')
    
    args = parser.parse_args()
    
    # First 4 samples from CSV
    image_paths = [
        'images/10112.jpg',
        'images/6215.jpg',
        'images/4163.jpg',
        'images/7408.jpg'
    ]
    
    labels = [
        'larix_decidua',
        'magnolia_stellata',
        'pinus_virginiana',
        'pinus_virginiana'
    ]
    
    predicted_labels = [
        'catalpa_speciosa',
        'magnolia_tripetala',
        'prunus_virginiana',
        'prunus_virginiana'
    ]
    
    # Convert to Path objects and check existence
    base_path = Path('.')
    image_paths_full = [base_path / path for path in image_paths]
    
    for path in image_paths_full:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_label_issues_figure(image_paths_full, labels, predicted_labels, output_path)

if __name__ == '__main__':
    main()

