#!/usr/bin/env python3
"""
Download pretrained model weights to local cache.
This script should be run in an environment with internet access.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import timm
import torch
import yaml


def download_model_weights(model_name: str, num_classes: int = 1000) -> None:
    """Download pretrained weights for a model."""
    print(f"Downloading pretrained weights for {model_name}...")
    try:
        # Create model with pretrained=True to trigger download
        model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
        )
        print(f"✓ Successfully downloaded and cached weights for {model_name}")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Print cache location info
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        if cache_dir.exists():
            print(f"  Cache directory: {cache_dir}")
        
    except Exception as e:
        print(f"✗ Error downloading weights: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pretrained model weights")
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (e.g., resnet50d)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML file (will read model_name from config)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of classes (default: 1000, will be overridden in training)",
    )
    
    args = parser.parse_args()
    
    if args.config:
        # Load model name from config
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        model_name = config["model"]["model_name"]
        print(f"Loaded model name from config: {model_name}")
    elif args.model:
        model_name = args.model
    else:
        parser.error("Either --model or --config must be provided")
    
    download_model_weights(model_name, args.num_classes)


if __name__ == "__main__":
    main()

