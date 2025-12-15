#!/usr/bin/env bash
set -euo pipefail

# Paths to base weights for resuming
RESUME_DIR=${RESUME_DIR:-weights}
DEVICE=${DEVICE:-cuda}

# Finetune EffNet-B4 from existing fold weights (expects effnet_b4_baseline_fold*.pth in RESUME_DIR)
python src/train.py --config configs/effnet_b4_finetune.yaml --device "${DEVICE}" --resume_dir "${RESUME_DIR}" || true

# Log the finetune run (uses summary written by training)
python src/log_results.py --config configs/effnet_b4_finetune.yaml --summary logs/effnet_b4_finetune_summary.json --notes "finetune from baseline" || true

# Generate ensembles
# EffNet-B4 finetune + ResNet50d (with TTA)
python src/inference.py \
  --config configs/resnet50d_baseline.yaml \
  --config configs/effnet_b4_finetune.yaml \
  --tta hflip \
  --device "${DEVICE}" \
  --output_name effnetb4_finetune_resnet50d_ens.csv || true

# EffNet-B4 finetune + ResNet200d (with TTA)
python src/inference.py \
  --config configs/resnet200d.yaml \
  --config configs/effnet_b4_finetune.yaml \
  --tta hflip \
  --device "${DEVICE}" \
  --output_name effnetb4_finetune_resnet200d_ens.csv || true

# Three-model ensemble (optional)
python src/inference.py \
  --config configs/resnet50d_baseline.yaml \
  --config configs/resnet200d.yaml \
  --config configs/effnet_b4_finetune.yaml \
  --tta hflip \
  --device "${DEVICE}" \
  --output_name effnetb4_finetune_resnet50d_resnet200d_ens.csv || true
