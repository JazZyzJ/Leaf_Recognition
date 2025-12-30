

# ResNet200d Clean Finetune
python src/train.py --config configs/resnet200d_clean_finetune.yaml --resume_dir weights

# ResNet50d Baseline Clean Finetune
python src/train.py --config configs/resnet50d_baseline_clean_finetune.yaml --resume_dir weights

# EfficientNet-B4 Clean Finetune
python src/train.py --config configs/effnet_b4_clean_finetune.yaml --resume_dir weights