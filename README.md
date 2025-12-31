# Leaf Classification Project

PyTorch pipeline for the Kaggle "Classification of Leaves" competition. The repository provides
data loading, K-fold training, inference/ensembling utilities.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
python src/train.py --config configs/resnet50d_baseline.yaml --device cuda
# EfficientNet-B4 (EMA enabled in config by default)
python src/train.py --config configs/effnet_b4.yaml --device cuda
# ResNet200d stronger backbone
python src/train.py --config configs/resnet200d.yaml --device cuda
```

Key artifacts are written to `logs/`, `weights/`, `oof/`, and the augmented `train_folds.csv` file used
for reproducibility.
Configs can toggle extras such as Exponential Moving Average (`training.ema.*`) and Mixup/CutMix (`training.mixup.*`).

### Clean-data and fine-tuning configs

```bash
# Train on the cleaned dataset
python src/train.py --config configs/effnet_b4_clean.yaml --device cuda
python src/train.py --config configs/resnet50d_baseline_clean.yaml --device cuda
python src/train.py --config configs/resnet200d_clean.yaml --device cuda

# Low-LR fine-tuning from clean checkpoints
python src/train.py --config configs/effnet_b4_clean_finetune.yaml --device cuda
python src/train.py --config configs/resnet50d_baseline_clean_finetune.yaml --device cuda
python src/train.py --config configs/resnet200d_clean_finetune.yaml --device cuda
```

## Inference

Average fold predictions and generate a submission:

```bash
python src/inference.py --config configs/resnet50d_baseline.yaml --device cuda
# Multi-model ensemble with horizontal-flip TTA
python src/inference.py \
  --config configs/resnet50d_baseline.yaml \
  --config configs/effnet_b4.yaml \
  --tta hflip \
  --device cuda \
  --output_name resnet50d_effnetb4_ens.csv
```

Outputs are saved under `submissions/`. Custom weight files can be provided via `--weights-pattern` or a
comma-separated list passed to the same flag; repeat `--config` and `--weights-pattern` to ensemble different experiments.

## Experiment logging

After finishing a training or inference run (e.g., on the remote 4090 machine), capture the metrics with:

```bash
python src/log_results.py --config configs/resnet50d_baseline.yaml \
  --summary logs/resnet50d_baseline_summary.json \
  --lb_public 0.9601 --notes "5-fold baseline"
```

This appends a JSON line to `logs/experiment_results.jsonl`. Once the file is copied back to this workspace,
regenerate `results.md` via:

```bash
python src/update_results_md.py
```

The script reads the JSONL log (keeping the latest run per experiment by default) and rebuilds the Markdown table.



### Automated remote run

On the remote 4090 box, the entire train→log flow can be triggered with:

```bash
./scripts/run_resnet50d.sh configs/resnet50d_baseline.yaml cuda
```

Optional environment variables:

```bash
LB_PUBLIC=0.9610 NOTES="resnet50d baseline @512" ./scripts/run_resnet50d.sh
```

This exports the same JSON log so that syncing the `logs/` directory back to this repo + running
`python src/update_results_md.py` keeps `results.md` up to date automatically.
The script accepts any config path, so it works for EfficientNet-B4 and future experiments as well.

## Repository layout

```
classify-leaves/
├── configs/
│   └── resnet50d_baseline.yaml
├── src/
│   ├── dataset.py
│   ├── inference.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
├── logs/  # training metrics & label mappings
├── weights/  # fold checkpoints
├── oof/  # out-of-fold predictions
├── submissions/
├── train.csv / test.csv
└── images/
```
