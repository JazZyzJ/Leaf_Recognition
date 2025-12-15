#!/usr/bin/env bash
set -euo pipefail

# Simple sweep for EffNet-B4 finetune from existing checkpoints.
# Customize arrays below to try more LRs/epochs.
LRS=(5e-5 1e-4)
EPOCHS=(3 5)
BASE_CONFIG=${BASE_CONFIG:-configs/effnet_b4_finetune.yaml}
RESUME_DIR=${RESUME_DIR:-weights}
DEVICE=${DEVICE:-cuda}

for LR in "${LRS[@]}"; do
  for EP in "${EPOCHS[@]}"; do
    TMP_CFG=$(mktemp /tmp/effb4_ft_XXXX.yaml)
    python - <<PY
import yaml, copy
from pathlib import Path
base = yaml.safe_load(Path("${BASE_CONFIG}").read_text())
exp_name = f"effnet_b4_ft_lr{LR}_e{EP}"
base["experiment"]["name"] = exp_name
base["experiment"]["save_folds_csv"] = f"train_folds_{exp_name}.csv"
base["training"]["lr"] = float(${LR})
base["training"]["num_epochs"] = int(${EP})
base["training"]["t_max"] = int(${EP})
base["training"]["aug_desc"] = f"Finetune lr={LR} ep={EP}"
Path("${TMP_CFG}").write_text(yaml.safe_dump(base))
print(TMP_CFG)
PY
    echo "Running finetune LR=${LR}, epochs=${EP} (config: ${TMP_CFG})"
    python src/train.py --config "${TMP_CFG}" --device "${DEVICE}" --resume_dir "${RESUME_DIR}"
    EXP_NAME=$(python - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("${TMP_CFG}").read_text())
print(cfg["experiment"]["name"])
PY
)
    SUMMARY_PATH="logs/${EXP_NAME}_summary.json"
    if [[ -f "${SUMMARY_PATH}" ]]; then
      python src/log_results.py --config "${TMP_CFG}" --summary "${SUMMARY_PATH}" --notes "ft sweep" || true
    fi
  done
done
