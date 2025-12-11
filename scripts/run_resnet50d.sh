#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/resnet50d_baseline.yaml}
DEVICE=${2:-cuda}
LOG_PATH=${LOG_PATH:-logs/experiment_results.jsonl}
ADDITIONAL_LOG_ARGS=()

if [[ -n "${LB_PUBLIC:-}" ]]; then
  ADDITIONAL_LOG_ARGS+=("--lb_public" "${LB_PUBLIC}")
fi
if [[ -n "${LB_PRIVATE:-}" ]]; then
  ADDITIONAL_LOG_ARGS+=("--lb_private" "${LB_PRIVATE}")
fi
if [[ -n "${NOTES:-}" ]]; then
  ADDITIONAL_LOG_ARGS+=("--notes" "${NOTES}")
fi

python src/train.py --config "${CONFIG}" --device "${DEVICE}"

EXP_NAME=$(python - <<PY
import yaml
with open("${CONFIG}", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["experiment"]["name"])
PY
)

SUMMARY_PATH="logs/${EXP_NAME}_summary.json"
if [[ ! -f "${SUMMARY_PATH}" ]]; then
  echo "Expected summary file not found: ${SUMMARY_PATH}" >&2
  exit 1
fi

python src/log_results.py --config "${CONFIG}" --summary "${SUMMARY_PATH}" --log_path "${LOG_PATH}" "${ADDITIONAL_LOG_ARGS[@]}"
