from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log experiment results for later reporting.")
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Config file(s). Repeat flag to log an ensemble (same as inference).",
    )
    parser.add_argument(
        "--configs",
        dest="config",
        action="append",
        help="Alias for --config to match inference-style multi-config usage.",
    )
    parser.add_argument(
        "--summary",
        action="append",
        required=False,
        help="Summary JSON per config (same order as --config). If one is provided, it is reused.",
    )
    parser.add_argument(
        "--cv",
        type=float,
        default=None,
        help="Override CV accuracy (otherwise read from summary).",
    )
    parser.add_argument("--lb_public", type=float, default=None, help="Public LB score if available.")
    parser.add_argument("--lb_private", type=float, default=None, help="Private LB score if available.")
    parser.add_argument("--notes", default="", help="Optional free-form notes for the run.")
    parser.add_argument(
        "--log_path",
        default="logs/experiment_results.jsonl",
        help="Where to append the experiment metadata.",
    )
    return parser.parse_args()


def load_summary(summary_path: Path) -> Dict[str, Any]:
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_entry(
    configs: List[Dict[str, Any]],
    args: argparse.Namespace,
    summaries: List[Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    multi = len(configs) > 1

    experiments = [cfg["experiment"]["name"] for cfg in configs]
    model_names = [cfg["model"]["model_name"] for cfg in configs]
    img_sizes = [cfg["training"]["img_size"] for cfg in configs]
    aug_notes = [cfg["training"].get("aug_desc", "") for cfg in configs]
    epochs = [cfg["training"]["num_epochs"] for cfg in configs]
    batch_sizes = [cfg["training"]["batch_size"] for cfg in configs]

    cv_list: List[float] = []
    for summary in summaries:
        if summary is not None and "cv_accuracy" in summary:
            cv_list.append(summary["cv_accuracy"])

    cv_acc: Optional[float] = args.cv
    if cv_acc is None:
        if len(cv_list) == 1:
            cv_acc = cv_list[0]
        elif len(cv_list) > 1:
            cv_acc = sum(cv_list) / len(cv_list)

    if cv_acc is None:
        raise ValueError("CV accuracy is not provided. Pass --cv or provide valid summary files.")

    summary_paths: List[Optional[str]] = []
    for path in (args.summary or []):
        summary_paths.append(path)
    if summary_paths and len(summary_paths) == 1 and len(configs) > 1:
        # Reuse the single provided summary path for all configs if only one given
        summary_paths = summary_paths * len(configs)
    while len(summary_paths) < len(configs):
        summary_paths.append(None)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": "+".join(experiments) if multi else experiments[0],
        "model_name": "+".join(model_names) if multi else model_names[0],
        "img_size": img_sizes if multi else img_sizes[0],
        "aug_notes": " | ".join(aug_notes) if multi else aug_notes[0],
        "epochs": epochs if multi else epochs[0],
        "batch_size": batch_sizes if multi else batch_sizes[0],
        "cv_acc": cv_acc,
        "lb_public": args.lb_public,
        "lb_private": args.lb_private,
        "notes": args.notes,
        "summary_path": summary_paths if multi else summary_paths[0],
        "config_path": args.config if multi else args.config[0],
    }

    if multi:
        entry["per_config"] = [
            {
                "experiment": experiments[i],
                "model_name": model_names[i],
                "img_size": img_sizes[i],
                "aug_notes": aug_notes[i],
                "epochs": epochs[i],
                "batch_size": batch_sizes[i],
                "summary_path": summary_paths[i],
                "config_path": args.config[i],
            }
            for i in range(len(configs))
        ]
        entry["cv_components"] = cv_list if cv_list else None

    return entry


def append_entry(log_path: Path, entry: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main() -> None:
    args = parse_args()
    config_paths = args.config
    configs: List[Dict[str, Any]] = [load_config(cfg_path) for cfg_path in config_paths]

    summaries: List[Optional[Dict[str, Any]]] = []
    provided_summaries = args.summary or []

    for idx, cfg in enumerate(configs):
        summary_obj: Optional[Dict[str, Any]] = None
        summary_path: Optional[str] = None

        if provided_summaries:
            if len(provided_summaries) == 1:
                summary_path = provided_summaries[0]
            elif idx < len(provided_summaries):
                summary_path = provided_summaries[idx]

        if summary_path:
            summary_obj = load_summary(Path(summary_path))
        elif args.cv is None:
            # Fallback: try default summary path logs/{experiment_name}_summary.json
            summary_guess = Path("logs") / f"{cfg['experiment']['name']}_summary.json"
            if summary_guess.exists():
                summary_obj = load_summary(summary_guess)
                if not provided_summaries:
                    provided_summaries.append(str(summary_guess))

        summaries.append(summary_obj)

    entry = build_entry(configs, args, summaries)
    append_entry(Path(args.log_path), entry)
    print(f"Logged experiment entry for {entry['experiment']} to {args.log_path}")


if __name__ == "__main__":
    main()
