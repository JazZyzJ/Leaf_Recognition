from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log experiment results for later reporting.")
    parser.add_argument("--config", required=True, help="Config used for the run.")
    parser.add_argument(
        "--summary",
        required=False,
        help="Path to the summary JSON generated after training.",
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


def build_entry(config: Dict[str, Any], args: argparse.Namespace, summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    training_cfg = config["training"]
    experiment_cfg = config["experiment"]
    model_cfg = config["model"]

    cv_acc = args.cv
    if cv_acc is None and summary:
        cv_acc = summary.get("cv_accuracy")

    if cv_acc is None:
        raise ValueError("CV accuracy is not provided. Pass --cv or provide a valid summary file.")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment_cfg["name"],
        "model_name": model_cfg["model_name"],
        "img_size": training_cfg["img_size"],
        "aug_notes": training_cfg.get("aug_desc", ""),
        "epochs": training_cfg["num_epochs"],
        "batch_size": training_cfg["batch_size"],
        "cv_acc": cv_acc,
        "lb_public": args.lb_public,
        "lb_private": args.lb_private,
        "notes": args.notes,
        "summary_path": str(args.summary) if args.summary else None,
        "config_path": args.config,
    }
    return entry


def append_entry(log_path: Path, entry: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = None
    if args.summary:
        summary = load_summary(Path(args.summary))
    elif args.cv is None:
        summary_guess = Path("logs") / f"{config['experiment']['name']}_summary.json"
        if summary_guess.exists():
            summary = load_summary(summary_guess)
            args.summary = str(summary_guess)

    entry = build_entry(config, args, summary)
    append_entry(Path(args.log_path), entry)
    print(f"Logged experiment entry for {entry['experiment']} to {args.log_path}")


if __name__ == "__main__":
    main()
