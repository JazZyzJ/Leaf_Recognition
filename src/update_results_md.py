from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync experiment_results.jsonl into results.md table.")
    parser.add_argument(
        "--log_path",
        default="logs/experiment_results.jsonl",
        help="JSONL file created by log_results.py",
    )
    parser.add_argument("--results_md", default="results.md", help="Markdown table output.")
    parser.add_argument(
        "--keep_all_runs",
        action="store_true",
        help="If set, keep all runs instead of only the latest per experiment.",
    )
    return parser.parse_args()


def load_entries(log_path: Path) -> List[Dict]:
    if not log_path.exists():
        return []
    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def deduplicate_entries(entries: List[Dict], keep_all: bool) -> List[Dict]:
    if keep_all:
        return sorted(entries, key=lambda x: x["timestamp"], reverse=True)

    latest: Dict[tuple, Dict] = {}
    for entry in entries:
        # Treat different notes as different methods; key combines experiment and notes
        notes = entry.get("notes", "") or ""
        key = (entry["experiment"], notes)
        if key not in latest or entry["timestamp"] > latest[key]["timestamp"]:
            latest[key] = entry
    return sorted(latest.values(), key=lambda x: x["timestamp"], reverse=True)


def format_value(value, default="TBD"):
    if value is None or value == "":
        return default
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_table(entries: List[Dict]) -> str:
    header = (
        "| model_name | img_size | aug_notes | epochs | batch_size | CV_acc | LB_public | LB_private | notes |\n"
        "|------------|---------:|-----------|-------:|-----------:|-------:|----------:|-----------:|-------|\n"
    )
    rows = []
    for entry in entries:
        rows.append(
            "| {model} | {img} | {aug} | {epochs} | {batch} | {cv} | {lb_pub} | {lb_priv} | {notes} |".format(
                model=entry["model_name"],
                img=entry["img_size"],
                aug=entry.get("aug_notes", "") or "N/A",
                epochs=entry["epochs"],
                batch=entry["batch_size"],
                cv=format_value(entry.get("cv_acc")),
                lb_pub=format_value(entry.get("lb_public")),
                lb_priv=format_value(entry.get("lb_private")),
                notes=entry.get("notes", "") or "-",
            )
        )
    if not rows:
        rows.append("| - | - | - | - | - | TBD | TBD | TBD | - |")
    return header + "\n".join(rows) + "\n"


def write_results_md(results_md: Path, table: str) -> None:
    with open(results_md, "w", encoding="utf-8") as f:
        f.write(table)


def main() -> None:
    args = parse_args()
    entries = load_entries(Path(args.log_path))
    entries = deduplicate_entries(entries, keep_all=args.keep_all_runs)
    table = build_table(entries)
    write_results_md(Path(args.results_md), table)
    print(f"Wrote {len(entries)} entries to {args.results_md}")


if __name__ == "__main__":
    main()
