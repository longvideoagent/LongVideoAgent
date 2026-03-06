#!/usr/bin/env python3
"""Randomly sample N questions from LongTVQA / LongTVQA+ training set.

Supports input in JSON list or JSONL, and writes subset to JSON/JSONL.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json_or_jsonl(path: Path, input_format: str) -> Tuple[List[Dict[str, Any]], str]:
    """Load rows and return (rows, detected_format)."""
    if input_format not in {"auto", "json", "jsonl"}:
        raise ValueError(f"Unsupported --input-format: {input_format}")

    if input_format in {"auto", "json"}:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"JSON must be a list of question objects: {path}")
            rows = [x for x in data if isinstance(x, dict)]
            if not rows:
                raise ValueError(f"No valid question objects found in JSON: {path}")
            return rows, "json"
        except json.JSONDecodeError:
            if input_format == "json":
                raise

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL line {line_no} is not an object: {path}")
            rows.append(obj)

    if not rows:
        raise ValueError(f"Questions file is empty or invalid JSON/JSONL: {path}")
    return rows, "jsonl"


def _write_subset(rows: List[Dict[str, Any]], output_path: Path, output_format: str, default_format: str) -> str:
    fmt = default_format if output_format == "preserve" else output_format
    if fmt not in {"json", "jsonl"}:
        raise ValueError(f"Unsupported output format: {fmt}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            for item in rows:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return fmt


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomly sample N questions from LongTVQA / LongTVQA+ train set")
    parser.add_argument("--input-path", type=str, required=True, help="Path to train questions file (JSON list or JSONL)")
    parser.add_argument("--output-path", type=str, required=True, help="Path to write sampled subset")
    parser.add_argument("--num-samples", type=int, required=True, help="Number of questions to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--input-format",
        type=str,
        default="auto",
        choices=["auto", "json", "jsonl"],
        help="Input file format (default: auto detect)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="preserve",
        choices=["preserve", "json", "jsonl"],
        help="Output format (default: keep input format)",
    )
    parser.add_argument(
        "--allow-smaller",
        action="store_true",
        help="If N > total rows, output all rows instead of error",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    rows, detected_format = _load_json_or_jsonl(input_path, args.input_format)
    total = len(rows)

    n = args.num_samples
    if n > total:
        if args.allow_smaller:
            n = total
        else:
            raise ValueError(
                f"Requested --num-samples={n} but dataset has only {total} rows. "
                "Use --allow-smaller to output all rows."
            )

    rng = random.Random(args.seed)
    sampled_rows = rng.sample(rows, k=n)

    final_format = _write_subset(sampled_rows, output_path, args.output_format, detected_format)

    print(f"Input: {input_path}")
    print(f"Detected input format: {detected_format}")
    print(f"Total rows: {total}")
    print(f"Sampled rows: {n}")
    print(f"Seed: {args.seed}")
    print(f"Output format: {final_format}")
    print(f"Output path: {output_path}")


if __name__ == "__main__":
    main()
