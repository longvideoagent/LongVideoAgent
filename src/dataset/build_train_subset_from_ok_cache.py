#!/usr/bin/env python3
"""Build train subset by intersecting train questions with OK-only grounding cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def _normalize_question(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Questions file must be JSON list or JSONL: {path}")
            rows = [x for x in data if isinstance(x, dict)]
            if not rows:
                raise ValueError(f"No dict rows found in JSON list: {path}")
            return rows
        except json.JSONDecodeError:
            pass

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL line {line_no} is not a dict: {path}")
            rows.append(obj)
    if not rows:
        raise ValueError(f"Questions file is empty or invalid JSON/JSONL: {path}")
    return rows


def _load_ok_questions_from_cache(path: Path) -> Set[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    ok_questions: Set[str] = set()

    def _consume_entry(entry: Dict[str, Any]) -> None:
        status = str(entry.get("status") or "").strip().lower()
        if status != "ok":
            return
        q = _normalize_question(entry.get("question") or entry.get("q") or "")
        if q:
            ok_questions.add(q)

    if isinstance(raw, dict):
        for _, v in raw.items():
            if isinstance(v, dict):
                _consume_entry(v)
    elif isinstance(raw, list):
        for v in raw:
            if isinstance(v, dict):
                _consume_entry(v)
    else:
        raise ValueError(f"Unsupported cache JSON format (expect object/list): {path}")

    return ok_questions


def _extract_question(item: Dict[str, Any]) -> str:
    return _normalize_question(item.get("question") or item.get("q") or "")


def _filter_intersection(train_rows: List[Dict[str, Any]], ok_questions: Set[str]) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    for row in train_rows:
        q = _extract_question(row)
        if q and q in ok_questions:
            matched.append(row)
    return matched


def _resolve_output_path(train_path: Path, output_dir: Path, n_final: int) -> Path:
    stem = train_path.stem
    suffix = train_path.suffix or ".json"
    return output_dir / f"{stem}_ok_intersection_{n_final}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter OK-only questions from grounding cache and intersect with train questions"
    )
    parser.add_argument("--cache-json", type=str, required=True, help="Path to grounding cache JSON")
    parser.add_argument("--train-path", type=str, required=True, help="Path to train questions JSON/JSONL")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same directory as --train-path)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional explicit output path. If set, overrides --output-dir naming.",
    )
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=100,
        help="Round final sample size down to this bucket size (default: 100)",
    )
    args = parser.parse_args()

    cache_path = Path(args.cache_json)
    train_path = Path(args.train_path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    if not train_path.is_file():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if args.bucket_size <= 0:
        raise ValueError("--bucket-size must be positive")

    ok_questions = _load_ok_questions_from_cache(cache_path)
    train_rows = _load_json_or_jsonl(train_path)
    matched_rows = _filter_intersection(train_rows, ok_questions)

    n_intersection = len(matched_rows)
    n_final = (n_intersection // args.bucket_size) * args.bucket_size
    final_rows = matched_rows[:n_final]

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_dir = Path(args.output_dir) if args.output_dir else train_path.parent
        output_path = _resolve_output_path(train_path, output_dir, n_final)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_rows, f, ensure_ascii=False, indent=2)

    print(f"cache_path: {cache_path}")
    print(f"train_path: {train_path}")
    print(f"ok_questions_in_cache: {len(ok_questions)}")
    print(f"train_rows: {len(train_rows)}")
    print(f"intersection_rows: {n_intersection}")
    print(f"bucket_size: {args.bucket_size}")
    print(f"final_rows: {n_final}")
    print(f"output_path: {output_path}")


if __name__ == "__main__":
    main()
