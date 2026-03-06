#!/usr/bin/env python3
"""Prepare normalized LongTVQA/LongTVQA_plus assets for unified evaluators.

Outputs:
1) Normalized LongTVQA questions JSONL (vid_name + answer_idx)
2) Normalized LongTVQA_plus questions JSON (vid_name + answer_idx)
3) clip_bbox_mapping.json for TVQA+ evaluators
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON list: {path}")
            return [row for row in data if isinstance(row, dict)]
        except json.JSONDecodeError:
            pass

    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    if not rows:
        raise ValueError(f"Empty or invalid JSON/JSONL: {path}")
    return rows


def parse_answer_to_idx(answer_value: Any) -> int:
    if isinstance(answer_value, int):
        if 0 <= answer_value <= 4:
            return answer_value
        raise ValueError(f"Invalid integer answer index: {answer_value}")
    if isinstance(answer_value, str):
        s = answer_value.strip().lower()
        m = re.fullmatch(r"a([0-4])", s)
        if m:
            return int(m.group(1))
        if s.isdigit():
            idx = int(s)
            if 0 <= idx <= 4:
                return idx
    raise ValueError(f"Unsupported answer format: {answer_value}")


def normalize_question_entry(raw_q: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(raw_q)
    if "vid_name" not in q:
        q["vid_name"] = q.get("occur_clip", "")
    if not q["vid_name"]:
        raise ValueError("Missing clip id: expected `vid_name` or `occur_clip`.")
    if "answer_idx" not in q and "answer" in q:
        q["answer_idx"] = parse_answer_to_idx(q["answer"])
    return q


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _box_from_dict(d: Dict[str, Any]) -> List[Any] | None:
    x = _to_float(d.get("x", d.get("left")))
    y = _to_float(d.get("y", d.get("top")))
    w = _to_float(d.get("width", d.get("w")))
    h = _to_float(d.get("height", d.get("h")))
    if None in (x, y, w, h):
        return None
    label = d.get("label", d.get("name", d.get("category", "object")))
    return [x, y, w, h, str(label)]


def _box_from_sequence(seq: Iterable[Any]) -> List[Any] | None:
    vals = list(seq)
    if len(vals) < 4:
        return None
    x = _to_float(vals[0])
    y = _to_float(vals[1])
    w = _to_float(vals[2])
    h = _to_float(vals[3])
    if None in (x, y, w, h):
        return None
    label = str(vals[4]) if len(vals) >= 5 else "object"
    return [x, y, w, h, label]


def normalize_boxes(raw: Any) -> List[List[Any]]:
    if raw is None:
        return []
    out: List[List[Any]] = []
    if isinstance(raw, dict):
        box = _box_from_dict(raw)
        return [box] if box is not None else []
    if isinstance(raw, (list, tuple)):
        if raw and isinstance(raw[0], (int, float, str)):
            box = _box_from_sequence(raw)
            return [box] if box is not None else []
        for item in raw:
            if isinstance(item, dict):
                box = _box_from_dict(item)
                if box is not None:
                    out.append(box)
            elif isinstance(item, (list, tuple)):
                box = _box_from_sequence(item)
                if box is not None:
                    out.append(box)
    return out


def normalize_bbox_by_frame(raw_bbox: Any) -> Dict[str, List[List[Any]]]:
    result: Dict[str, List[List[Any]]] = {}
    if isinstance(raw_bbox, dict):
        for frame_id, frame_boxes in raw_bbox.items():
            boxes = normalize_boxes(frame_boxes)
            if boxes:
                result[str(frame_id)] = boxes
        return result

    if isinstance(raw_bbox, list):
        for item in raw_bbox:
            if not isinstance(item, dict):
                continue
            frame_id = item.get("frame_id", item.get("frame", item.get("frame_idx")))
            if frame_id is None:
                continue
            boxes_raw = item.get("bbox", item.get("boxes", item.get("objects", item)))
            boxes = normalize_boxes(boxes_raw)
            if boxes:
                result.setdefault(str(frame_id), []).extend(boxes)
        return result
    return result


def dedupe_boxes(boxes: List[List[Any]]) -> List[List[Any]]:
    seen: set[Tuple[Any, Any, Any, Any, Any]] = set()
    out: List[List[Any]] = []
    for box in boxes:
        key = (box[0], box[1], box[2], box[3], box[4])
        if key in seen:
            continue
        seen.add(key)
        out.append(box)
    return out


def build_clip_bbox_mapping(questions: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[List[Any]]]]:
    clip_map: Dict[str, Dict[str, List[List[Any]]]] = {}
    for q in questions:
        clip = q.get("vid_name") or q.get("occur_clip")
        if not clip:
            continue
        raw_bbox = q.get("bbox")
        if raw_bbox is None:
            continue
        clip_frames = normalize_bbox_by_frame(raw_bbox)
        if not clip_frames:
            continue
        dst = clip_map.setdefault(str(clip), {})
        for frame_id, boxes in clip_frames.items():
            dst.setdefault(frame_id, []).extend(boxes)

    for clip, frames in clip_map.items():
        for frame_id, boxes in list(frames.items()):
            frames[frame_id] = dedupe_boxes(boxes)
        sorted_items = sorted(frames.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])
        clip_map[clip] = dict(sorted_items)
    return clip_map


def detect_first_question_file(dir_path: Path, preferred_patterns: List[str]) -> Path | None:
    for pattern in preferred_patterns:
        matches = sorted(dir_path.glob(pattern))
        if matches:
            return matches[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare normalized LongTVQA assets for unified evaluators")
    parser.add_argument("--longtvqa-questions", type=str, default=None, help="LongTVQA question file path")
    parser.add_argument("--longtvqa-plus-questions", type=str, default=None, help="LongTVQA_plus question file path")
    parser.add_argument("--longtvqa-dir", type=str, default=None, help="LongTVQA dataset directory (for auto detect)")
    parser.add_argument("--longtvqa-plus-dir", type=str, default=None, help="LongTVQA_plus dataset directory (auto detect)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for prepared files")
    parser.add_argument(
        "--longtvqa-out-name",
        type=str,
        default="LongTVQA_val_normalized.jsonl",
        help="Output filename for normalized LongTVQA questions",
    )
    parser.add_argument(
        "--longtvqa-plus-out-name",
        type=str,
        default="LongTVQA_plus_val_normalized.json",
        help="Output filename for normalized LongTVQA_plus questions",
    )
    parser.add_argument(
        "--bbox-out-name",
        type=str,
        default="clip_bbox_mapping.json",
        help="Output filename for TVQA+ clip bbox mapping",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tvqa_q_path = Path(args.longtvqa_questions) if args.longtvqa_questions else None
    if tvqa_q_path is None and args.longtvqa_dir:
        tvqa_q_path = detect_first_question_file(
            Path(args.longtvqa_dir),
            ["*val*.jsonl", "*val*.json", "*LongTVQA*.jsonl", "*LongTVQA*.json"],
        )
    if tvqa_q_path is None:
        raise ValueError("Cannot locate LongTVQA question file. Provide --longtvqa-questions or --longtvqa-dir.")

    tvqa_plus_q_path = Path(args.longtvqa_plus_questions) if args.longtvqa_plus_questions else None
    if tvqa_plus_q_path is None and args.longtvqa_plus_dir:
        tvqa_plus_q_path = detect_first_question_file(
            Path(args.longtvqa_plus_dir),
            ["*plus*val*.json", "*plus*val*.jsonl", "*LongTVQA_plus*.json", "*LongTVQA_plus*.jsonl"],
        )
    if tvqa_plus_q_path is None:
        raise ValueError(
            "Cannot locate LongTVQA_plus question file. Provide --longtvqa-plus-questions or --longtvqa-plus-dir."
        )

    tvqa_rows = [normalize_question_entry(x) for x in load_json_or_jsonl(str(tvqa_q_path))]
    tvqa_plus_rows = [normalize_question_entry(x) for x in load_json_or_jsonl(str(tvqa_plus_q_path))]
    bbox_map = build_clip_bbox_mapping(tvqa_plus_rows)

    tvqa_out = output_dir / args.longtvqa_out_name
    tvqa_plus_out = output_dir / args.longtvqa_plus_out_name
    bbox_out = output_dir / args.bbox_out_name

    save_jsonl(tvqa_out, tvqa_rows)
    save_json(tvqa_plus_out, tvqa_plus_rows)
    save_json(bbox_out, bbox_map)

    frame_count = sum(len(frames) for frames in bbox_map.values())
    box_count = sum(len(boxes) for frames in bbox_map.values() for boxes in frames.values())
    print(f"LongTVQA questions: {len(tvqa_rows)}")
    print(f"LongTVQA+ questions: {len(tvqa_plus_rows)}")
    print(f"BBox clips: {len(bbox_map)}")
    print(f"BBox frames: {frame_count}")
    print(f"BBox boxes: {box_count}")
    print(f"Saved: {tvqa_out}")
    print(f"Saved: {tvqa_plus_out}")
    print(f"Saved: {bbox_out}")


if __name__ == "__main__":
    main()
