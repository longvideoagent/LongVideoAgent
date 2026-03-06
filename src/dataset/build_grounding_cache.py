"""Build grounding cache JSON for offline training preprocessing.

This script runs the initial grounding step (not re-grounding) used in
`evaluate_api_unified.py` and stores question->clip predictions to a cache JSON.
"""

import argparse
import concurrent.futures
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI


DEFAULT_GROUNDING_MODEL = "grok-4-fast-reasoning"
DEFAULT_GROUNDING_BASE_URL = "https://api2.aigcbest.top/v1"


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
        try:
            q["answer_idx"] = parse_answer_to_idx(q["answer"])
        except ValueError:
            pass
    return q


def load_questions(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Questions file must contain a JSON list: {path}")
            return [normalize_question_entry(q) for q in data]
        except json.JSONDecodeError:
            pass

    questions: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(normalize_question_entry(json.loads(line)))
    if not questions:
        raise ValueError(f"Questions file is empty or invalid: {path}")
    return questions


def load_subtitles_file(path: str) -> Dict[str, str]:
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            raise ValueError(f"Subtitles file must contain a JSON object: {path}")
        except json.JSONDecodeError:
            pass

    subtitles: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                if len(item) == 1:
                    key, val = next(iter(item.items()))
                    subtitles[str(key)] = str(val)
                elif "clip" in item and "subtitle" in item:
                    subtitles[str(item["clip"])] = str(item["subtitle"])
    if not subtitles:
        raise ValueError(f"Subtitles file is empty or invalid: {path}")
    return subtitles


def build_subtitles_for_episode(clip_subtitles: Dict[str, str], episode_prefix: str) -> str:
    matching_clips = {k: v for k, v in clip_subtitles.items() if k.startswith(episode_prefix)}
    sorted_clips = sorted(matching_clips.items())
    formatted_subtitles = [f"<{clip_key}>{subtitle_text}</{clip_key}>" for clip_key, subtitle_text in sorted_clips]
    return "\n".join(formatted_subtitles)


def extract_episode_prefix_for_grounding(vid_name: str, dataset: str) -> str:
    if dataset == "tvqa":
        return vid_name
    return vid_name[:6]


def postprocess_response(response: str) -> str:
    type_match = re.search(r"<type>.*?</type>", response, re.DOTALL)
    if type_match:
        return response[: type_match.end()]
    time_match = re.search(r"<time>.*?</time>", response, re.DOTALL)
    if time_match:
        return response[: time_match.end()]
    return response


def extract_clip_content(text: str) -> str:
    match = re.search(r"<clip>(.*?)</clip>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def build_grounding_prompt(question_data: Dict[str, Any], sub_block: str) -> str:
    return f"""
Question: {question_data['q']}
Options:
a0: {question_data.get('a0', '')}
a1: {question_data.get('a1', '')}
a2: {question_data.get('a2', '')}
a3: {question_data.get('a3', '')}
a4: {question_data.get('a4', '')}

Subtitles: {sub_block}

The subtitles are formatted as <clip_label>subtitle_content</clip_label>, where each < > pair contains a clip label followed by its corresponding subtitle content.

Based on the question and subtitles, determine:
1. The specific clip label where the answer to this question occurs or is mentioned (output in <clip>label</clip> format)
Please analyze the given question and provide the following information:
<clip>
clip_label (the specific clip where the question's answer can be found in the video)
</clip>
"""


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "model"


def save_cache(cache: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def run_one(
    idx: int,
    question_data: Dict[str, Any],
    dataset: str,
    clip_subtitles: Dict[str, str],
    client: OpenAI,
    model: str,
    max_tokens: int,
    temperature: float,
) -> Tuple[str, Dict[str, Any]]:
    key = str(idx)
    vid_name = question_data["vid_name"]
    episode_prefix = extract_episode_prefix_for_grounding(vid_name, dataset)
    episode_sub_block = build_subtitles_for_episode(clip_subtitles, episode_prefix)
    prompt_content = build_grounding_prompt(question_data, episode_sub_block)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_content}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw_response = response.choices[0].message.content or ""
        processed_response = postprocess_response(raw_response)
        predicted_clip = extract_clip_content(processed_response)

        entry = {
            "question": question_data["q"],
            "clip": predicted_clip,
            "vid_name": vid_name,
            "status": "ok" if predicted_clip else "empty_clip",
            "raw_response": raw_response,
            "processed_response": processed_response,
        }
        return key, entry
    except Exception as e:
        entry = {
            "question": question_data["q"],
            "clip": "",
            "vid_name": vid_name,
            "status": "error",
            "error": str(e),
        }
        return key, entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline grounding cache JSON (question -> occur clip).")
    parser.add_argument("--dataset", type=str, required=True, choices=["tvqa", "tvqa_plus"], help="Dataset type")
    parser.add_argument("--questions-path", type=str, required=True, help="Training questions JSON/JSONL path")
    parser.add_argument("--subs-path", type=str, required=True, help="Clip subtitles JSON/JSONL path")
    parser.add_argument("--grounding-model", type=str, default=DEFAULT_GROUNDING_MODEL, help="Grounding model name")
    parser.add_argument(
        "--grounding-base-url",
        type=str,
        default=DEFAULT_GROUNDING_BASE_URL,
        help="Grounding API base URL",
    )
    parser.add_argument(
        "--grounding-api-key",
        type=str,
        default=None,
        help="Grounding API key (default: env qdd_api)",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save cache JSON")
    parser.add_argument(
        "--output-filename",
        type=str,
        default=None,
        help="Output cache filename (default: grounding_cache_{dataset}_{model}.json)",
    )
    parser.add_argument("--threads", type=int, default=8, help="Number of concurrent API workers")
    parser.add_argument("--max-samples", type=int, default=0, help="Process only first N samples (0 means all)")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens for grounding call")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--save-every", type=int, default=50, help="Persist cache every N finished samples")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing cache and regenerate all")
    args = parser.parse_args()

    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")

    api_key = args.grounding_api_key or os.getenv("qdd_api")
    if not api_key:
        raise ValueError("Missing grounding API key. Set --grounding-api-key or env qdd_api")

    questions_path = Path(args.questions_path)
    subs_path = Path(args.subs_path)
    if not questions_path.is_file():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    if not subs_path.is_file():
        raise FileNotFoundError(f"Subtitles file not found: {subs_path}")

    output_dir = Path(args.output_dir)
    default_filename = f"grounding_cache_{args.dataset}_{safe_model_name(args.grounding_model)}.json"
    output_filename = args.output_filename or default_filename
    output_path = output_dir / output_filename

    questions = load_questions(str(questions_path))
    if args.max_samples > 0:
        questions = questions[: args.max_samples]
    clip_subtitles = load_subtitles_file(str(subs_path))

    cache: Dict[str, Any] = {}
    if output_path.is_file() and not args.overwrite:
        with open(output_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                cache = loaded
            else:
                raise ValueError(f"Existing cache is not a JSON object: {output_path}")

    done_questions = set()
    for _, entry in cache.items():
        if isinstance(entry, dict) and "question" in entry:
            done_questions.add(str(entry["question"]).strip())

    pending: List[Tuple[int, Dict[str, Any]]] = []
    for idx, q in enumerate(questions, start=1):
        question_text = str(q.get("q", "")).strip()
        if not question_text:
            continue
        if not args.overwrite and question_text in done_questions:
            continue
        pending.append((idx, q))

    print(f"Loaded questions: {len(questions)}")
    print(f"Existing cache entries: {len(done_questions)}")
    print(f"Pending API calls: {len(pending)}")
    print(f"Output path: {output_path}")

    if not pending:
        print("Nothing to do.")
        return

    client = OpenAI(api_key=api_key, base_url=args.grounding_base_url)
    finished = 0
    ok_count = 0
    err_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [
            executor.submit(
                run_one,
                idx,
                q,
                args.dataset,
                clip_subtitles,
                client,
                args.grounding_model,
                args.max_tokens,
                args.temperature,
            )
            for idx, q in pending
        ]
        for future in concurrent.futures.as_completed(futures):
            key, entry = future.result()
            cache[key] = entry
            finished += 1
            if entry.get("status") == "ok":
                ok_count += 1
            else:
                err_count += 1

            if finished % args.save_every == 0:
                save_cache(cache, output_path)
                print(
                    f"[progress] finished={finished}/{len(pending)} "
                    f"ok={ok_count} non_ok={err_count} saved={output_path}"
                )

    save_cache(cache, output_path)
    print(
        f"Done. finished={finished}/{len(pending)} ok={ok_count} non_ok={err_count}. "
        f"Saved cache to: {output_path}"
    )


if __name__ == "__main__":
    main()
