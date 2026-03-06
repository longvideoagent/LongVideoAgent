"""Unified API evaluator for TVQA and TVQA+.

This script keeps one evaluation pipeline and switches dataset-specific
input handling via `--dataset`.
"""

import argparse
import base64
import concurrent.futures
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from PIL import Image

grounding_client = None
vision_client = None
main_client = None


@dataclass
class EvalConfig:
    dataset: str
    questions_path: str
    subs_path: str
    base_frame_dir: str
    bbox_json_path: str | None
    output_filename: str
    detailed_output_filename: str
    grounding_model: str
    vision_model: str
    main_model: str
    grounding_base_url: str
    vision_base_url: str
    main_base_url: str
    grounding_api_key: str | None
    vision_api_key: str | None
    main_api_key: str | None
    num_threads: int
    grounding_cache_json_path: str
    verbose: bool
    debug: bool


config: EvalConfig | None = None


DATASET_DEFAULTS: Dict[str, Dict[str, str | None]] = {
    "tvqa": {
        "questions_path": "../Tvqa/house_met/tvqa_val_house_met.jsonl",
        "subs_path": "../Tvqa/tvqa_subtitles.json",
        "base_frame_dir": "../Tvqa/house_met_frames",
        "bbox_json_path": None,
        "output_filename": "./eval_tvqa_grok-4-fast-reasoning.json",
        "detailed_output_filename": "./data_log_eval_tvqa_grok-4-fast-reasoning.json",
    },
    "tvqa_plus": {
        "questions_path": "../Tvqa_data/tvqa_plus_val.json",
        "subs_path": "../Tvqa_data/all_episodes_subtitles_by_clips.json",
        "base_frame_dir": "../Tvqa_data/bbt_frames",
        "bbox_json_path": "../Tvqa_data/clip_bbox_mapping.json",
        "output_filename": "./eval_tvqa_plus_grok-4-fast-reasoning.json",
        "detailed_output_filename": "./data_log_eval_tvqa_plus_grok-4-fast-reasoning.json",
    },
}


DEFAULT_GROUNDING_MODEL = "grok-4-fast-reasoning"
DEFAULT_VISION_MODEL = "gpt-4o"
DEFAULT_MAIN_MODEL = "grok-4-fast-reasoning"
DEFAULT_GROUNDING_BASE_URL = "https://api2.aigcbest.top/v1"
DEFAULT_VISION_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MAIN_BASE_URL = "https://api2.aigcbest.top/v1"
DEFAULT_GROUNDING_CACHE_JSON = "/home/rliuay/runtao/proj_videoqa/Tvqa_data/grounding_pairs_6000_samples.json"


def log_info(msg: str) -> None:
    if config is None or config.verbose:
        print(msg)


def log_debug(msg: str) -> None:
    if config is not None and config.debug:
        print(msg)


def log_warn(msg: str) -> None:
    print(msg)


def make_progress_bar(total: int, desc: str):
    if config is not None and not config.verbose:
        return None
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc=desc, unit="q", dynamic_ncols=True, leave=True)
    except Exception:
        class _SimpleProgress:
            def __init__(self, total_count: int, title: str):
                self.total = max(total_count, 1)
                self.n = 0
                self.title = title

            def update(self, n: int = 1):
                self.n += n
                pct = min(100.0, self.n * 100.0 / self.total)
                print(f"\r{self.title}: {self.n}/{self.total} ({pct:.1f}%)", end="", flush=True)

            def close(self):
                print()

        return _SimpleProgress(total, desc)


def initialize_clients() -> None:
    global grounding_client, vision_client, main_client, config
    if config is None:
        raise RuntimeError("Config not initialized.")

    grounding_client = OpenAI(
        api_key=config.grounding_api_key,
        base_url=config.grounding_base_url,
    )
    vision_client = OpenAI(
        api_key=config.vision_api_key,
        base_url=config.vision_base_url,
    )
    main_client = OpenAI(
        api_key=config.main_api_key,
        base_url=config.main_base_url,
    )


def _extract_episode_prefix_for_subtitles(vid_name: str) -> str:
    if config is None:
        raise RuntimeError("Config not initialized.")
    if config.dataset == "tvqa":
        if not vid_name:
            return ""
        parts = vid_name.split("_")
        if len(parts) >= 2:
            return "_".join(parts[:2])
        return parts[0]
    return vid_name.split("_")[0] if vid_name else ""


def _extract_episode_prefix_for_grounding(vid_name: str) -> str:
    if config is None:
        raise RuntimeError("Config not initialized.")
    if config.dataset == "tvqa":
        return vid_name
    return vid_name[:6]


def _initial_answer_format_hint() -> str:
    if config is None:
        raise RuntimeError("Config not initialized.")
    if config.dataset == "tvqa_plus":
        return (
            "The answer must be concise and direct, in the format <answer>ax</answer>, "
            "where 'x' is the index of the selected option (e.g., <answer>a1</answer> for option a1)."
        )
    return "The answer must be concise and direct."


def _load_questions(path: str) -> List[Dict[str, Any]]:
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


def bbox_to_string_simplified(file_path: str | None = None, key: str | None = None) -> str:
    if config is None:
        raise RuntimeError("Config not initialized.")
    if config.dataset != "tvqa_plus":
        return ""

    file_path = file_path or config.bbox_json_path
    if not file_path:
        return ""

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        if key not in json_data:
            return f"Error: Key '{key}' not found in JSON data."

        bbox_data = json_data[key]
        result: List[str] = []
        sorted_frame_ids = sorted(bbox_data.keys(), key=lambda x: int(x) if x.isdigit() else x)
        for frame_id in sorted_frame_ids:
            result.append(f"Frame {frame_id}:")
            for bbox in bbox_data[frame_id]:
                x, y, width, height, name = bbox
                result.append(f"  - {name}: ({x}, {y}, {width}, {height})")
        return "\n".join(result)
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except json.JSONDecodeError:
        return f"Error: Invalid JSON format in file '{file_path}'."
    except Exception as e:
        return f"Error processing JSON data: {str(e)}"


def convert_image_to_base64_data_url(path: str) -> str | None:
    try:
        with Image.open(path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def process_and_query_seg(
    seg: dict,
    vid: str,
    text_client: OpenAI,
    vision_client_instance: OpenAI,
    base_frame_dir: str | None = None,
    model: str | None = None,
) -> str:
    if config is None:
        raise RuntimeError("Config not initialized.")
    base_frame_dir = base_frame_dir or config.base_frame_dir
    model = model or config.vision_model

    messages_content = []
    frame_nums = list(range(1, 181, 12))
    log_debug(f"  Selected {len(frame_nums)} frames for vision query: {frame_nums}")
    for fn in frame_nums:
        img_path = Path(base_frame_dir, vid, f"{fn:05d}.jpg")
        if img_path.is_file():
            url = convert_image_to_base64_data_url(str(img_path))
            if url:
                messages_content.append({"type": "image_url", "image_url": {"url": url}})

    if config.dataset == "tvqa_plus":
        bbox_info = bbox_to_string_simplified(key=vid)
        prompt = (
            f"Images 1-{len(frame_nums)} are video frames extracted from frames 1 to 180. "
            f"Bounding box information is provided in {bbox_info}. "
            "You can focus on the key objects and actions within these bounding boxes in the frames.\n"
            "And here is a description of what I want to know:\n"
            f"{seg['description']}"
        )
    else:
        prompt = (
            f"Images 1-{len(frame_nums)} are video frames extracted from frames 1 to 180.\n"
            "And here is a description of what I want to know:\n"
            f"{seg['description']}"
        )

    messages_content.append({"type": "text", "text": prompt})
    resp = vision_client_instance.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": messages_content}],
    )
    return resp.choices[0].message.content


def parse_action_from_response(response: str) -> Tuple[str, str]:
    search_match = re.search(r"<search>(.*?)</search>", response, re.DOTALL)
    if search_match:
        return "search", search_match.group(1).strip()

    grounding_match = re.search(r"<request_grounding>(.*?)</request_grounding>", response, re.DOTALL)
    if grounding_match:
        return "request_grounding", grounding_match.group(1).strip()

    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer_match:
        return "answer", answer_match.group(1).strip()

    return "invalid", ""


def execute_action(
    action_type: str,
    content: str,
    vid: str,
    text_client: OpenAI,
    vision_client_instance: OpenAI,
    question_data: Dict,
    episode_sub_block: str,
    clip_subtitles: Dict[str, str],
) -> Tuple[str, bool]:
    if action_type == "answer":
        return f"\n<answer>{content}</answer>", True

    if action_type == "search":
        try:
            seg = {"description": content}
            information_parts: List[str] = []
            try:
                vision_response = process_and_query_seg(seg, vid, text_client, vision_client_instance)
                information_parts.append(f"Visual Description:\n{vision_response.strip()}")
            except Exception as e:
                log_warn(f"Vision LLM call failed: {e}")
                information_parts.append(f"Visual Description: Error - {str(e)}")

            combined_info = "\n".join(information_parts)
            if config and config.dataset == "tvqa_plus":
                bbox_info = bbox_to_string_simplified(key=vid)
                return f"\n<information>Bounding Box:{bbox_info}\n{combined_info}</information>\n", False
            return f"\n<information>\n{combined_info}</information>\n", False
        except Exception as e:
            log_warn(f"Search action failed: {e}")
            return f"\n<information>Error: Failed to get information - {str(e)}</information>\n", False

    if action_type == "request_grounding":
        try:
            grounding_result = re_analyze_single_question_api(question_data, episode_sub_block, vid, attempt_round=1)
            if "error" in grounding_result:
                log_warn(f"Grounding failed: {grounding_result['error']}")
                result_content = f"Grounding failed for query: {content}. Error: {grounding_result['error']}"
                return f"\n<grounding_info>{result_content}</grounding_info>\n", False
            predicted_clip = grounding_result.get("predicted_clip", vid)
            new_sub = get_clip_subtitle(clip_subtitles, predicted_clip)
            result_content = f"<New_clip>{predicted_clip} + {new_sub}</New_clip>"
            return f"\n{result_content}\n", False
        except Exception as e:
            log_warn(f"Grounding action failed: {e}")
            return f"\n<grounding_info>Error: Failed to perform grounding - {str(e)}</grounding_info>\n", False

    return "\nMy action is not correct. I need to search, request grounding, or answer.\n", False


def get_clip_subtitle(clip_subtitles: Dict[str, str], clip_name: str) -> str:
    subtitle_text = clip_subtitles.get(clip_name, "")
    if subtitle_text:
        return f"<{clip_name}>{subtitle_text}</{clip_name}>"
    log_warn(f"Warning: No subtitle found for clip {clip_name}")
    return ""


def build_subtitles_for_episode(clip_subtitles: Dict[str, str], episode_prefix: str) -> str:
    matching_clips = {k: v for k, v in clip_subtitles.items() if k.startswith(episode_prefix)}
    sorted_clips = sorted(matching_clips.items())
    formatted_subtitles = [f"<{clip_key}>{subtitle_text}</{clip_key}>" for clip_key, subtitle_text in sorted_clips]
    return "\n".join(formatted_subtitles)


def get_subtitles_for_video(clip_subtitles: Dict[str, str], vid_name: str) -> str:
    episode_prefix = _extract_episode_prefix_for_subtitles(vid_name)
    return build_subtitles_for_episode(clip_subtitles, episode_prefix)


def grounding_llm_generate(user_content: str, model: str | None = None) -> str:
    if config is None:
        raise RuntimeError("Config not initialized.")
    model = model or config.grounding_model
    try:
        response = grounding_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.6,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        log_warn(f"Grounding API call failed: {e}")
        return "Error: Failed to generate response"


def main_llm_generate(conversation_history: str, model: str | None = None) -> str:
    if config is None:
        raise RuntimeError("Config not initialized.")
    model = model or config.main_model
    try:
        response = main_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": conversation_history}],
            temperature=0.6,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        log_warn(f"Main pipeline API call failed: {e}")
        return "Error: Failed to generate response"


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


def re_analyze_single_question_api(question_data: Dict, sub_block: str, vid: str, attempt_round: int) -> Dict:
    original_idx = question_data.get("original_idx", "?")
    log_debug(f"      Question {original_idx}: API Analysis round {attempt_round}")

    prompt_content = f"""
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
{vid} may not contain the scene or context related to the question. Please determine a different specific clip label.
Please analyze the given question and provide the following information:
<clip>
clip_label (the specific clip where the question's answer can be found in the video)
</clip>
"""
    try:
        raw_response = grounding_llm_generate(prompt_content)
        log_debug(f"        Raw response: {raw_response}")
        if raw_response.startswith("Error:"):
            return {"error": raw_response}
        processed_response = postprocess_response(raw_response)
        predicted_clip = extract_clip_content(processed_response)
        log_debug(f"        Predicted clip: {predicted_clip}")
        return {
            "predicted_clip": predicted_clip,
            "raw_response": raw_response,
            "processed_response": processed_response,
        }
    except Exception as e:
        log_warn(f"        Analysis error: {e}")
        return {"error": str(e)}


def analyze_single_question_api(
    question_data: Dict,
    sub_block: str,
    attempt_round: int,
    json_path: str | None = None,
) -> Dict:
    if config is None:
        raise RuntimeError("Config not initialized.")
    json_path = json_path or config.grounding_cache_json_path

    original_idx = question_data.get("original_idx", "?")
    log_debug(f"      Question {original_idx}: API Analysis round {attempt_round}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            grounding_pairs = json.load(f)
        target_question = question_data["q"].strip()
        for _, entry in grounding_pairs.items():
            if "question" in entry and entry["question"].strip() == target_question:
                predicted_clip = entry.get("clip", "")
                log_debug(f"        Found match in JSON: Predicted clip: {predicted_clip}")
                return {
                    "predicted_clip": predicted_clip,
                    "raw_response": "From JSON cache",
                    "processed_response": "From JSON cache",
                }
        log_debug("        No match found in JSON, falling back to API...")
    except Exception as e:
        log_debug(f"        JSON loading error: {e}, falling back to API...")

    prompt_content = f"""
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
    try:
        raw_response = grounding_llm_generate(prompt_content)
        log_debug(f"        Raw response: {raw_response}")
        if raw_response.startswith("Error:"):
            return {"error": raw_response}
        processed_response = postprocess_response(raw_response)
        predicted_clip = extract_clip_content(processed_response)
        log_debug(f"        Predicted clip: {predicted_clip}")
        return {
            "predicted_clip": predicted_clip,
            "raw_response": raw_response,
            "processed_response": processed_response,
        }
    except Exception as e:
        log_warn(f"        Analysis error: {e}")
        return {"error": str(e)}


def process_single_question(
    prompt: str,
    vid: str,
    question: str,
    question_data: Dict,
    episode_sub_block: str,
    clip_subtitles: Dict[str, str],
    max_turn: int = 5,
) -> Dict[str, Any]:
    record = {"vid": vid, "question": question, "turns": [], "final_answer": "", "prompt": prompt}
    conversation_history = prompt
    final_answer = ""

    for turn in range(max_turn):
        log_debug(f"\n{'=' * 60}")
        log_debug(f"  Turn {turn + 1}/{max_turn}")
        log_debug(f"{'=' * 60}")

        if turn == max_turn - 1:
            conversation_history += (
                "\nThis is the final turn. Please directly perform Action C and "
                "provide the final answer in <answer>...</answer> format.\n"
            )

        raw_response = main_llm_generate(conversation_history)
        response = postprocess_response(raw_response) if "postprocess_response" in globals() else raw_response
        log_debug(f"LLM Response:\n{response}")

        action_type, content = parse_action_from_response(response)
        preview = f"{content[:100]}...{'...' if len(content) > 100 else ''}"
        log_debug(f"Parsed action - Type: {action_type}, Content: {preview}")

        turn_record = {
            "turn": turn + 1,
            "response": response,
            "action_type": action_type,
            "content": content,
            "is_done": False,
        }

        log_debug(f"Executing action: {action_type}")
        result_content, is_done = execute_action(
            action_type,
            content,
            vid,
            text_client=grounding_client,
            vision_client_instance=vision_client,
            question_data=question_data,
            episode_sub_block=episode_sub_block,
            clip_subtitles=clip_subtitles,
        )

        turn_record["result_content"] = result_content
        turn_record["is_done"] = is_done
        log_debug(f"Action result:\n{result_content}")

        if is_done and action_type == "answer":
            final_answer = content
            log_debug(f"Found answer in turn {turn + 1}: {final_answer}")
            record["turns"].append(turn_record)
            break

        conversation_history += result_content
        log_debug("Updated conversation history with result content")

        if action_type == "request_grounding" and "<New_clip>" in result_content:
            match = re.search(r"<New_clip>(.*?) \+", result_content, re.DOTALL)
            if match:
                vid = match.group(1).strip()
                log_debug(f"Updated vid to: {vid}")

        record["turns"].append(turn_record)
        if turn == max_turn - 1:
            log_debug(f"  Reached maximum turns ({max_turn})")

    record["final_answer"] = final_answer
    record["conversation_history"] = conversation_history
    return record


def process_question_wrapper(q: Dict, total: int, clip_subtitles: Dict[str, str], max_turn: int) -> Dict[str, Any]:
    try:
        q["original_idx"] = total
        log_info(f"Processing question {total}")

        episode_prefix = _extract_episode_prefix_for_grounding(q["vid_name"])
        episode_sub_block = build_subtitles_for_episode(clip_subtitles, episode_prefix)
        grounding_result = analyze_single_question_api(q, episode_sub_block, attempt_round=1)
        predicted_clip = grounding_result.get("predicted_clip", q["vid_name"])
        log_debug(f"Predicted clip for question {total}: {predicted_clip}")

        sub_block = get_clip_subtitle(clip_subtitles, predicted_clip)
        initial_prompt = f"""You must follow these rules in every turn:
Reasoning First: Start by conducting your reasoning inside <reasoning>...</reasoning>. This is where you analyze the current information and decide your next step.
Choose One Action: After reasoning, you must choose exactly one of the following three actions:

Action A: If the current information is insufficient or you are somewhat uncertain, and you need to search for visual information on the current located <clipX>, then search for visual information. If your reasoning indicates that you lack necessary visual knowledge, you can call a visual engine. To do this, use the following format: <search>query</search>

Action B: If the current information is insufficient or you are somewhat uncertain, and you cannot obtain the final answer from the previous location and its possible visual information, then you need to call the grounding agent again for relocation, and output in the <request_grounding> format.

Action C: Provide the Final Answer
If your reasoning indicates that you have enough information to answer, provide the final answer inside <answer>...</answer>.
{_initial_answer_format_hint()}

question: {q['q']}
a0: {q['a0']}
a1: {q['a1']}
a2: {q['a2']}
a3: {q['a3']}
a4: {q['a4']}
<information>subtitles: {sub_block}</information>
"""

        record = process_single_question(
            prompt=initial_prompt,
            vid=predicted_clip,
            question=q["q"],
            question_data=q,
            episode_sub_block=episode_sub_block,
            clip_subtitles=clip_subtitles,
            max_turn=max_turn,
        )
        record["predicted_clip"] = predicted_clip
        record["gt_answer_idx"] = q.get("answer_idx")
        log_info(f"  Result: {len(record['turns'])} turns, Answer: {record['final_answer']}")
        return record
    except Exception as e:
        log_warn(f"Error processing question {total}: {e}")
        return {}


def run_enhanced_pipeline(checkpoint_step: str, max_turn: int = 5, gpu_memory_utilization: float = 0.4, num_threads: int = 10) -> None:
    if config is None:
        raise RuntimeError("Config not initialized.")

    clip_subtitles = load_subtitles_file(config.subs_path)
    questions = _load_questions(config.questions_path)

    results: List[Dict[str, Any]] = []
    total = len(questions)
    consecutive_errors = 0
    max_consecutive_errors = 5

    pbar = make_progress_bar(total, "Evaluating")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(process_question_wrapper, q, idx, clip_subtitles, max_turn)
            for idx, q in enumerate(questions, start=1)
        ]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            if record:
                results.append(record)
                consecutive_errors = 0
            else:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    log_warn(f"Too many consecutive errors ({consecutive_errors}), stopping...")
                    break
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()

    log_info(f"\nTotal processed: {total}")

    simplified_results = [
        {
            "vid": result["vid"],
            "question": result["question"],
            "num_turns": len(result["turns"]),
            "final_answer": result["final_answer"],
            "predicted_clip": result.get("predicted_clip", ""),
        }
        for result in results
    ]

    detailed_results = []
    for result in results:
        last_response = result["turns"][-1]["response"] if result["turns"] else ""
        detailed_results.append(
            {
                "vid": result["vid"],
                "question": result["question"],
                "conversation_history": result["conversation_history"],
                "last_llm_response": last_response,
            }
        )

    total_vision_calls = sum(sum(1 for t in result["turns"] if t["action_type"] == "search") for result in results)
    total_grounding_calls = sum(
        sum(1 for t in result["turns"] if t["action_type"] == "request_grounding") for result in results
    )

    correct_count = 0
    for result in results:
        gt_answer_idx = result.get("gt_answer_idx")
        if gt_answer_idx is None:
            continue
        gt_answer = f"a{gt_answer_idx}"
        pred_answer = result["final_answer"].strip().lower()
        if pred_answer == gt_answer.lower():
            correct_count += 1
    accuracy = correct_count / len(questions) if questions else 0.0

    simplified_output = {
        "dataset": config.dataset,
        "checkpoint_step": checkpoint_step,
        "model_path": f"../qwen2.5-7b-grpo_step-{checkpoint_step}",
        "gpu_memory_utilization": gpu_memory_utilization,
        "total": total,
        "max_turn": max_turn,
        "metadata": {
            "avg_turns": sum(r["num_turns"] for r in simplified_results) / len(simplified_results)
            if simplified_results
            else 0,
            "vision_calls_total": total_vision_calls,
            "grounding_calls_total": total_grounding_calls,
            "completed_questions": len([r for r in simplified_results if r["final_answer"]]),
            "completion_rate": len([r for r in simplified_results if r["final_answer"]]) / len(simplified_results)
            if simplified_results
            else 0,
            "accuracy": accuracy,
        },
        "results": simplified_results,
    }

    with open(config.output_filename, "w", encoding="utf-8") as f:
        json.dump(simplified_output, f, ensure_ascii=False, indent=2)
    log_info(f"Summary results saved to {config.output_filename}")

    with open(config.detailed_output_filename, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)

    if simplified_results:
        metadata = simplified_output["metadata"]
        log_info(f"\nStatistics for {config.dataset} checkpoint-{checkpoint_step}:")
        log_info(f"Model path: ../qwen2.5-7b-grpo_step-{checkpoint_step}")
        log_info(f"GPU memory utilization: {gpu_memory_utilization}")
        log_info(f"Average turns per question: {metadata['avg_turns']:.2f}")
        log_info(f"Total vision calls: {metadata['vision_calls_total']}")
        log_info(f"Total grounding calls: {metadata['grounding_calls_total']}")
        log_info(f"Vision calls per question: {metadata['vision_calls_total'] / len(simplified_results):.2f}")
        log_info(f"Completed questions: {metadata['completed_questions']}/{len(simplified_results)}")
        log_info(f"Completion rate: {metadata['completion_rate']:.2%}")
        log_info(f"Accuracy: {metadata['accuracy']:.2%}")

        turn_counts: Dict[int, int] = {}
        for r in simplified_results:
            turns = r["num_turns"]
            turn_counts[turns] = turn_counts.get(turns, 0) + 1
        log_info("\nTurn distribution:")
        for turns in sorted(turn_counts.keys()):
            count = turn_counts[turns]
            log_info(f"  {turns} turns: {count} questions ({count / len(simplified_results) * 100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="统一版 API 评测脚本 (TVQA / TVQA+)")
    parser.add_argument("--dataset", type=str, required=True, choices=["tvqa", "tvqa_plus"], help="数据集类型")
    parser.add_argument("--checkpoint_step", "-c", type=str, default="api", help="主模型 checkpoint 步数（用于记录）")
    parser.add_argument("--max_turn", "-t", type=int, default=5, help="最大对话轮数")
    parser.add_argument("--gpu_memory_utilization", "-g", type=float, default=0.1, help="GPU内存利用率 (仅用于记录)")
    parser.add_argument("--threads", type=int, default=30, help="并发线程数")
    parser.add_argument("--questions-path", type=str, default=None, help="问题文件路径")
    parser.add_argument("--subs-path", type=str, default=None, help="字幕文件路径")
    parser.add_argument("--base-frame-dir", type=str, default=None, help="视频帧根目录")
    parser.add_argument("--bbox-json-path", type=str, default=None, help="bbox JSON 路径（tvqa_plus 使用）")
    parser.add_argument("--output-filename", type=str, default=None, help="汇总输出文件")
    parser.add_argument("--detailed-output-filename", type=str, default=None, help="详细日志输出文件")
    parser.add_argument("--grounding-model", type=str, default=DEFAULT_GROUNDING_MODEL, help="grounding API 模型名")
    parser.add_argument("--vision-model", type=str, default=DEFAULT_VISION_MODEL, help="vision API 模型名")
    parser.add_argument("--main-model", type=str, default=DEFAULT_MAIN_MODEL, help="主推理 API 模型名")
    parser.add_argument("--grounding-base-url", type=str, default=DEFAULT_GROUNDING_BASE_URL, help="grounding API base URL")
    parser.add_argument("--vision-base-url", type=str, default=DEFAULT_VISION_BASE_URL, help="vision API base URL")
    parser.add_argument("--main-base-url", type=str, default=DEFAULT_MAIN_BASE_URL, help="主推理 API base URL")
    parser.add_argument("--grounding-api-key", type=str, default=None, help="grounding API key（默认读取 qdd_api）")
    parser.add_argument("--vision-api-key", type=str, default=None, help="vision API key（默认读取 aliyun_api）")
    parser.add_argument("--main-api-key", type=str, default=None, help="主推理 API key（默认读取 qdd_api）")
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=True, help="显示进度与关键日志（默认开启）")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false", help="关闭常规日志输出")
    parser.add_argument("--debug", action="store_true", default=False, help="输出详细调试日志")
    parser.add_argument(
        "--grounding-cache-json-path",
        type=str,
        default=DEFAULT_GROUNDING_CACHE_JSON,
        help="grounding question->clip 缓存 JSON 路径",
    )
    args = parser.parse_args()

    ds = DATASET_DEFAULTS[args.dataset]
    questions_path = args.questions_path or str(ds["questions_path"])
    subs_path = args.subs_path or str(ds["subs_path"])
    base_frame_dir = args.base_frame_dir or str(ds["base_frame_dir"])
    bbox_json_path = args.bbox_json_path if args.bbox_json_path is not None else ds["bbox_json_path"]
    output_filename = args.output_filename or str(ds["output_filename"])
    detailed_output_filename = args.detailed_output_filename or str(ds["detailed_output_filename"])

    grounding_api_key = args.grounding_api_key or os.getenv("qdd_api")
    vision_api_key = args.vision_api_key or os.getenv("aliyun_api")
    main_api_key = args.main_api_key or os.getenv("qdd_api")

    if not Path(questions_path).is_file():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    if not Path(subs_path).is_file():
        raise FileNotFoundError(f"Subtitles file not found: {subs_path}")
    if not Path(base_frame_dir).is_dir():
        raise FileNotFoundError(f"Frame directory not found: {base_frame_dir}")
    if args.dataset == "tvqa_plus":
        if not bbox_json_path or not Path(str(bbox_json_path)).is_file():
            raise FileNotFoundError(f"BBox JSON file not found: {bbox_json_path}")
    if args.threads <= 0:
        raise ValueError("--threads 必须为正整数")
    if not (0.0 < args.gpu_memory_utilization <= 1.0):
        raise ValueError("--gpu_memory_utilization 必须在 (0, 1] 范围内")

    global config
    config = EvalConfig(
        dataset=args.dataset,
        questions_path=questions_path,
        subs_path=subs_path,
        base_frame_dir=base_frame_dir,
        bbox_json_path=str(bbox_json_path) if bbox_json_path else None,
        output_filename=output_filename,
        detailed_output_filename=detailed_output_filename,
        grounding_model=args.grounding_model,
        vision_model=args.vision_model,
        main_model=args.main_model,
        grounding_base_url=args.grounding_base_url,
        vision_base_url=args.vision_base_url,
        main_base_url=args.main_base_url,
        grounding_api_key=grounding_api_key,
        vision_api_key=vision_api_key,
        main_api_key=main_api_key,
        num_threads=args.threads,
        grounding_cache_json_path=args.grounding_cache_json_path,
        verbose=args.verbose,
        debug=args.debug,
    )
    initialize_clients()

    run_enhanced_pipeline(
        checkpoint_step=args.checkpoint_step,
        max_turn=args.max_turn,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_threads=args.threads,
    )


if __name__ == "__main__":
    main()
