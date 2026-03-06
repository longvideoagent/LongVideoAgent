import argparse
import base64
import io
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from openai import OpenAI
from PIL import Image
from vllm import LLM, SamplingParams

# --- 全局变量 ---
main_llm = None
main_tokenizer = None
sampling_params = None
grounding_client = None
vision_client = None


@dataclass
class EvalConfig:
    model_dir_template: str
    questions_path: str
    subs_path: str
    base_frame_dir: str
    output_filename_template: str
    grounding_model: str
    vision_model: str
    grounding_base_url: str
    vision_base_url: str
    grounding_api_key: str | None
    vision_api_key: str | None


config: EvalConfig | None = None


DEFAULT_MODEL_DIR_TEMPLATE = "../tvqa_qwen_3b-step-{checkpoint_step}"
DEFAULT_SUBS_PATH = "../Tvqa_data/LongTVQA_subtitles_clip_level.jsonl"
DEFAULT_QUESTIONS_PATH = "../Tvqa_data/LongTVQA_val.jsonl"
DEFAULT_BASE_FRAME_DIR = "../Tvqa_data/frames"
DEFAULT_OUTPUT_FILENAME_TEMPLATE = "./eval_tvqa-{checkpoint_step}.json"
DEFAULT_GROUNDING_MODEL = "grok-4-fast-reasoning"
DEFAULT_VISION_MODEL = "grok-4-fast-reasoning"
DEFAULT_GROUNDING_BASE_URL = "https://api2.aigcbest.top/v1"
DEFAULT_VISION_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def resolve_model_path(checkpoint_step: str, model_dir_template: str) -> str:
    return model_dir_template.format(checkpoint_step=checkpoint_step)


def initialize_clients() -> None:
    global grounding_client, vision_client, config
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

def initialize_main_model(checkpoint_step: str, gpu_memory_utilization: float = 0.4) -> str:
    """初始化本地 VLLM 主模型"""
    global main_llm, main_tokenizer, sampling_params

    if config is None:
        raise RuntimeError("Config not initialized.")
    model_path = resolve_model_path(checkpoint_step, config.model_dir_template)
    print(f"  -> Loading main model from: {model_path}")

    
    main_llm = LLM(
        model=model_path,
        tokenizer=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True
    )
    
    # 配置生成参数
    sampling_params = SamplingParams(
        temperature=0.6,  # 与原 main_llm_generate 保持一致
        max_tokens=1024,  # 与原 main_llm_generate 保持一致
        skip_special_tokens=True,
    )
    print("✅ Main VLLM Model initialized.")
    return model_path

def main_llm_generate(conversation_history: str, model: str = "qwen2.5-7b") -> str:
    """
    使用本地 VLLM 模型生成响应（替换原 API 调用）
    
    Args:
        conversation_history: 输入的对话历史
        model: 模型名称（占位符，实际使用全局 main_llm）
    
    Returns:
        生成的响应文本
    """
    global main_llm, sampling_params
    
    if not all([main_llm, sampling_params]):
        raise RuntimeError("Main VLLM Model not initialized.")
    
    try:
        # 执行推理
        outputs = main_llm.generate([conversation_history], sampling_params)
        response = outputs[0].outputs[0].text.strip()
        return response
    except Exception as e:
        print(f"Main VLLM inference failed: {e}")
        return f"Error: Failed to generate response - {str(e)}"

# --- 以下函数保持不变 ---
def sample_frames_from_time_str(
    time_str: str,
    fps: int = 3,
    num_frames: int = 7,
    window_sec: int = 5
) -> list[int]:
    ts = re.findall(r'\d{2}:\d{2}:\d{2}', time_str)
    if not ts:
        return []
    to_sec = lambda t: sum(int(x) * 60 ** i for i, x in enumerate(reversed(t.split(':'))))
    if len(ts) == 1:
        c = to_sec(ts[0]); start, end = max(0, c - window_sec), c + window_sec
    else:
        c1, c2 = map(to_sec, ts[:2]); start, end = min(c1, c2), max(c1, c2)
    s_idx, e_idx = math.floor(start * fps), math.floor(end * fps)
    total = e_idx - s_idx + 1
    if total <= 0: return []
    if total <= num_frames:
        return [i + 1 for i in range(s_idx, e_idx + 1)]
    step = (total - 1) / (num_frames - 1)
    return sorted(s_idx + round(i * step) + 1 for i in range(num_frames))

def convert_image_to_base64_data_url(path: str) -> str | None:
    try:
        with Image.open(path) as img:
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            buf = io.BytesIO(); img.save(buf, format='JPEG')
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None

def process_and_query_seg(
    seg: dict,
    vid: str,
    text_client: OpenAI,
    vision_client: OpenAI,
    base_frame_dir: str | None = None,
    model: str | None = None,
) -> str:
    if config is None:
        raise RuntimeError("Config not initialized.")
    base_frame_dir = base_frame_dir or config.base_frame_dir
    model = model or config.vision_model

    messages_content = []
    frame_nums = list(range(1, 181, 15))
    print(f"  Selected {len(frame_nums)} frames for vision query: {frame_nums}")
    for fn in frame_nums:
        img_path = Path(base_frame_dir, vid, f"{fn:05d}.jpg")
        if img_path.is_file():
            url = convert_image_to_base64_data_url(str(img_path))
            if url:
                messages_content.append({
                    "type": "image_url",
                    "image_url": {"url": url}
                })

    messages_content.append({
        "type": "text",
        "text": (
            f"Images 1-{len(frame_nums)} are video frames, you can focus on the key objects and actions in these frames.\n And here is a description of what I want to know:\n"
            f"{seg['description']}"
        )
    })

    resp = vision_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": messages_content}]
    )
    return resp.choices[0].message.content

def parse_action_from_response(response: str) -> Tuple[str, str]:
    search_match = re.search(r'<search>(.*?)</search>', response, re.DOTALL)
    if search_match:
        content = search_match.group(1).strip()
        end_pos = search_match.end()
        response = response[:end_pos]
        return 'search', content
    
    grounding_match = re.search(r'<request_grounding>(.*?)</request_grounding>', response, re.DOTALL)
    if grounding_match:
        content = grounding_match.group(1).strip()
        end_pos = grounding_match.end()
        response = response[:end_pos]
        return 'request_grounding', content
    
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
        end_pos = answer_match.end()
        response = response[:end_pos]
        return 'answer', content
    
    return 'invalid', ''

def execute_action(action_type: str, content: str, 
                  vid: str, text_client: OpenAI, vision_client: OpenAI,
                  question_data: Dict, episode_sub_block: str, clip_subtitles: Dict[str, str],
                  bbox_data: Dict = None) -> Tuple[str, bool]:
    if action_type == 'answer':
        return f"\n<answer>{content}</answer>", True
    
    elif action_type == 'search':
        try:
            seg = {"description": content}
            information_parts = []
            try:
                vision_response = process_and_query_seg(seg, vid, text_client, vision_client)
                information_parts.append(f"Visual Description:\n{vision_response.strip()}")
            except Exception as e:
                print(f"Vision LLM call failed: {e}")
                information_parts.append(f"Visual Description: Error - {str(e)}")
            
            combined_info = "\n".join(information_parts)
            return f'\n<information>{combined_info}</information>\n', False
        except Exception as e:
            print(f"Search action failed: {e}")
            return f'\n<information>Error: Failed to get information - {str(e)}</information>\n', False
    
    elif action_type == 'request_grounding':
        try:
            grounding_result = re_analyze_single_question_api(question_data, episode_sub_block, vid, attempt_round=1)
            if "error" in grounding_result:
                print(f"Grounding failed: {grounding_result['error']}")
                result_content = f"Grounding failed for query: {content}. Error: {grounding_result['error']}"
                return f'\n<grounding_info>{result_content}</grounding_info>\n', False
            else:
                predicted_clip = grounding_result.get("predicted_clip", vid)
                new_sub = get_subtitles_for_video(clip_subtitles, predicted_clip)
                result_content = f"<New_clip>{predicted_clip} + {new_sub}</New_clip>"
                return f'\n{result_content}\n', False
        except Exception as e:
            print(f"Grounding action failed: {e}")
            return f'\n<grounding_info>Error: Failed to perform grounding - {str(e)}</grounding_info>\n', False
    
    else:
        return '\nMy action is not correct. I need to search, request grounding, or answer.\n', False
def get_clip_subtitle(clip_subtitles: Dict[str, str], clip_name: str) -> str:
    """
    根据clip名称获取对应的单个clip字幕。
    
    Args:
        clip_subtitles: 包含所有clip字幕的字典，键为clip名称，值为字幕文本
        clip_name: 指定的clip名称（如 "s01e02_seg01_clip_01"）
    
    Returns:
        格式化的字幕字符串，形如 "<clip_name>subtitle_text</clip_name>"；
        如果未找到，返回空字符串
    """
    subtitle_text = clip_subtitles.get(clip_name, "")
    if subtitle_text:
        return f"<{clip_name}>{subtitle_text}</{clip_name}>"
    print(f"Warning: No subtitle found for clip {clip_name}")
    return ""

def build_subtitles_for_episode(clip_subtitles: Dict[str, str], episode_prefix: str) -> str:
    matching_clips = {k: v for k, v in clip_subtitles.items() if k.startswith(episode_prefix)}
    sorted_clips = sorted(matching_clips.items())
    formatted_subtitles = [f"<{clip_key}>{subtitle_text}</{clip_key}>" for clip_key, subtitle_text in sorted_clips]
    return "\n".join(formatted_subtitles)

def get_subtitles_for_video(clip_subtitles: Dict[str, str], vid_name: str) -> str:
    episode_prefix = extract_episode_prefix(vid_name)
    return build_subtitles_for_episode(clip_subtitles, episode_prefix)


def extract_episode_prefix(clip_name: str) -> str:
    """Extract clip prefix used by clip-level subtitle keys.

    Examples:
    - s05e02_seg02_clip_00 -> s05e02
    - met_s01e02_seg01_clip_03 -> met_s01e02
    """
    if not clip_name:
        return ""
    parts = clip_name.split("_")
    if len(parts) >= 2 and re.fullmatch(r"s\d{2}e\d{2}", parts[1]):
        return f"{parts[0]}_{parts[1]}"
    if re.fullmatch(r"s\d{2}e\d{2}", parts[0]):
        return parts[0]
    return parts[0]


def parse_answer_to_idx(answer_value: Any) -> int:
    """Parse answer formats to integer option index [0, 4]."""
    if isinstance(answer_value, int):
        if 0 <= answer_value <= 4:
            return answer_value
        raise ValueError(f"Invalid integer answer index: {answer_value}")

    if isinstance(answer_value, str):
        m = re.fullmatch(r"a([0-4])", answer_value.strip().lower())
        if m:
            return int(m.group(1))
        if answer_value.strip().isdigit():
            idx = int(answer_value.strip())
            if 0 <= idx <= 4:
                return idx

    raise ValueError(f"Unsupported answer format: {answer_value}")


def normalize_question_entry(raw_q: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize dataset entries to the fields expected by current evaluator."""
    q = dict(raw_q)
    if "vid_name" not in q:
        q["vid_name"] = q.get("occur_clip", "")
    if not q["vid_name"]:
        raise ValueError("Missing clip id: expected `vid_name` or `occur_clip`.")
    if "answer_idx" not in q:
        q["answer_idx"] = parse_answer_to_idx(q.get("answer"))
    return q


def load_questions_file(path: str) -> List[Dict[str, Any]]:
    """Load questions from JSON array or line-delimited JSONL."""
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Questions file must contain a JSON list: {path}")
            return data
        except json.JSONDecodeError:
            pass

    questions: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if not questions:
        raise ValueError(f"Questions file is empty or invalid: {path}")
    return questions


def load_subtitles_file(path: str) -> Dict[str, str]:
    """Load subtitles from JSON object or JSONL key-value lines."""
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

def grounding_llm_generate(user_content: str, model: str | None = None) -> str:
    if config is None:
        raise RuntimeError("Config not initialized.")
    if grounding_client is None:
        raise RuntimeError("Grounding client not initialized.")
    model = model or config.grounding_model

    try:
        response = grounding_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.6,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Grounding API call failed: {e}")
        return "Error: Failed to generate response"

def postprocess_response(response: str) -> str:
    type_match = re.search(r'<type>.*?</type>', response, re.DOTALL)
    if type_match:
        return response[:type_match.end()]
    time_match = re.search(r'<time>.*?</time>', response, re.DOTALL)
    if time_match:
        return response[:time_match.end()]
    return response

def postprocess_main_response(response: str) -> str:
    """
    Postprocess the main LLM response by truncating it to the end of the first matched action tag,
    with priority: <answer> > <request_grounding> > <search>.
    
    Args:
        response: The raw response from the main LLM.
    
    Returns:
        The truncated response up to the end of the first matched tag, or the original response if no match.
    """
    # 先检查 <answer>，优先级最高
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        return response[:answer_match.end()]
    
    # 然后检查 <request_grounding>
    grounding_match = re.search(r'<request_grounding>(.*?)</request_grounding>', response, re.DOTALL)
    if grounding_match:
        return response[:grounding_match.end()]
    
    # 最后检查 <search>
    search_match = re.search(r'<search>(.*?)</search>', response, re.DOTALL)
    if search_match:
        return response[:search_match.end()]
    
    # 如果都没有匹配，返回原始响应
    return response

def extract_clip_content(text: str) -> str:
    match = re.search(r"<clip>(.*?)</clip>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def re_analyze_single_question_api(question_data: Dict, sub_block: str, vid: str, attempt_round: int) -> Dict:
    original_idx = question_data.get('original_idx', '?')
    print(f"      Question {original_idx}: API Analysis round {attempt_round}")
    
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
        print(f"        Raw response: {raw_response}")
        if raw_response.startswith("Error:"):
            return {"error": raw_response}
        
        processed_response = postprocess_response(raw_response)
        predicted_clip = extract_clip_content(processed_response)
        
        print(f"        Predicted clip: {predicted_clip}")
        
        return {
            "predicted_clip": predicted_clip,
            "raw_response": raw_response,
            "processed_response": processed_response
        }
    except Exception as e:
        print(f"        ❌ Analysis error: {e}")
        return {"error": str(e)}

def analyze_single_question_api(question_data: Dict, sub_block: str, attempt_round: int) -> Dict:
    original_idx = question_data.get('original_idx', '?')
    print(f"      Question {original_idx}: API Analysis round {attempt_round}")
    
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
        print(f"        Raw response: {raw_response}")
        if raw_response.startswith("Error:"):
            return {"error": raw_response}
        
        processed_response = postprocess_response(raw_response)
        predicted_clip = extract_clip_content(processed_response)
        
        print(f"        Predicted clip: {predicted_clip}")
        
        return {
            "predicted_clip": predicted_clip,
            "raw_response": raw_response,
            "processed_response": processed_response
        }
    except Exception as e:
        print(f"        ❌ Analysis error: {e}")
        return {"error": str(e)}

def process_single_question(prompt: str, vid: str, question: str, 
                           question_data: Dict, episode_sub_block: str, clip_subtitles: Dict[str, str],
                           max_turn: int = 5) -> Dict[str, Any]:
    record = {
        "vid": vid, 
        "question": question, 
        "turns": [],
        "final_answer": "",
        "prompt": prompt
    }
    
    conversation_history = prompt
    final_answer = ""
    
    for turn in range(max_turn):
        print(f"\n{'='*60}")
        print(f"  Turn {turn + 1}/{max_turn}")
        print(f"{'='*60}")
        
        raw_response = main_llm_generate(conversation_history)
        response = postprocess_main_response(raw_response)
        print(f"🤖 LLM Response:\n{response}")
        
        action_type, content = parse_action_from_response(response)
        print(f"🏷️  Parsed action - Type: {action_type}, Content: {content[:100]}...{'...' if len(content) > 100 else ''}")
        
        turn_record = {
            "turn": turn + 1,
            "response": response,
            "action_type": action_type,
            "content": content,
            "is_done": False
        }
        
        print(f"🚀 Executing action: {action_type}")
        result_content, is_done = execute_action(
            action_type, content, vid, text_client=grounding_client, vision_client=grounding_client,
            question_data=question_data, episode_sub_block=episode_sub_block, clip_subtitles=clip_subtitles
        )
        
        turn_record["result_content"] = result_content
        turn_record["is_done"] = is_done
        
        print(f"📋 Action result:\n{result_content}")
        
        if is_done and action_type == 'answer':
            final_answer = content
            print(f"🎉 Found answer in turn {turn + 1}: {final_answer}")
            record["turns"].append(turn_record)
            break
        
        conversation_history += result_content
        print(f"🔄 Updated conversation history with result content")
        
        if action_type == 'request_grounding' and "<New_clip>" in result_content:
            match = re.search(r"<New_clip>(.*?) \+", result_content, re.DOTALL)
            if match:
                vid = match.group(1).strip()
                print(f"Updated vid to: {vid}")
        
        record["turns"].append(turn_record)
        
        if turn == max_turn - 1:
            print(f"  Reached maximum turns ({max_turn})")
    
    record["final_answer"] = final_answer
    record["conversation_history"] = conversation_history
    
    return record

def run_enhanced_pipeline(
    checkpoint_step: str,
    max_turn: int = 5,
    gpu_memory_utilization: float = 0.4,
) -> None:
    if config is None:
        raise RuntimeError("Config not initialized.")

    model_path = initialize_main_model(checkpoint_step, gpu_memory_utilization)

    clip_subtitles = load_subtitles_file(config.subs_path)
    questions = load_questions_file(config.questions_path)
    normalized_questions = [normalize_question_entry(q) for q in questions]



    results: List[Dict[str, Any]] = []
    total = 0
    consecutive_errors = 0
    max_consecutive_errors = 5

    for q in normalized_questions:
        try:
            q['original_idx'] = total
            total += 1
            print(f"Processing question {total}")
            
            episode_prefix = q.get("episode_name", "") or extract_episode_prefix(q["vid_name"])
            episode_sub_block = build_subtitles_for_episode(clip_subtitles, episode_prefix)
            
            grounding_result = analyze_single_question_api(q, episode_sub_block, attempt_round=1)
            predicted_clip = grounding_result.get("predicted_clip", q["vid_name"])
            print(f"Predicted clip for question {total}: {predicted_clip}")
            
            # 使用get_clip_subtitle获取单个clip的字幕（仅predicted_clip）
            sub_block = get_clip_subtitle(clip_subtitles, predicted_clip)
            
            initial_prompt = f"""You must follow these rules in every turn:
Reasoning First: Start by conducting your reasoning inside <reasoning>...</reasoning>. This is where you analyze the current information and decide your next step.
Choose One Action: After reasoning, you must choose exactly one of the following three actions:

Action A: If the current information is insufficient or you are somewhat uncertain, and you need to search for visual information on the current located <clipX>, then search for visual information. If your reasoning indicates that you lack necessary visual knowledge, you can call a visual engine. To do this, use the following format: <search>query</search>

Action B: If the current information is insufficient or you are somewhat uncertain, and you cannot obtain the final answer from the previous location and its possible visual information, then you need to call the grounding agent again for relocation, and output in the <request_grounding> format.

Action C: Provide the Final Answer
If your reasoning indicates that you have enough information to answer, provide the final answer inside <answer>...</answer>.
The answer must be concise and direct, in the format <answer>ax</answer>, where 'x' is the index of the selected option (e.g., <answer>a1</answer> for option a1).

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
                question=q['q'],
                question_data=q,
                episode_sub_block=episode_sub_block,
                clip_subtitles=clip_subtitles,
                max_turn=max_turn
            )
            
            record["predicted_clip"] = predicted_clip
            results.append(record)
            consecutive_errors = 0
            
            print(f"  Result: {len(record['turns'])} turns, Answer: {record['final_answer']}, gt: a{q['answer_idx']}")
            if record['final_answer'].strip().lower() == f"a{q['answer_idx']}".lower():
                print("  ✅ Correct answer!")
        
        except Exception as e:
            print(f"Error processing question {total}: {e}")
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                print(f"Too many consecutive errors ({consecutive_errors}), stopping...")
                break
    
    print(f"\nTotal processed: {total}")
    
    simplified_results = []
    for result in results:
        simplified_result = {
            "vid": result["vid"],
            "question": result["question"],
            "num_turns": len(result["turns"]),
            "final_answer": result["final_answer"],
            "predicted_clip": result.get("predicted_clip", "")
        }
        simplified_results.append(simplified_result)
    
    total_vision_calls = sum(sum(1 for t in result["turns"] if t["action_type"] == "search") for result in results)
    total_grounding_calls = sum(sum(1 for t in result["turns"] if t["action_type"] == "request_grounding") for result in results)
    
    correct_count = 0
    for result, q in zip(results, normalized_questions):
        gt_answer = f"a{q['answer_idx']}"
        pred_answer = result["final_answer"].strip().lower()
        if pred_answer == gt_answer.lower():
            correct_count += 1
    
    accuracy = correct_count / len(questions) if len(questions) > 0 else 0.0
    
    simplified_output = {
        "checkpoint_step": checkpoint_step,
        "model_path": model_path,
        "gpu_memory_utilization": gpu_memory_utilization,
        "total": total,
        "max_turn": max_turn,
        "metadata": {
            "avg_turns": sum(r["num_turns"] for r in simplified_results) / len(simplified_results) if simplified_results else 0,
            "vision_calls_total": total_vision_calls,
            "grounding_calls_total": total_grounding_calls,
            "completed_questions": len([r for r in simplified_results if r["final_answer"]]),
            "completion_rate": len([r for r in simplified_results if r["final_answer"]]) / len(simplified_results) if simplified_results else 0,
            "accuracy": accuracy
        },
        "results": simplified_results
    }
    
    simplified_filename = config.output_filename_template.format(checkpoint_step=checkpoint_step)
    with open(simplified_filename, "w", encoding="utf-8") as f:
        json.dump(simplified_output, f, ensure_ascii=False, indent=2)
    
    print(f"Summary results saved to {simplified_filename}")
    
    if simplified_results:
        metadata = simplified_output["metadata"]
        print(f"\n📊 Statistics for checkpoint-{checkpoint_step}:")
        print(f"Model path: {model_path}")
        print(f"GPU memory utilization: {gpu_memory_utilization}")
        print(f"Average turns per question: {metadata['avg_turns']:.2f}")
        print(f"Total vision calls: {metadata['vision_calls_total']}")
        print(f"Total grounding calls: {metadata['grounding_calls_total']}")
        print(f"Vision calls per question: {metadata['vision_calls_total']/len(simplified_results):.2f}")
        print(f"Completed questions: {metadata['completed_questions']}/{len(simplified_results)}")
        print(f"Completion rate: {metadata['completion_rate']:.2%}")
        print(f"Accuracy: {metadata['accuracy']:.2%}")
        
        turn_counts = {}
        for r in simplified_results:
            turns = r["num_turns"]
            turn_counts[turns] = turn_counts.get(turns, 0) + 1
        
        print(f"\n🔄 Turn distribution:")
        for turns in sorted(turn_counts.keys()):
            count = turn_counts[turns]
            print(f"  {turns} turns: {count} questions ({count/len(simplified_results)*100:.1f}%)")
    
    # 释放 VLLM 资源
    if main_llm:
        del main_llm
        torch.cuda.empty_cache()
        print("🔄 Main VLLM resources released")

def main():
    parser = argparse.ArgumentParser(description="运行增强版流水线 - 本地VLLM主模型")
    parser.add_argument("--checkpoint_step", "-c", type=str, default="api", help="主模型的checkpoint步数")
    parser.add_argument("--max_turn", "-t", type=int, default=5, help="最大对话轮数")
    parser.add_argument("--gpu_memory_utilization", "-g", type=float, default=0.4, help="GPU内存利用率 (0.0-1.0)")
    parser.add_argument(
        "--model-dir-template",
        type=str,
        default=DEFAULT_MODEL_DIR_TEMPLATE,
        help="主模型目录模板，支持 {checkpoint_step} 占位符",
    )
    parser.add_argument("--questions-path", type=str, default=DEFAULT_QUESTIONS_PATH, help="问题文件 JSON 路径")
    parser.add_argument("--subs-path", type=str, default=DEFAULT_SUBS_PATH, help="字幕文件 JSON 路径")
    parser.add_argument("--base-frame-dir", type=str, default=DEFAULT_BASE_FRAME_DIR, help="视频帧根目录")
    parser.add_argument(
        "--output-filename-template",
        type=str,
        default=DEFAULT_OUTPUT_FILENAME_TEMPLATE,
        help="输出文件模板，支持 {checkpoint_step} 占位符",
    )
    parser.add_argument("--grounding-model", type=str, default=DEFAULT_GROUNDING_MODEL, help="grounding API 模型名")
    parser.add_argument("--vision-model", type=str, default=DEFAULT_VISION_MODEL, help="vision API 模型名")
    parser.add_argument(
        "--grounding-base-url",
        type=str,
        default=DEFAULT_GROUNDING_BASE_URL,
        help="grounding API base URL",
    )
    parser.add_argument(
        "--vision-base-url",
        type=str,
        default=DEFAULT_VISION_BASE_URL,
        help="vision API base URL",
    )
    parser.add_argument(
        "--grounding-api-key",
        type=str,
        default=None,
        help="grounding API key（默认读取环境变量 qdd_api）",
    )
    parser.add_argument(
        "--vision-api-key",
        type=str,
        default=None,
        help="vision API key（默认读取环境变量 aliyun_api）",
    )

    args = parser.parse_args()

    grounding_api_key = args.grounding_api_key or os.getenv("qdd_api")
    vision_api_key = args.vision_api_key or os.getenv("aliyun_api")

    if not Path(args.questions_path).is_file():
        raise FileNotFoundError(f"Questions file not found: {args.questions_path}")
    if not Path(args.subs_path).is_file():
        raise FileNotFoundError(f"Subtitles file not found: {args.subs_path}")
    if not Path(args.base_frame_dir).is_dir():
        raise FileNotFoundError(f"Frame directory not found: {args.base_frame_dir}")
    if "{checkpoint_step}" not in args.model_dir_template:
        raise ValueError("--model-dir-template 必须包含 {checkpoint_step}")
    if "{checkpoint_step}" not in args.output_filename_template:
        raise ValueError("--output-filename-template 必须包含 {checkpoint_step}")
    if not (0.0 < args.gpu_memory_utilization <= 1.0):
        raise ValueError("--gpu_memory_utilization 必须在 (0, 1] 范围内")

    global config
    config = EvalConfig(
        model_dir_template=args.model_dir_template,
        questions_path=args.questions_path,
        subs_path=args.subs_path,
        base_frame_dir=args.base_frame_dir,
        output_filename_template=args.output_filename_template,
        grounding_model=args.grounding_model,
        vision_model=args.vision_model,
        grounding_base_url=args.grounding_base_url,
        vision_base_url=args.vision_base_url,
        grounding_api_key=grounding_api_key,
        vision_api_key=vision_api_key,
    )
    initialize_clients()

    run_enhanced_pipeline(
        checkpoint_step=args.checkpoint_step,
        max_turn=args.max_turn,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

if __name__ == "__main__":
    main()
