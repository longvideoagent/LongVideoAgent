#lvagent/tvqa/evaluate_api.py
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
    questions_path: str
    subs_path: str
    base_frame_dir: str
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


config: EvalConfig | None = None


DEFAULT_SUBS_PATH = "../Tvqa/tvqa_subtitles.json"
DEFAULT_QUESTIONS_PATH = "../Tvqa/house_met/tvqa_val_house_met.jsonl"
DEFAULT_BASE_FRAME_DIR = "../Tvqa/house_met_frames"
DEFAULT_OUTPUT_FILENAME = "./eval_grok-4-fast-reasoning.json"
DEFAULT_DETAILED_OUTPUT_FILENAME = "./data_log_eval_grok-4-fast-reasoning.json"
DEFAULT_GROUNDING_MODEL = "grok-4-fast-reasoning"
DEFAULT_VISION_MODEL = "gpt-4o"
DEFAULT_MAIN_MODEL = "grok-4-fast-reasoning"
DEFAULT_GROUNDING_BASE_URL = "https://api2.aigcbest.top/v1"
DEFAULT_VISION_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MAIN_BASE_URL = "https://api2.aigcbest.top/v1"


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
    # Modified: No longer sample based on time_str; instead, fixed sampling from clip: frames 1 to 180, every 10 frames
    frame_nums = list(range(1, 181, 12))  # Results in [1, 11, 21, ..., 171], approximately 18 frames

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
            f"Images 1-{len(frame_nums)} are video frames extracted from frames 1 to 180.\nAnd here is a description of what I want to know:\n"
            f"{seg['description']}"
        )
    })

    resp = vision_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": messages_content}]
    )
    return resp.choices[0].message.content
# ───────────── 简化版action解析 ─────────────────────────────────────
def parse_action_from_response(response: str) -> Tuple[str, str]:
    """
    从响应中解析action信息 - 修改版，支持三个action，无time_range
    
    Returns:
        (action_type, content)
        action_type: 'search', 'request_grounding', 'answer', 'invalid'
    """
    # 检查search（使用re.search从头匹配第一个出现的标签）
    search_match = re.search(r'<search>(.*?)</search>', response, re.DOTALL)
    if search_match:
        content = search_match.group(1).strip()
        # 截断response到这个标签结束的位置
        end_pos = search_match.end()
        response = response[:end_pos]  # 实现后截断：只保留到第一个标签结束
        return 'search', content
    
    # 检查request_grounding
    grounding_match = re.search(r'<request_grounding>(.*?)</request_grounding>', response, re.DOTALL)
    if grounding_match:
        content = grounding_match.group(1).strip()
        # 截断response到这个标签结束的位置
        end_pos = grounding_match.end()
        response = response[:end_pos]
        return 'request_grounding', content
    
    # 检查answer
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
        # 截断response到这个标签结束的位置
        end_pos = answer_match.end()
        response = response[:end_pos]
        return 'answer', content
    
    # 无有效action
    return 'invalid', ''

# ───────────── 执行Action的处理函数 ───────────────────────────────────────
def execute_action(action_type: str, content: str, 
                  vid: str, text_client: OpenAI, vision_client: OpenAI,
                  question_data: Dict, episode_sub_block: str, clip_subtitles: Dict[str, str],
                  bbox_data: Dict = None) -> Tuple[str, bool]:
    """
    执行具体的action - 修改版，无time_range，支持request_grounding，仅视觉查询
    
    Args:
        action_type: 'answer', 'search', 'request_grounding', 'invalid'
        content: action内容
        vid: 视频ID
        text_client: 文本API客户端
        vision_client: 视觉API客户端
        question_data: 问题数据字典，包含q, a0-a4等（用于grounding）
        episode_sub_block: 剧集级别的字幕块（用于grounding）
        clip_subtitles: clip级别的字幕字典（用于获取新sub）
        bbox_data: bbox数据字典（可选，但本版本不使用）
        
    Returns:
        (result_content, is_done) - 如果是answer则返回True，否则返回False
    """
    if action_type == 'answer':
        return f"\n<answer>{content}</answer>", True  # 返回答案并结束
    
    elif action_type == 'search':
        try:
            # 🔥 只执行视觉查询（移除时间依赖和bbox查询）

            seg = {
                "description": content  # query作为description
            }
            
            information_parts = []
            
            # 原有的视觉LLM查询
            try:
                vision_response = process_and_query_seg(seg, vid, text_client, vision_client)
                information_parts.append(f"Visual Description:\n{vision_response.strip()}")
                
            except Exception as e:
                print(f"Vision LLM call failed: {e}")
                information_parts.append(f"Visual Description: Error - {str(e)}")
            

            
            # 3. 合并所有信息（仅视觉）
            combined_info = "\n".join(information_parts)
            return f'\n<information>\n{combined_info}</information>\n', False
            
        except Exception as e:
            print(f"Search action failed: {e}")
            return f'\n<information>Error: Failed to get information - {str(e)}</information>\n', False
    
    elif action_type == 'request_grounding':
        try:
            # 调用grounding逻辑，基于question_data和episode_sub_block
            grounding_result = re_analyze_single_question_api(question_data, episode_sub_block, vid, attempt_round=1)
            
            # 获取predicted_clip
            if "error" in grounding_result:
                print(f"Grounding failed: {grounding_result['error']}")
                result_content = f"Grounding failed for query: {content}. Error: {grounding_result['error']}"
                return f'\n<grounding_info>{result_content}</grounding_info>\n', False  # 不结束，继续循环
            else:
                predicted_clip = grounding_result.get("predicted_clip", vid)  # Fallback to current vid
                # 使用predicted_clip索引sub
                new_sub = get_clip_subtitle(clip_subtitles, predicted_clip)
                # 构建返回字符串
                result_content = f"<New_clip>{predicted_clip} + {new_sub}</New_clip>"
                return f'\n{result_content}\n', False  # 不结束，继续循环
        
        except Exception as e:
            print(f"Grounding action failed: {e}")
            return f'\n<grounding_info>Error: Failed to perform grounding - {str(e)}</grounding_info>\n', False

    else:  # invalid action
        return '\nMy action is not correct. I need to search, request grounding, or answer.\n', False

# ───────────── 字幕处理函数（从grounding_clip.py复制） ───────────────────────
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
    """为指定剧集构建拼接后的字幕，按clip顺序拼接"""
    # 找到所有匹配该剧集的clips
    matching_clips = {k: v for k, v in clip_subtitles.items() if k.startswith(episode_prefix)}
    
    # 按key排序（确保顺序）
    sorted_clips = sorted(matching_clips.items())
    
    # 拼接格式化的字幕
    formatted_subtitles = []
    for clip_key, subtitle_text in sorted_clips:
        formatted_subtitles.append(f"<{clip_key}>{subtitle_text}</{clip_key}>")
    
    return "\n".join(formatted_subtitles)

def extract_episode_prefix(clip_name: str) -> str:
    """从clip名称中提取剧集前缀（episode prefix，如 'met_s01e01'）"""
    if not clip_name:
        return ""
    parts = clip_name.split('_')
    if len(parts) >= 2:
        return '_'.join(parts[:2])  # 取前两部分并用 '_' 连接
    return parts[0]  # 如果不足两部分，返回第一部分

def get_subtitles_for_video(clip_subtitles: Dict[str, str], vid_name: str) -> str:
    """根据视频名获取对应的字幕块"""
    # 使用 extract_episode_prefix 提取剧集前缀
    episode_prefix = extract_episode_prefix(vid_name)
    return build_subtitles_for_episode(clip_subtitles, episode_prefix)

def grounding_llm_generate(user_content: str, model: str | None = None) -> str:
    """
    使用API调用LLM生成响应（专用于grounding/定位）
    """
    if config is None:
        raise RuntimeError("Config not initialized.")
    model = model or config.grounding_model
    try:
        response = grounding_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.6,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Grounding API call failed: {e}")
        return "Error: Failed to generate response"

def main_llm_generate(conversation_history: str, model: str | None = None) -> str:
    """
    使用API调用LLM生成响应（专用于主要pipeline的文本推理）
    """
    if config is None:
        raise RuntimeError("Config not initialized.")
    model = model or config.main_model
    try:
        response = main_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": conversation_history}],
            temperature=0.6,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Main pipeline API call failed: {e}")
        return "Error: Failed to generate response"

# ─────────────── 后处理函数（从grounding_clip.py复制） ──────────────────────
def postprocess_response(response: str) -> str:
    """对响应进行后处理和截断"""
    type_match = re.search(r'<type>.*?</type>', response, re.DOTALL)
    if type_match:
        return response[:type_match.end()]
    
    time_match = re.search(r'<time>.*?</time>', response, re.DOTALL)
    if time_match:
        return response[:time_match.end()]
    
    return response
def extract_clip_content(text: str) -> str:
    """提取<clip></clip>标签内容"""
    match = re.search(r"<clip>(.*?)</clip>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

# ─────────────── 单次分析函数（从grounding_clip.py复制，并适应推理阶段：移除GT对比） ──────────────────────
def re_analyze_single_question_api(question_data: Dict, sub_block: str, vid: str, attempt_round: int) -> Dict:
    """
    使用API分析单个问题的clip定位（推理阶段：移除GT对比）
    """
    original_idx = question_data.get('original_idx', '?')
    print(f"      Question {original_idx}: API Analysis round {attempt_round}")
    
    # 构建prompt（简化：假设选项固定为a0-a4）
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
def extract_clip_content(text: str) -> str:
    """提取<clip></clip>标签内容"""
    match = re.search(r"<clip>(.*?)</clip>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

# ─────────────── 单次分析函数（从grounding_clip.py复制，并适应推理阶段：移除GT对比） ──────────────────────
def analyze_single_question_api(
    question_data: Dict, 
    sub_block: str, 
    attempt_round: int,
    json_path: str = "/home/rliuay/runtao/proj_videoqa/Tvqa_data/grounding_pairs_6000_samples.json"
) -> Dict:
    """
    使用API分析单个问题的clip定位（推理阶段：移除GT对比）
    修改：先从JSON文件中查找匹配的question，如果找到则使用其clip作为predicted_clip；
    如果未找到，再调用API进行grounding。
    """
    original_idx = question_data.get('original_idx', '?')
    print(f"      Question {original_idx}: API Analysis round {attempt_round}")
    
    # 首先尝试从JSON文件中加载并匹配
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            grounding_pairs = json.load(f)
        
        target_question = question_data['q'].strip()  # 目标问题，移除可能的空白
        
        for key, entry in grounding_pairs.items():
            if 'question' in entry and entry['question'].strip() == target_question:
                predicted_clip = entry.get('clip', '')
                print(f"        Found match in JSON: Predicted clip: {predicted_clip}")
                return {
                    "predicted_clip": predicted_clip,
                    "raw_response": "From JSON cache",
                    "processed_response": "From JSON cache"
                }
        
        print("        No match found in JSON, falling back to API...")
    
    except Exception as e:
        print(f"        JSON loading error: {e}, falling back to API...")
    
    # 如果JSON中未找到，执行原API逻辑
    # 构建prompt（简化：假设选项固定为a0-a4）
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

# ───────────── 单样本处理函数（模块化提取，便于调试） ───────────────────────────────────
def process_single_question(prompt: str, vid: str, question: str, 
                           question_data: Dict, episode_sub_block: str, clip_subtitles: Dict[str, str],
                           max_turn: int = 5) -> Dict[str, Any]:
    """
    处理单个问题的核心逻辑（从单样本demo提取，便于独立调试或在批量中调用）。
    - 支持新action，无时间。
    - 使用main_llm_generate（API调用替换本地模型）。
    - 移除bbox依赖，仅视觉查询。
    - 修改：移除ground_truth和sub_block参数，因为sub_block已包含在prompt中，且推理阶段无需ground_truth。
    - 添加clip_subtitles参数，用于request_grounding时获取新sub。
    """
    # 初始化记录
    record = {
        "vid": vid, 
        "question": question, 
        "turns": [],
        "final_answer": "",
        "prompt": prompt
    }
    
    # 初始化完整对话prompt
    conversation_history = prompt
    
    # 动态多轮对话循环
    final_answer = ""
    
    for turn in range(max_turn):
        print(f"\n{'='*60}")
        print(f"  Turn {turn + 1}/{max_turn}")
        print(f"{'='*60}")
        
        # 如果是最后一轮，追加强制Action C的提示
        if turn == max_turn - 1:
            conversation_history += (
                "\nThis is the final turn. Please directly perform Action C and "
                "provide the final answer in <answer>...</answer> format.\n"
            )
        # 使用API生成响应（替换local_llm_generate）
        raw_response = main_llm_generate(conversation_history)
        
        # 响应截断处理（如果有postprocess_response，保留；否则使用原始）
        response = postprocess_response(raw_response) if 'postprocess_response' in globals() else raw_response
        print(f"🤖 LLM Response:\n{response}")
        
        # 直接解析action（无time）
        action_type, content = parse_action_from_response(response)
        print(f"🏷️  Parsed action - Type: {action_type}, Content: {content[:100]}...{'...' if len(content) > 100 else ''}")
        
        # 记录当前轮次
        turn_record = {
            "turn": turn + 1,
            "response": response,
            "action_type": action_type,
            "content": content,
            "is_done": False
        }
        
        # 执行action（无bbox版本）
        print(f"🚀 Executing action: {action_type}")
        result_content, is_done = execute_action(
            action_type, content, vid, text_client=grounding_client, vision_client=vision_client,
            question_data=question_data, episode_sub_block=episode_sub_block, clip_subtitles=clip_subtitles
        )
        
        turn_record["result_content"] = result_content
        turn_record["is_done"] = is_done
        
        print(f"📋 Action result:\n{result_content}")
        
        # 如果是答案，结束对话
        if is_done and action_type == 'answer':
            final_answer = content
            print(f"🎉 Found answer in turn {turn + 1}: {final_answer}")
            record["turns"].append(turn_record)
            break
        
        # 如果不是答案，将结果拼接到对话历史
        conversation_history += result_content
        print(f"🔄 Updated conversation history with result content")
        
        # 更新vid为最新predicted_clip（如果grounding返回了新clip）
        if action_type == 'request_grounding' and "<New_clip>" in result_content:
            # 提取predicted_clip（假设格式为<New_clip>clip_id + sub</New_clip>）
            match = re.search(r"<New_clip>(.*?) \+", result_content, re.DOTALL)
            if match:
                vid = match.group(1).strip()
                print(f"Updated vid to: {vid}")
        
        record["turns"].append(turn_record)
        
        # 检查是否超过最大轮数
        if turn == max_turn - 1:
            print(f"  Reached maximum turns ({max_turn})")
    
    # 保存结果
    record["final_answer"] = final_answer
    record["conversation_history"] = conversation_history
    
    return record

# ──────────────── 新增：处理单个问题的包装函数（用于多线程） ──────────────────────────
def process_question_wrapper(q: Dict, total: int, clip_subtitles: Dict[str, str], max_turn: int) -> Dict[str, Any]:
    """
    包装函数：处理单个问题，用于多线程调用。
    返回record字典。
    """
    try:
        # 为每个问题添加original_idx（便于追踪）
        q['original_idx'] = total
        print(f"Processing question {total}")
        
        # 先运行定位逻辑：使用整个episode的sub_block进行grounding
        episode_prefix = q["vid_name"]
        episode_sub_block = build_subtitles_for_episode(clip_subtitles, episode_prefix)
        
        # 调用grounding分析（默认1轮，推理阶段无GT）
        grounding_result = analyze_single_question_api(q, episode_sub_block, attempt_round=1)
        
        # 获取predicted_clip，如果失败则fallback到原vid_name
        predicted_clip = grounding_result.get("predicted_clip", q["vid_name"])
        print(f"Predicted clip for question {total}: {predicted_clip}")
        
        # 基于predicted_clip获取新的sub_block（仅该clip的字幕）
        sub_block = get_clip_subtitle(clip_subtitles, predicted_clip)
        
        # 构建初始prompt（新action空间，使用新的sub_block）
        initial_prompt = f"""You must follow these rules in every turn:
Reasoning First: Start by conducting your reasoning inside <reasoning>...</reasoning>. This is where you analyze the current information and decide your next step.
Choose One Action: After reasoning, you must choose exactly one of the following three actions:

Action A: If the current information is insufficient or you are somewhat uncertain, and you need to search for visual information on the current located <clipX>, then search for visual information. If your reasoning indicates that you lack necessary visual knowledge, you can call a visual engine. To do this, use the following format: <search>query</search>

Action B: If the current information is insufficient or you are somewhat uncertain, and you cannot obtain the final answer from the previous location and its possible visual information, then you need to call the grounding agent again for relocation, and output in the <request_grounding> format.

Action C: Provide the Final Answer
If your reasoning indicates that you have enough information to answer, provide the final answer inside <answer>...</answer>.
The answer must be concise and direct.

question: {q['q']}
a0: {q['a0']}
a1: {q['a1']}
a2: {q['a2']}
a3: {q['a3']}
a4: {q['a4']}
<information>subtitles: {sub_block}</information>
"""
        
        # 处理单个问题（使用predicted_clip作为vid，移除ground_truth和sub_block，添加clip_subtitles）
        record = process_single_question(
            prompt=initial_prompt,
            vid=predicted_clip,
            question=q['q'],
            question_data=q,
            episode_sub_block=episode_sub_block,
            clip_subtitles=clip_subtitles,
            max_turn=max_turn
        )
        
        # 在record中添加grounding信息（无GT相关）
        record["predicted_clip"] = predicted_clip
        
        # 简单统计
        print(f"  Result: {len(record['turns'])} turns, Answer: {record['final_answer']}")

        return record

    except Exception as e:
        print(f"Error processing question {total}: {e}")
        return {}  # 返回空字典表示失败

# ───────────── 主流程（批量处理版，集成新action空间，无时间依赖，仅视觉查询） ───────────────────────────────────
def run_enhanced_pipeline(checkpoint_step: str, max_turn: int = 5, 
                         gpu_memory_utilization: float = 0.4, num_threads: int = 10) -> None:
    """
    增强版主流程：批量处理多个问题，支持新action空间（search, request_grounding, answer）。
    - 移除时间范围依赖，仅在search action时执行视觉查询（使用process_and_query_seg）。
    - request_grounding使用占位符处理。
    - 移除bbox依赖，仅视觉查询。
    - 提取单样本逻辑到process_single_question，便于调试。
    - 简化输出，仅保存一个结果文件（summary）。
    - 修改：在处理单个问题前，先运行定位逻辑（grounding_clip.py逻辑），获取predicted_clip作为vid。
    - 基于predicted_clip获取新的sub_block，用于initial_prompt。
    - process_single_question移除ground_truth和sub_block参数。
    - 推理阶段调整：移除所有GT对比和相关记录字段，防止GT泄露。
    - 添加clip_subtitles传递，用于request_grounding时获取新sub。
    - 移除本地模型初始化，使用API客户端。
    - 新增：支持多线程处理，默认10个线程。
    
    Args:
        checkpoint_step: checkpoint步数
        max_turn: 最大对话轮数
        gpu_memory_utilization: GPU内存利用率
        num_threads: 线程数，默认10
    """
    if config is None:
        raise RuntimeError("Config not initialized.")

    # 加载数据 - 修改subs_path为clip级别的字幕文件（与grounding_clip.py一致）
    subs_path = config.subs_path
    with open(subs_path, encoding='utf-8') as f:
        clip_subtitles = json.load(f)

    # 加载 .jsonl 文件
    questions_path = config.questions_path
    questions = []
    with open(questions_path, encoding='utf-8') as f:
        for line in f:
            # 跳过空行（如果有）
            if line.strip():
                questions.append(json.loads(line))
    


    results: List[Dict[str, Any]] = []
    total = 0
    consecutive_errors = 0
    max_consecutive_errors = 5

    # 使用线程池并行处理问题
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交任务：每个问题一个future
        futures = []
        for idx, q in enumerate(questions, start=1):
            total = idx
            future = executor.submit(process_question_wrapper, q, total, clip_subtitles, max_turn)
            futures.append(future)

        # 收集结果
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            if record:  # 如果成功
                results.append(record)
                consecutive_errors = 0
            else:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(f"Too many consecutive errors ({consecutive_errors}), stopping...")
                    break  # 注意：这不会停止已提交的任务，但可以提前结束收集

    print(f"\nTotal processed: {total}")

    # ─── 保存简化结果文件 ─────────────────────────
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
    
    # 新增：保存详细的 conversation_history 和最后一轮 LLM output
    detailed_results = []
    for result in results:
        # 获取最后一轮的 response（如果存在 turns）
        last_response = result["turns"][-1]["response"] if result["turns"] else ""
        detailed_result = {
            "vid": result["vid"],
            "question": result["question"],
            "conversation_history": result["conversation_history"],
            "last_llm_response": last_response
        }
        detailed_results.append(detailed_result)
        
    total_vision_calls = sum(sum(1 for t in result["turns"] if t["action_type"] == "search") for result in results)
    total_grounding_calls = sum(sum(1 for t in result["turns"] if t["action_type"] == "request_grounding") for result in results)
    
    # 新增：计算准确率（在测试完全部题目之后统计）
    correct_count = 0
    for result, q in zip(results, questions):
        gt_answer = f"a{q['answer_idx']}"  # GT answer，如 "a0"
        pred_answer = result["final_answer"].strip().lower()  # 预测答案，确保小写比较
        if pred_answer == gt_answer.lower():
            correct_count += 1
    
    accuracy = correct_count / len(questions) if len(questions) > 0 else 0.0
    
    simplified_output = {
        "checkpoint_step": checkpoint_step,
        "model_path": f"../qwen2.5-7b-grpo_step-{checkpoint_step}",
        "gpu_memory_utilization": gpu_memory_utilization,
        "total": total,
        "max_turn": max_turn,
        "metadata": {
            "avg_turns": sum(r["num_turns"] for r in simplified_results) / len(simplified_results) if simplified_results else 0,
            "vision_calls_total": total_vision_calls,
            "grounding_calls_total": total_grounding_calls,
            "completed_questions": len([r for r in simplified_results if r["final_answer"]]),
            "completion_rate": len([r for r in simplified_results if r["final_answer"]]) / len(simplified_results) if simplified_results else 0,
            "accuracy": accuracy  # 新增准确率
        },
        "results": simplified_results
    }
    
    simplified_filename = config.output_filename
    with open(simplified_filename, "w", encoding="utf-8") as f:
        json.dump(simplified_output, f, ensure_ascii=False, indent=2)
    
    print(f"Summary results saved to {simplified_filename}")

    # 新增：只保存 detailed_results
    detailed_filename = config.detailed_output_filename
    with open(detailed_filename, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    # 统计信息
    if simplified_results:
        metadata = simplified_output["metadata"]
        print(f"\n📊 Statistics for 7B checkpoint-{checkpoint_step}:")
        print(f"Model path: ../qwen2.5-7b-grpo_step-{checkpoint_step}")
        print(f"GPU memory utilization: {gpu_memory_utilization}")
        print(f"Average turns per question: {metadata['avg_turns']:.2f}")
        print(f"Total vision calls: {metadata['vision_calls_total']}")
        print(f"Total grounding calls: {metadata['grounding_calls_total']}")
        print(f"Vision calls per question: {metadata['vision_calls_total']/len(simplified_results):.2f}")
        print(f"Completed questions: {metadata['completed_questions']}/{len(simplified_results)}")
        print(f"Completion rate: {metadata['completion_rate']:.2%}")
        print(f"Accuracy: {metadata['accuracy']:.2%}")  # 新增打印准确率
        
        # 轮次分布统计
        turn_counts = {}
        for r in simplified_results:
            turns = r["num_turns"]
            turn_counts[turns] = turn_counts.get(turns, 0) + 1
        
        print(f"\n🔄 Turn distribution:")
        for turns in sorted(turn_counts.keys()):
            count = turn_counts[turns]
            print(f"  {turns} turns: {count} questions ({count/len(simplified_results)*100:.1f}%)")
# Add this at the end of the file
def main():
    parser = argparse.ArgumentParser(description="运行增强版流水线 - API主模型 (TVQA)")
    parser.add_argument("--checkpoint_step", "-c", type=str, default="api", help="主模型的checkpoint步数")
    parser.add_argument("--max_turn", "-t", type=int, default=5, help="最大对话轮数")
    parser.add_argument("--gpu_memory_utilization", "-g", type=float, default=0.1, help="GPU内存利用率 (仅用于记录)")
    parser.add_argument("--threads", type=int, default=30, help="并发线程数")
    parser.add_argument("--questions-path", type=str, default=DEFAULT_QUESTIONS_PATH, help="问题文件 JSONL 路径")
    parser.add_argument("--subs-path", type=str, default=DEFAULT_SUBS_PATH, help="字幕文件 JSON 路径")
    parser.add_argument("--base-frame-dir", type=str, default=DEFAULT_BASE_FRAME_DIR, help="视频帧根目录")
    parser.add_argument("--output-filename", type=str, default=DEFAULT_OUTPUT_FILENAME, help="汇总输出文件")
    parser.add_argument("--detailed-output-filename", type=str, default=DEFAULT_DETAILED_OUTPUT_FILENAME, help="详细日志输出文件")
    parser.add_argument("--grounding-model", type=str, default=DEFAULT_GROUNDING_MODEL, help="grounding API 模型名")
    parser.add_argument("--vision-model", type=str, default=DEFAULT_VISION_MODEL, help="vision API 模型名")
    parser.add_argument("--main-model", type=str, default=DEFAULT_MAIN_MODEL, help="主推理 API 模型名")
    parser.add_argument("--grounding-base-url", type=str, default=DEFAULT_GROUNDING_BASE_URL, help="grounding API base URL")
    parser.add_argument("--vision-base-url", type=str, default=DEFAULT_VISION_BASE_URL, help="vision API base URL")
    parser.add_argument("--main-base-url", type=str, default=DEFAULT_MAIN_BASE_URL, help="主推理 API base URL")
    parser.add_argument("--grounding-api-key", type=str, default=None, help="grounding API key（默认读取环境变量 qdd_api）")
    parser.add_argument("--vision-api-key", type=str, default=None, help="vision API key（默认读取环境变量 aliyun_api）")
    parser.add_argument("--main-api-key", type=str, default=None, help="主推理 API key（默认读取环境变量 qdd_api）")
    args = parser.parse_args()

    grounding_api_key = args.grounding_api_key or os.getenv("qdd_api")
    vision_api_key = args.vision_api_key or os.getenv("aliyun_api")
    main_api_key = args.main_api_key or os.getenv("qdd_api")

    if not Path(args.questions_path).is_file():
        raise FileNotFoundError(f"Questions file not found: {args.questions_path}")
    if not Path(args.subs_path).is_file():
        raise FileNotFoundError(f"Subtitles file not found: {args.subs_path}")
    if not Path(args.base_frame_dir).is_dir():
        raise FileNotFoundError(f"Frame directory not found: {args.base_frame_dir}")
    if args.threads <= 0:
        raise ValueError("--threads 必须为正整数")
    if not (0.0 < args.gpu_memory_utilization <= 1.0):
        raise ValueError("--gpu_memory_utilization 必须在 (0, 1] 范围内")

    global config
    config = EvalConfig(
        questions_path=args.questions_path,
        subs_path=args.subs_path,
        base_frame_dir=args.base_frame_dir,
        output_filename=args.output_filename,
        detailed_output_filename=args.detailed_output_filename,
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
