import torch
import re
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
import io
import base64
from PIL import Image
from pathlib import Path
from openai import OpenAI
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class VisionGenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool = False
    # 新增：视觉相关配置
    base_frame_dir: str = "../bbt_frames"
    fps: int = 3
    num_frames: int = 7
    window_sec: int = 5
    vision_model: str = "gpt-4o"
    grounding_model: str = "grok-4-fast-reasoning"
    grounding_temperature: float = 0.6
    grounding_max_tokens: int = 512
    api_key: Optional[str] = None  # 通用API密钥，如果不同客户端使用不同，可扩展
    grounding_api: Optional[str] = None
    vision_api: Optional[str] = None
    main_api: Optional[str] = None
    grounding_base_url: str = "https://api2.aigcbest.top/v1"
    vision_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    main_base_url: str = "https://api2.aigcbest.top/v1"
    video_id_key: str = "vid_name"  # 数据集中视频ID的键名
    topk: int = 3
    bbox_json_path: str = "bbox_annotations.json"
    rollout_n: int = 1
    frame_start: int = 1
    frame_end: int = 180
    frame_step: int = 15
    # 新增：字幕路径等
    subs_path: str = "../Tvqa_data/all_episodes_subtitles_by_clips.json"

class VisionLLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: VisionGenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        # 初始化bbox缓存（尽管本版不使用，但保持兼容）
        self.bbox_json_cache = None

        # 初始化API客户端
        shared_api_key = config.api_key
        self.grounding_client = OpenAI(
            api_key=config.grounding_api or shared_api_key or os.getenv("qdd_api"),
            base_url=config.grounding_base_url,
        )
        self.vision_client = OpenAI(
            api_key=config.vision_api or shared_api_key or os.getenv("aliyun_api"),
            base_url=config.vision_base_url,
        )
        self.main_client = OpenAI(
            api_key=config.main_api or shared_api_key or os.getenv("qdd_api"),
            base_url=config.main_base_url,
        )

        # 加载clip级字幕（全局共享）
        with open(config.subs_path, encoding="utf-8") as f:
            self.clip_subtitles = json.load(f)

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    # ================= bbox function (保持但不使用) =================
    def _load_bbox_json(self) -> dict:
        """加载BBOX标注JSON文件"""
        if self.bbox_json_cache is None:
            try:
                with open(self.config.bbox_json_path, 'r', encoding='utf-8') as f:
                    self.bbox_json_cache = json.load(f)
            except FileNotFoundError:
                print(f"BBOX JSON文件未找到: {self.config.bbox_json_path}")
                self.bbox_json_cache = {}
            except json.JSONDecodeError:
                print(f"BBOX JSON文件格式错误: {self.config.bbox_json_path}")
                self.bbox_json_cache = {}
            except Exception as e:
                print(f"加载BBOX JSON文件失败: {e}")
                self.bbox_json_cache = {}
        return self.bbox_json_cache

 

    def _get_bbox_content(self, vid: str) -> str:
        """根据vid获取BBOX信息"""
        if not vid:
            return "{}"  # 如果vid为空，返回空JSON字符串
        bbox_data = self._load_bbox_json()
        return json.dumps(bbox_data.get(vid, {}))

    # ================= 字幕处理函数 =================
    def extract_episode_prefix(self, clip_name: str) -> str:
        """从clip名称中提取剧集前缀（episode prefix，如 's01_e01'）"""
        if not clip_name:
            return ""
        return clip_name[:6]
    def build_subtitles_for_episode(self, episode_prefix: str) -> str:
        """为指定剧集构建拼接后的全部字幕（episode 级别）"""
        if not episode_prefix:
            return ""
        matching_clips = {k: v for k, v in self.clip_subtitles.items() if k.startswith(episode_prefix)}
        if not matching_clips:
            print(f"Warning: No subtitles found for episode prefix '{episode_prefix}'")
            return ""
        sorted_clips = sorted(matching_clips.items())  # 按 clip_key 排序，确保顺序
        formatted_subtitles = [f"<{clip_key}>{subtitle_text}</{clip_key}>" for clip_key, subtitle_text in sorted_clips]
        return "\n".join(formatted_subtitles)

    def get_clip_subtitle(self, clip_name: str) -> str:
        """获取单个 clip 的字幕（clip 级别）"""
        if not clip_name:
            return ""
        subtitle = self.clip_subtitles.get(clip_name, "")
        if not subtitle:
            print(f"Warning: No subtitle found for clip '{clip_name}'")
        return subtitle
    # ================= 视觉查询函数 =================
    def enhanced_process_and_query_seg(self, seg: dict, vid: str) -> str:
        """增强版视觉查询：固定采样，无时间依赖"""
        messages_content = []
        # 固定采样：默认 1-180，每15帧（可配置）
        frame_nums = list(range(self.config.frame_start, self.config.frame_end + 1, self.config.frame_step))

        for fn in frame_nums:
            img_path = Path(self.config.base_frame_dir, vid, f"{fn:05d}.jpg")
            if img_path.is_file():
                url = self._convert_image_to_base64_data_url(str(img_path))
                if url:
                    messages_content.append({
                        "type": "image_url",
                        "image_url": {"url": url}
                    })

        messages_content.append({
            "type": "text",
            "text": (
                f"Images 1-{len(frame_nums)} are video frames. Focus on key objects, actions, scene transitions, and inferences.\nAnswer the following question based on these images:\n"
                f"{seg['description']}"
            )
        })

        try:
            # Vision query should use the dedicated vision client endpoint.
            resp = self.vision_client.chat.completions.create(
                model=self.config.vision_model,
                messages=[{"role": "user", "content": messages_content}]
            )
            time.sleep(0.1)  # 限流
            return resp.choices[0].message.content
        except Exception as e:
            print(f"Vision LLM call failed: {e}")
            return f"Error: {str(e)}"

    def _convert_image_to_base64_data_url(self, path: str) -> Optional[str]:
        """Convert image to base64 data URL."""
        try:
            with Image.open(path) as img:
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                buf = io.BytesIO()
                img.save(buf, format='JPEG')
                return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        except (OSError, ValueError):
            return None

    # ================= grounding 函数 =================
    def analyze_single_question_api(self, question_data: Dict, sub_block: Optional[str], current_vid: str = "") -> Dict:
        """API分析clip定位"""

        if sub_block is None and current_vid:
            episode_prefix = self.extract_episode_prefix(current_vid)
            sub_block = self.build_subtitles_for_episode(episode_prefix)
        prompt_content = f"""
Question: {question_data['q']}
Options:
a0: {question_data.get('a0', '')}
a1: {question_data.get('a1', '')}
a2: {question_data.get('a2', '')}
a3: {question_data.get('a3', '')}
a4: {question_data.get('a4', '')}
Subtitles: {sub_block}

The subtitles are formatted as <clip_label>subtitle_content</clip_label>.

Based on the question and subtitles, determine the specific clip label where the event or context related to the question occurs.
{current_vid} may not contain the scene or context related to the question. Please determine a different specific clip label.
You can output in <clip>clip_label</clip>.
"""
        try:
            raw_response = self.grounding_client.chat.completions.create(
                model=self.config.grounding_model,
                messages=[{"role": "user", "content": prompt_content}],
                temperature=self.config.grounding_temperature,
                max_tokens=self.config.grounding_max_tokens,
            ).choices[0].message.content
            processed_response = self.postprocess_response(raw_response)
            predicted_clip = self.extract_clip_content(processed_response)
            return {
                "predicted_clip": predicted_clip,
                "raw_response": raw_response,
                "processed_response": processed_response
            }
        except Exception as e:
            return {"error": str(e)}

    def postprocess_response(self, response: str) -> str:
        """后处理响应"""
        clip_match = re.search(r'<clip>.*?</clip>', response, re.DOTALL)
        if clip_match:
            return response[:clip_match.end()]
        return response

    def extract_clip_content(self, text: str) -> str:
        """提取<clip>内容"""
        match = re.search(r"<clip>(.*?)</clip>", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    # ================= action 解析与执行 =================
    def parse_action_from_response(self, response: str) -> Tuple[str, str]:
        """解析action"""
        search_match = re.search(r'<search>(.*?)</search>', response, re.DOTALL)
        if search_match:
            return 'search', search_match.group(1).strip()
        grounding_match = re.search(r'<request_grounding>(.*?)</request_grounding>', response, re.DOTALL)
        if grounding_match:
            return 'request_grounding', grounding_match.group(1).strip()
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            return 'answer', answer_match.group(1).strip()
        return 'invalid', ''

    def execute_action(self, action_type: str, content: str, 
                      vid: str, question_data: Dict, episode_sub_block: str) -> Tuple[str, bool]:
        """执行单个action"""
        if action_type == 'answer':
            return f"\n<answer>{content}</answer>", True
        elif action_type == 'search':
            seg = {"description": content}
            vision_response = self.enhanced_process_and_query_seg(seg, vid)
            bbox_content = self._get_bbox_content(vid)  # 获取BBOX信息
            return f"\n<information>Bounding BOX:\n{bbox_content.strip()}\nVisual Description:\n{vision_response.strip()}</information>\n", False
        elif action_type == 'request_grounding':
            grounding_result = self.analyze_single_question_api(question_data, sub_block=None, current_vid=vid)
            if "error" in grounding_result:
                return f"\n<grounding_info>Error: {grounding_result['error']}</grounding_info>\n", False
            predicted_clip = grounding_result.get("predicted_clip", vid)
            new_sub = self.get_clip_subtitle(predicted_clip)
            result_content = f"<New_clip>{predicted_clip} + {new_sub}</New_clip>"
            return f"\n{result_content}\n", False
        else:
            return "\nMy action is not correct. I need to search, request grounding, or answer.\n", False

    def _postprocess_responses(self, responses: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """后处理响应"""
        responses_str = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        processed_responses = []
        for resp in responses_str:
            # 截断到第一个有效标签结束
            search_match = re.search(r'<search>(.*?)</search>', resp, re.DOTALL)
            if search_match:
                processed_responses.append(resp[:search_match.end()])
                continue
            grounding_match = re.search(r'<request_grounding>(.*?)</request_grounding>', resp, re.DOTALL)
            if grounding_match:
                processed_responses.append(resp[:grounding_match.end()])
                continue
            answer_match = re.search(r'<answer>(.*?)</answer>', resp, re.DOTALL)
            if answer_match:
                processed_responses.append(resp[:answer_match.end()])
                continue
            processed_responses.append(resp)
        responses = self._batch_tokenize(processed_responses)
        return responses, processed_responses

    def postprocess_predictions(self, predictions: List[str]) -> Tuple[List[str], List[str]]:
        """提取action_type和content"""
        actions = []
        contents = []
        for prediction in predictions:
            action_type, content = self.parse_action_from_response(prediction)
            actions.append(action_type)
            contents.append(content)
        return actions, contents

    def execute_predictions(self, predictions: List[str], _pad_token: int, active_mask=None,
                        question_data: List[Dict]=None, episode_sub_blocks: List[str]=None) -> Tuple[List[str], List[bool], List[int], List[int], List[int]]:
        """批量执行predictions"""
        if active_mask is None:
            raise ValueError("active_mask must be provided")

        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs = [''] * len(predictions)
        dones = [True] * len(predictions)
        valid_action = [0] * len(predictions)
        is_vision = [0] * len(predictions)
        is_grounding = [0] * len(predictions)

        # 收集需要API调用的轨迹
        api_tasks = []
        for i, (action, content, active) in enumerate(zip(cur_actions, contents, active_mask)):
            if not active or action not in ['search', 'request_grounding']:
                continue
            vid = self.video_ids[i]
            q_data = question_data[i] if question_data else {}
            ep_sub = episode_sub_blocks[i] if episode_sub_blocks else ""
            api_tasks.append((i, action, content, vid, q_data, ep_sub))

        # 使用ThreadPoolExecutor并行执行API调用
        if api_tasks:
            with ThreadPoolExecutor(max_workers=self.config.rollout_n) as executor:
                # 提交任务
                future_to_index = {
                    executor.submit(self.execute_action, action, content, vid, q_data, ep_sub): i
                    for i, action, content, vid, q_data, ep_sub in api_tasks
                }
                # 收集结果
                for future in as_completed(future_to_index):
                    i = future_to_index[future]
                    try:
                        result_content, is_done = future.result()
                        next_obs[i] = result_content
                        dones[i] = is_done
                        valid_action[i] = 1 if cur_actions[i] != 'invalid' else 0
                        is_vision[i] = 1 if cur_actions[i] == 'search' else 0
                        is_grounding[i] = 1 if cur_actions[i] == 'request_grounding' else 0

                        # 更新vid如果grounding
                        if cur_actions[i] == 'request_grounding' and "<New_clip>" in result_content:
                            match = re.search(r"<New_clip>(.*?) \+", result_content)
                            if match:
                                self.video_ids[i] = match.group(1).strip()
                    except Exception as e:
                        print(f"API call failed for index {i}: {e}")
                        next_obs[i] = f"\n<grounding_info>Error: {str(e)}</grounding_info>\n"
                        dones[i] = False
                        valid_action[i] = 0
                        is_vision[i] = 0
                        is_grounding[i] = 0

        return next_obs, dones, valid_action, is_vision, is_grounding

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations."""
        next_obs_ids = self.tokenizer(next_obs, padding='longest', return_tensors='pt', add_special_tokens=False)['input_ids']
        if next_obs_ids.shape[1] > self.config.max_obs_length:
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]
        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                              next_obs_ids: torch.Tensor) -> DataProto:
        """Update rolling state."""
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        return new_rollings

    def _update_right_side(self, right_side: Dict, cur_responses: torch.Tensor, 
                           next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        tensors = [right_side['responses'], cur_responses]
        tensors_with_mask = [right_side['responses_with_info_mask'], cur_responses]
        if next_obs_ids is not None:
            tensors.append(next_obs_ids)
            info_mask = torch.full(next_obs_ids.size(), self.tokenizer.pad_token_id, dtype=next_obs_ids.dtype, device=next_obs_ids.device)
            tensors_with_mask.append(info_mask)
        responses = torch.cat(tensors, dim=1)
        responses_with_info_mask = torch.cat(tensors_with_mask, dim=1)
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = self.config.max_prompt_length
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """Generation with GPU padding."""
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        padding_size = num_gpus - remainder
        padded_batch = {}
        for k, v in active_batch.batch.items():
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
        padded_active_batch = DataProto.from_dict(padded_batch)
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {k: v[:-padding_size] if isinstance(v, torch.Tensor) else v for k, v in padded_output.meta_info.items()}
            padded_output.meta_info = trimmed_meta
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor, video_ids: List[str] = None,
                     question_data: List[Dict] = None, episode_sub_blocks: List[str] = None) -> DataProto:
        """Main generation loop."""
        self.video_ids = video_ids or []

        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': torch.empty_like(initial_input_ids[:, []]), 'responses_with_info_mask': torch.empty_like(initial_input_ids[:, []])}
        

        # 如果未提供 question_data，从 gen_batch.non_tensor_batch['extra_info'] 构建
        if question_data is None:
            extra_info_data = gen_batch.non_tensor_batch.get('extra_info', [])
            batch_size = gen_batch.batch['input_ids'].shape[0]
            
            # 处理 extra_info_data 的格式
            extra_infos = []
            if isinstance(extra_info_data, list):
                extra_infos = extra_info_data
            elif isinstance(extra_info_data, dict):
                extra_infos = [extra_info_data]
            else:
                try:
                    extra_infos = list(extra_info_data)
                except TypeError as e:
                    print(f"Warning: Invalid extra_info format in gen_batch.non_tensor_batch ({e}), using defaults")
                    extra_infos = [{} for _ in range(batch_size)]
            
            # 验证长度
            if len(extra_infos) != batch_size:
                print(f"Warning: extra_info length mismatch: expected {batch_size}, got {len(extra_infos)}")
                extra_infos = extra_infos[:batch_size] + [{} for _ in range(batch_size - len(extra_infos))]
            
            question_data = []
            for extra in extra_infos:
                if not isinstance(extra, dict):
                    print(f"Warning: Invalid extra_info, not a dict: {extra}")
                    q_data = {'q': ''}
                else:
                    q_data = {'q': extra.get('original_question', '')}
                    # 提取 choices，映射 '0' 到 'a0'，'1' 到 'a1' 等
                    choices = extra.get('choices', {})
                    if isinstance(choices, dict):
                        for i in range(5):  # 假设最多 5 个选项
                            q_data[f'a{i}'] = choices.get(str(i), '')
                    else:
                        print(f"Warning: Invalid choices format in extra_info, expected dict: {choices}")
                question_data.append(q_data)
            


        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.zeros_like(active_mask, dtype=torch.int)
        valid_action_stats = torch.zeros_like(active_mask, dtype=torch.int)
        valid_vision_stats = torch.zeros_like(active_mask, dtype=torch.int)
        valid_grounding_stats = torch.zeros_like(active_mask, dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        for step in range(self.config.max_turns):
            print(f"=========Turn{step}=========:")
            if not active_mask.any():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(rollings.batch, keys=['input_ids', 'attention_mask', 'position_ids'])
            rollings_active = DataProto.from_dict({k: v[active_mask] for k, v in rollings.batch.items()})
            gen_output = self._generate_with_gpu_padding(rollings_active)
            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            next_obs, dones, valid_action, is_vision, is_grounding = self.execute_predictions(
                responses_str, self.tokenizer.pad_token_id, active_mask, question_data, episode_sub_blocks
            )
            
            curr_active_mask = torch.tensor([not d for d in dones], dtype=torch.bool)
            active_mask = active_mask & curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[~curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action)
            valid_vision_stats += torch.tensor(is_vision)
            valid_grounding_stats += torch.tensor(is_grounding)

            next_obs_ids = self._process_next_obs(next_obs)
            rollings = self._update_rolling_state(rollings, responses_ids, next_obs_ids)
            original_right_side = self._update_right_side(original_right_side, responses_ids, next_obs_ids)

        # Final rollout if needed
        if active_mask.any():
            rollings.batch = self.tensor_fn.cut_to_effective_len(rollings.batch, keys=['input_ids', 'attention_mask', 'position_ids'])
            rollings_active = DataProto.from_dict({k: v[active_mask] for k, v in rollings.batch.items()})
            gen_output = self._generate_with_gpu_padding(rollings_active)
            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            _, dones, valid_action, is_vision, is_grounding = self.execute_predictions(
                responses_str, self.tokenizer.pad_token_id, active_mask, question_data, episode_sub_blocks
            )

            curr_active_mask = torch.tensor([not d for d in dones], dtype=torch.bool)
            active_mask = active_mask & curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action)
            valid_vision_stats += torch.tensor(is_vision)
            valid_grounding_stats += torch.tensor(is_grounding)

            original_right_side = self._update_right_side(original_right_side, responses_ids)

        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_vision_stats'] = valid_vision_stats.tolist()
        meta_info['valid_grounding_stats'] = valid_grounding_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict, right_side: Dict, meta_info: Dict) -> DataProto:
        """Compose final output with type safety."""
        # 确保输入都是 int64
        if isinstance(left_side['input_ids'], torch.Tensor):
            left_side['input_ids'] = left_side['input_ids'].to(torch.int64)
        
        if isinstance(right_side['responses'], torch.Tensor):
            right_side['responses'] = right_side['responses'].to(torch.int64)
        
        if isinstance(right_side['responses_with_info_mask'], torch.Tensor):
            right_side['responses_with_info_mask'] = right_side['responses_with_info_mask'].to(torch.int64)

        final_output = DataProto.from_dict(right_side)
        final_output.batch['prompts'] = left_side['input_ids']
        final_output.batch['input_ids'] = torch.cat([left_side['input_ids'], right_side['responses']], dim=1)
        
        # 确保拼接后的 input_ids 是 int64
        final_output.batch['input_ids'] = final_output.batch['input_ids'].to(torch.int64)
        
        final_output.batch['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(right_side['responses'])
        ], dim=1)
        
        final_output.batch['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(right_side['responses_with_info_mask'])
        ], dim=1)
        
        final_output.batch['position_ids'] = self.tensor_fn.create_position_ids(final_output.batch['attention_mask'])
        
        # 检查并添加 timing
        if 'timing' not in meta_info:
            meta_info['timing'] = {}
        final_output.meta_info.update(meta_info)
        
        return final_output
