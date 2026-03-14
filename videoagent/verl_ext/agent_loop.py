import base64
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__name__)


@register("longvideoagent_multiturn")
class LongVideoAgentLoop(AgentLoopBase):
    """LongVideoAgent multi-turn rollout implemented on verl_new AgentLoop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.response_length = int(self.rollout_config.response_length)
        self.max_tool_response_length = int(self.rollout_config.multi_turn.max_tool_response_length)

        custom_cfg = self.rollout_config.get("custom", {}) or {}
        video_cfg = custom_cfg.get("videoagent", {}) if hasattr(custom_cfg, "get") else {}

        def cfg(name: str, default: Any) -> Any:
            if hasattr(video_cfg, "get"):
                return video_cfg.get(name, default)
            return default

        max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        if max_assistant_turns is None:
            max_assistant_turns = cfg("max_turns", 1)
        self.max_assistant_turns = int(max_assistant_turns)

        max_user_turns = self.rollout_config.multi_turn.max_user_turns
        if max_user_turns is None:
            max_user_turns = self.max_assistant_turns
        self.max_user_turns = int(max_user_turns)

        max_obs_from_data = self.config.data.get("max_obs_length", self.response_length)
        self.max_obs_length = int(cfg("max_obs_length", max_obs_from_data))

        self.base_frame_dir = str(cfg("base_frame_dir", "../bbt_frames"))
        self.subs_path = str(cfg("subs_path", "../Tvqa_data/all_episodes_subtitles_by_clips.json"))
        self.bbox_json_path = str(cfg("bbox_json_path", "bbox_annotations.json"))

        self.vision_model = str(cfg("vision_model", "gpt-4o"))
        self.grounding_model = str(cfg("grounding_model", "grok-4-fast-reasoning"))
        self.grounding_temperature = float(cfg("grounding_temperature", 0.6))
        self.grounding_max_tokens = int(cfg("grounding_max_tokens", 512))

        self.frame_start = int(cfg("frame_start", 1))
        self.frame_end = int(cfg("frame_end", 180))
        self.frame_step = int(cfg("frame_step", 15))

        shared_api_key = cfg("api_key", None)
        grounding_api = cfg("grounding_api", shared_api_key or os.getenv("qdd_api"))
        vision_api = cfg("vision_api", shared_api_key or os.getenv("aliyun_api"))

        grounding_base_url = str(cfg("grounding_base_url", "https://api2.aigcbest.top/v1"))
        vision_base_url = str(cfg("vision_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"))

        try:
            from openai import OpenAI
        except Exception as exc:
            raise ImportError("LongVideoAgentLoop requires `openai` package, please install it.") from exc

        self.grounding_client = OpenAI(api_key=grounding_api, base_url=grounding_base_url)
        self.vision_client = OpenAI(api_key=vision_api, base_url=vision_base_url)

        self.clip_subtitles = self._load_json_safely(self.subs_path)
        self.bbox_cache = self._load_json_safely(self.bbox_json_path)

    @staticmethod
    def _load_json_safely(path: str) -> dict:
        if not path:
            return {}
        file_path = Path(path)
        if not file_path.exists():
            logger.warning("JSON file not found: %s", path)
            return {}
        try:
            with file_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            logger.exception("Failed to load json: %s", path)
            return {}

    @staticmethod
    def _extract_episode_prefix(clip_name: str) -> str:
        return clip_name[:6] if clip_name else ""

    def _build_subtitles_for_episode(self, episode_prefix: str) -> str:
        if not episode_prefix:
            return ""
        matched = {k: v for k, v in self.clip_subtitles.items() if str(k).startswith(episode_prefix)}
        if not matched:
            return ""
        parts = [f"<{clip_key}>{subtitle}</{clip_key}>" for clip_key, subtitle in sorted(matched.items())]
        return "\n".join(parts)

    def _get_clip_subtitle(self, clip_name: str) -> str:
        if not clip_name:
            return ""
        return str(self.clip_subtitles.get(clip_name, ""))

    def _get_bbox_content(self, vid: str) -> str:
        if not vid:
            return "{}"
        return json.dumps(self.bbox_cache.get(vid, {}))

    def _convert_image_to_data_url(self, image_path: Path) -> Optional[str]:
        if not image_path.is_file():
            return None
        try:
            with Image.open(image_path) as img:
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                encoded = base64.b64encode(buf.getvalue()).decode()
                return f"data:image/jpeg;base64,{encoded}"
        except Exception:
            return None

    def _query_vision(self, query: str, vid: str) -> str:
        frame_nums = list(range(self.frame_start, self.frame_end + 1, self.frame_step))
        messages_content = []
        for fn in frame_nums:
            image_path = Path(self.base_frame_dir, vid, f"{fn:05d}.jpg")
            image_url = self._convert_image_to_data_url(image_path)
            if image_url:
                messages_content.append({"type": "image_url", "image_url": {"url": image_url}})

        if not messages_content:
            return "No visual frames found for the current clip."

        messages_content.append(
            {
                "type": "text",
                "text": (
                    f"Images 1-{len(messages_content)} are sampled frames from one clip.\n"
                    "Focus on key objects, actions, scene transitions and event clues.\n"
                    f"Question: {query}"
                ),
            }
        )

        try:
            resp = self.vision_client.chat.completions.create(
                model=self.vision_model,
                messages=[{"role": "user", "content": messages_content}],
            )
            time.sleep(0.1)
            return resp.choices[0].message.content or ""
        except Exception as exc:
            return f"Vision LLM call failed: {exc}"

    @staticmethod
    def _postprocess_grounding_response(response: str) -> str:
        clip_match = re.search(r"<clip>.*?</clip>", response, re.DOTALL)
        if clip_match:
            return response[: clip_match.end()]
        return response

    @staticmethod
    def _extract_clip_tag(text: str) -> str:
        match = re.search(r"<clip>(.*?)</clip>", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _query_grounding(self, question_data: dict[str, str], sub_block: str, current_vid: str) -> str:
        if not sub_block:
            sub_block = self._build_subtitles_for_episode(self._extract_episode_prefix(current_vid))

        prompt = f"""
Question: {question_data.get("q", "")}
Options:
a0: {question_data.get("a0", "")}
a1: {question_data.get("a1", "")}
a2: {question_data.get("a2", "")}
a3: {question_data.get("a3", "")}
a4: {question_data.get("a4", "")}
Subtitles: {sub_block}

The subtitles are formatted as <clip_label>subtitle_content</clip_label>.
Based on the question and subtitles, locate the most relevant clip label.
{current_vid} may be wrong. Return answer as <clip>clip_label</clip>.
"""
        try:
            raw_response = self.grounding_client.chat.completions.create(
                model=self.grounding_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.grounding_temperature,
                max_tokens=self.grounding_max_tokens,
            ).choices[0].message.content
            cleaned = self._postprocess_grounding_response(raw_response or "")
            return self._extract_clip_tag(cleaned)
        except Exception:
            logger.exception("Grounding API call failed.")
            return ""

    @staticmethod
    def _truncate_to_first_action(response_text: str) -> str:
        for pattern in (
            r"<search>.*?</search>",
            r"<request_grounding>.*?</request_grounding>",
            r"<answer>.*?</answer>",
        ):
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                return response_text[: match.end()]
        return response_text

    @staticmethod
    def _parse_action(response_text: str) -> tuple[str, str]:
        search_match = re.search(r"<search>(.*?)</search>", response_text, re.DOTALL)
        if search_match:
            return "search", search_match.group(1).strip()
        grounding_match = re.search(r"<request_grounding>(.*?)</request_grounding>", response_text, re.DOTALL)
        if grounding_match:
            return "request_grounding", grounding_match.group(1).strip()
        answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
        if answer_match:
            return "answer", answer_match.group(1).strip()
        return "invalid", ""

    @staticmethod
    def _extract_video_id(extra_info: dict[str, Any]) -> str:
        for key in ("predicted_vid", "vid_name", "video_id"):
            value = extra_info.get(key)
            if value:
                return str(value)
        return "default_vid"

    @staticmethod
    def _build_question_data(extra_info: dict[str, Any]) -> dict[str, str]:
        question_data = {"q": str(extra_info.get("original_question", ""))}
        choices = extra_info.get("choices", {})
        if isinstance(choices, dict):
            for i in range(5):
                question_data[f"a{i}"] = str(choices.get(str(i), ""))
        return question_data

    def _execute_action(
        self,
        action_type: str,
        content: str,
        current_vid: str,
        question_data: dict[str, str],
        episode_sub_block: str,
    ) -> tuple[str, bool, str, dict[str, int]]:
        stats = {"valid_action": 0, "is_vision": 0, "is_grounding": 0}

        if action_type == "answer":
            return f"\n<answer>{content}</answer>", True, current_vid, stats

        if action_type == "search":
            stats = {"valid_action": 1, "is_vision": 1, "is_grounding": 0}
            vision_response = self._query_vision(content, current_vid)
            bbox_content = self._get_bbox_content(current_vid)
            observation = (
                f"\n<information>Bounding BOX:\n{bbox_content.strip()}\n"
                f"Visual Description:\n{vision_response.strip()}</information>\n"
            )
            return observation, False, current_vid, stats

        if action_type == "request_grounding":
            stats = {"valid_action": 1, "is_vision": 0, "is_grounding": 1}
            predicted_clip = self._query_grounding(question_data, episode_sub_block, current_vid) or current_vid
            subtitle = self._get_clip_subtitle(predicted_clip)
            observation = f"\n<New_clip>{predicted_clip} + {subtitle}</New_clip>\n"
            return observation, False, predicted_clip, stats

        return (
            "\nMy action is not correct. I need to search, request grounding, or answer.\n",
            False,
            current_vid,
            stats,
        )

    def _append_tokens(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        response_mask: list[int],
        token_ids: list[int],
        mask_value: int,
    ) -> bool:
        remain = self.response_length - len(response_mask)
        if remain <= 0:
            return False
        clipped = token_ids[:remain]
        if not clipped:
            return False
        prompt_ids.extend(clipped)
        response_ids.extend(clipped)
        response_mask.extend([mask_value] * len(clipped))
        return True

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info", {}) or {}
        if not isinstance(extra_info, dict):
            extra_info = {}

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        prompt_ids = await self.apply_chat_template(messages, images=images, videos=videos)

        response_ids: list[int] = []
        response_mask: list[int] = []
        metrics: dict[str, Any] = {}
        request_id = uuid4().hex

        current_vid = self._extract_video_id(extra_info)
        question_data = self._build_question_data(extra_info)
        episode_sub_block = str(extra_info.get("episode_sub_block", ""))

        assistant_turns = 0
        user_turns = 0
        valid_action_count = 0
        vision_count = 0
        grounding_count = 0
        action_history: list[str] = []

        while assistant_turns < self.max_assistant_turns and len(response_mask) < self.response_length:
            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=images,
                    video_data=videos,
                )

            if metrics.get("num_preempted") is None:
                metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
            else:
                metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

            decoded = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
            processed_text = self._truncate_to_first_action(decoded).strip()
            if not processed_text:
                processed_text = decoded.strip()
            if not processed_text:
                break

            assistant_ids = self.tokenizer.encode(processed_text, add_special_tokens=False)
            if not assistant_ids:
                assistant_ids = list(output.token_ids)
            if not self._append_tokens(prompt_ids, response_ids, response_mask, assistant_ids, mask_value=1):
                break

            assistant_turns += 1
            action_type, action_content = self._parse_action(processed_text)
            action_history.append(action_type)
            if action_type == "answer":
                break

            if user_turns >= self.max_user_turns:
                break

            with simple_timer("tool_calls", metrics):
                observation, should_stop, next_vid, action_stats = await self.loop.run_in_executor(
                    None,
                    lambda: self._execute_action(
                        action_type=action_type,
                        content=action_content,
                        current_vid=current_vid,
                        question_data=question_data,
                        episode_sub_block=episode_sub_block,
                    ),
                )

            current_vid = next_vid
            valid_action_count += int(action_stats["valid_action"])
            vision_count += int(action_stats["is_vision"])
            grounding_count += int(action_stats["is_grounding"])

            if should_stop:
                break

            observation_ids = self.tokenizer.encode(observation, add_special_tokens=False)
            if self.max_obs_length > 0:
                observation_ids = observation_ids[: self.max_obs_length]
            if self.max_tool_response_length > 0:
                observation_ids = observation_ids[: self.max_tool_response_length]

            if observation_ids and not self._append_tokens(
                prompt_ids, response_ids, response_mask, observation_ids, mask_value=0
            ):
                break
            user_turns += 1

        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = -1

        response_ids = response_ids[: self.response_length]
        response_mask = response_mask[: self.response_length]

        prompt_only_len = max(0, len(prompt_ids) - len(response_mask))
        output = AgentLoopOutput(
            prompt_ids=prompt_ids[:prompt_only_len],
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=None,
            routed_experts=None,
            multi_modal_data=multi_modal_data,
            num_turns=assistant_turns + user_turns + 1,
            metrics=metrics,
            extra_fields={
                "turn_scores": [],
                "tool_rewards": [],
                "predicted_vid": current_vid,
                "valid_action_count": valid_action_count,
                "vision_count": vision_count,
                "grounding_count": grounding_count,
                "tool_used": int((vision_count + grounding_count) > 0),
                "action_history": action_history,
            },
        )
        return output

