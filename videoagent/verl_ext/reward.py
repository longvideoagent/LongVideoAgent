import re
from typing import Any


TVQA_SOURCE = "tvqa_plus_vision"
QA_SOURCES = {
    "nq",
    "triviaqa",
    "popqa",
    "hotpotqa",
    "2wikimultihopqa",
    "musique",
    "bamboogle",
}


def compute_tvqa_score(solution_str: str, ground_truth: str, one_turn: bool = True) -> tuple[float, float]:
    """Keep LongVideoAgent TVQA+ reward semantics unchanged."""
    response = solution_str or ""
    total_score = 0.0
    second_score = 0.0

    if re.search(r"<reasoning>.*?</reasoning>", response, re.DOTALL):
        total_score += 0.2

    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if not answer_match:
        return total_score, second_score

    total_score += 0.2
    predicted_answer = answer_match.group(1).strip()

    predicted_num_match = re.search(r"(\d+)", predicted_answer)
    ground_truth_num_match = re.search(r"(\d+)", ground_truth or "")

    if predicted_num_match and ground_truth_num_match:
        if predicted_num_match.group(1) == ground_truth_num_match.group(1):
            total_score += 1.0

    if not one_turn:
        if re.search(r"<search>", response) or re.search(r"<request_grounding>", response):
            second_score += 1.0

    return total_score, second_score


def _normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _qa_em_score(solution_str: str, ground_truth: str) -> float:
    pred = _normalize_text(solution_str)
    gold = _normalize_text(ground_truth)
    if not pred or not gold:
        return 0.0
    if pred == gold:
        return 1.0
    return 1.0 if gold in pred else 0.0


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict[str, Any] = None, **kwargs):
    """Custom reward entrypoint for verl_new reward loop."""
    del kwargs
    extra_info = extra_info or {}

    if data_source == TVQA_SOURCE:
        num_turns = extra_info.get("num_turns")
        one_turn = bool(extra_info.get("one_turn", False))
        try:
            if num_turns is not None:
                one_turn = int(num_turns) <= 2
        except Exception:
            pass

        main_score, second_score = compute_tvqa_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            one_turn=one_turn,
        )
        acc = 1.0 if main_score >= 1.4 else 0.0
        return {
            "score": float(main_score),
            "acc": float(acc),
            "main_score": float(main_score),
            "second_score": float(second_score),
            "tool_use_bonus": float(second_score),
            "num_turns": num_turns,
            "predicted_vid": extra_info.get("predicted_vid", ""),
            "valid_action_count": extra_info.get("valid_action_count", 0),
            "vision_count": extra_info.get("vision_count", 0),
            "grounding_count": extra_info.get("grounding_count", 0),
        }

    if data_source in QA_SOURCES:
        em = _qa_em_score(solution_str=solution_str, ground_truth=ground_truth)
        return {"score": float(em), "acc": float(em)}

    return {"score": 0.0, "acc": 0.0}

