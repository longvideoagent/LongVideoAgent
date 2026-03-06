import argparse
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from difflib import SequenceMatcher

# 日志配置
def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """配置日志系统，支持控制台和文件输出"""
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers
    )
    return logging.getLogger(__name__)

def load_json(file_path: Path) -> Any:
    """加载JSON文件，检查文件存在并处理格式错误"""
    if not file_path.exists():
        raise ValueError(f"文件不存在: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"无效JSON格式: {file_path}, 错误: {e}")

def normalize_string(s: str) -> str:
    """规范化字符串：去除多余空格、转换为小写"""
    return ' '.join(s.strip().lower().split())

def extract_option_from_answer(final_answer: str) -> Optional[str]:
    """从final_answer提取选项，如'a4'"""
    if not final_answer:
        return None
    match = re.match(r'^a(\d):', final_answer.strip(), re.IGNORECASE)
    if match:
        return f"a{match.group(1)}"
    match = re.search(r'a(\d)', final_answer, re.IGNORECASE)
    if match:
        return f"a{match.group(1)}"
    return None

def fuzzy_match_content(norm_pred: str, entry: Dict, similarity_threshold: float) -> Optional[str]:
    """模糊匹配final_answer到a0-a4选项"""
    if not norm_pred:
        return None
    for i in range(5):
        option_key = f"a{i}"
        if option_key in entry:
            norm_option = normalize_string(entry[option_key])
            if norm_pred == norm_option or norm_pred in norm_option or norm_option in norm_pred:
                return f"a{i}"
            similarity = SequenceMatcher(None, norm_pred, norm_option).ratio()
            if similarity > similarity_threshold:
                return f"a{i}"
    return None

def get_gt_answer(data_entry: Dict) -> Optional[str]:
    """从数据集entry获取ground truth选项"""
    answer_idx = data_entry.get('answer_idx')
    if answer_idx is not None:
        try:
            idx = int(answer_idx)
            if 0 <= idx <= 4:
                return f"a{idx}"
        except ValueError:
            pass
    return None

def match_question_in_dataset(question: str, dataset: List[Dict]) -> Optional[Dict]:
    """在数据集JSON中匹配问题"""
    if not question:
        return None
    norm_q = normalize_string(question)
    for entry in dataset:
        if normalize_string(entry.get('q', '')) == norm_q:
            return entry
    return None

def flatten_history_vids(history_vids: Any) -> List[str]:
    """扁平化history_vids，提取所有视频ID"""
    if not history_vids:
        return []
    flattened = []
    if isinstance(history_vids, dict):
        for vids in history_vids.values():
            if isinstance(vids, list):
                flattened.extend(vids)
            else:
                flattened.append(vids)
    elif isinstance(history_vids, list):
        flattened = [vid for sublist in history_vids for vid in (sublist if isinstance(sublist, list) else [sublist])]
    return flattened

@dataclass
class EvalStats:
    """评测统计数据类"""
    turns_1_correct: int = 0
    turns_1_total: int = 0
    turns_1_grounding_correct: int = 0
    turns_gt_1_correct: int = 0
    turns_gt_1_total: int = 0
    turns_gt_1_grounding_correct: int = 0
    other_turns_correct: int = 0
    other_turns_total: int = 0
    other_grounding_correct: int = 0
    unmatched_questions: int = 0
    invalid_predictions: int = 0
    num_turns_distribution: Counter = Counter()

def increment_stats(
    stats: EvalStats,
    num_turns: int,
    is_correct: bool,
    is_grounding_correct: bool
) -> None:
    """根据num_turns更新统计"""
    stats.num_turns_distribution[num_turns] += 1
    if num_turns == 1:
        stats.turns_1_total += 1
        if is_correct:
            stats.turns_1_correct += 1
        if is_grounding_correct:
            stats.turns_1_grounding_correct += 1
    elif num_turns > 1:
        stats.turns_gt_1_total += 1
        if is_correct:
            stats.turns_gt_1_correct += 1
        if is_grounding_correct:
            stats.turns_gt_1_grounding_correct += 1
    else:
        stats.other_turns_total += 1
        if is_correct:
            stats.other_turns_correct += 1
        if is_grounding_correct:
            stats.other_grounding_correct += 1

def calculate_accuracy_by_turns(
    model_data: Dict,
    dataset: List[Dict],
    similarity_threshold: float,
    logger: logging.Logger
) -> Dict[str, Any]:
    """计算按num_turns分组的准确率和grounding正确率"""
    results = model_data.get("results", [])
    total_questions = len(results)
    expected_total = 300  # TVQA验证集标准总数

    stats = EvalStats()

    for item in results:
        question = item.get("question", "")
        final_answer = item.get("final_answer", "")
        num_turns = item.get("num_turns", 0)
        history_vids = item.get("history_vids", {})

        matched_entry = match_question_in_dataset(question, dataset)
        if not matched_entry:
            stats.unmatched_questions += 1
            logger.warning(f"未匹配问题: {question}")
            increment_stats(stats, num_turns, False, False)
            continue

        gt_option = get_gt_answer(matched_entry)
        if gt_option is None:
            logger.warning(f"无效GT for: {question}")
            increment_stats(stats, num_turns, False, False)
            continue

        true_vid = matched_entry.get('vid_name')
        if true_vid is None:
            logger.warning(f"无真实vid_name for: {question}")
            increment_stats(stats, num_turns, False, False)
            continue

        history_vids_flat = flatten_history_vids(history_vids)
        is_grounding_correct = true_vid in history_vids_flat

        pred_option = extract_option_from_answer(final_answer)
        if pred_option is None:
            norm_pred = normalize_string(final_answer)
            pred_option = fuzzy_match_content(norm_pred, matched_entry, similarity_threshold)

        if pred_option is None:
            stats.invalid_predictions += 1
            logger.warning(f"无效预测 for: {question} (final_answer: {final_answer})")
            increment_stats(stats, num_turns, False, is_grounding_correct)
            continue

        is_correct = pred_option.lower() == gt_option.lower()
        if not is_correct:
            logger.info(f"错误: {question} | Pred: {pred_option} | GT: {gt_option}")

        increment_stats(stats, num_turns, is_correct, is_grounding_correct)

    # 计算准确率
    turns_1_accuracy = stats.turns_1_correct / stats.turns_1_total if stats.turns_1_total > 0 else 0.0
    turns_gt_1_accuracy = stats.turns_gt_1_correct / stats.turns_gt_1_total if stats.turns_gt_1_total > 0 else 0.0
    other_turns_accuracy = stats.other_turns_correct / stats.other_turns_total if stats.other_turns_total > 0 else 0.0
    turns_1_grounding_accuracy = stats.turns_1_grounding_correct / stats.turns_1_total if stats.turns_1_total > 0 else 0.0
    turns_gt_1_grounding_accuracy = stats.turns_gt_1_grounding_correct / stats.turns_gt_1_total if stats.turns_gt_1_total > 0 else 0.0
    other_grounding_accuracy = stats.other_grounding_correct / stats.other_turns_total if stats.other_turns_total > 0 else 0.0

    overall_total = stats.turns_1_total + stats.turns_gt_1_total + stats.other_turns_total
    overall_correct = stats.turns_1_correct + stats.turns_gt_1_correct + stats.other_turns_correct
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
    overall_grounding_correct = stats.turns_1_grounding_correct + stats.turns_gt_1_grounding_correct + stats.other_grounding_correct
    overall_grounding_accuracy = overall_grounding_correct / overall_total if overall_total > 0 else 0.0

    if overall_total != expected_total:
        logger.warning(f"统计总数 {overall_total} 不等于预期 {expected_total}")

    return {
        "turns_1_accuracy": turns_1_accuracy,
        "turns_1_correct": stats.turns_1_correct,
        "turns_1_total": stats.turns_1_total,
        "turns_gt_1_accuracy": turns_gt_1_accuracy,
        "turns_gt_1_correct": stats.turns_gt_1_correct,
        "turns_gt_1_total": stats.turns_gt_1_total,
        "other_turns_accuracy": other_turns_accuracy,
        "other_turns_correct": stats.other_turns_correct,
        "other_turns_total": stats.other_turns_total,
        "turns_1_grounding_accuracy": turns_1_grounding_accuracy,
        "turns_1_grounding_correct": stats.turns_1_grounding_correct,
        "turns_gt_1_grounding_accuracy": turns_gt_1_grounding_accuracy,
        "turns_gt_1_grounding_correct": stats.turns_gt_1_grounding_correct,
        "other_grounding_accuracy": other_grounding_accuracy,
        "other_grounding_correct": stats.other_grounding_correct,
        "overall_accuracy": overall_accuracy,
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "overall_grounding_accuracy": overall_grounding_accuracy,
        "overall_grounding_correct": overall_grounding_correct,
        "unmatched_questions": stats.unmatched_questions,
        "invalid_predictions": stats.invalid_predictions,
        "num_turns_distribution": dict(stats.num_turns_distribution),
        "total_questions": total_questions
    }

def print_stats(stats: Dict[str, Any], logger: logging.Logger) -> None:
    """打印评测统计结果"""
    logger.info(f"总问题数: {stats['total_questions']}")
    logger.info(f"num_turns 分布: {stats['num_turns_distribution']}")
    logger.info(f"未匹配问题数: {stats['unmatched_questions']}")
    logger.info(f"无效预测数: {stats['invalid_predictions']}")
    logger.info(
        f"num_turns=1 准确率: {stats['turns_1_accuracy']:.4f} "
        f"({stats['turns_1_correct']}/{stats['turns_1_total']})"
    )
    logger.info(
        f"num_turns>1 准确率: {stats['turns_gt_1_accuracy']:.4f} "
        f"({stats['turns_gt_1_correct']}/{stats['turns_gt_1_total']})"
    )
    logger.info(
        f"其他 num_turns (如0或负数) 准确率: {stats['other_turns_accuracy']:.4f} "
        f"({stats['other_turns_correct']}/{stats['other_turns_total']})"
    )
    logger.info(
        f"num_turns=1 grounding正确率: {stats['turns_1_grounding_accuracy']:.4f} "
        f"({stats['turns_1_grounding_correct']}/{stats['turns_1_total']})"
    )
    logger.info(
        f"num_turns>1 grounding正确率: {stats['turns_gt_1_grounding_accuracy']:.4f} "
        f"({stats['turns_gt_1_grounding_correct']}/{stats['turns_gt_1_total']})"
    )
    logger.info(
        f"其他 num_turns grounding正确率: {stats['other_grounding_accuracy']:.4f} "
        f"({stats['other_grounding_correct']}/{stats['other_turns_total']})"
    )
    logger.info(
        f"总体准确率: {stats['overall_accuracy']:.4f} "
        f"({stats['overall_correct']}/{stats['overall_total']})"
    )
    logger.info(
        f"总体grounding正确率: {stats['overall_grounding_accuracy']:.4f} "
        f"({stats['overall_grounding_correct']}/{stats['overall_total']})"
    )
    logger.info(f"总计问题数: {stats['overall_total']}")

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="TVQA/TVQA+ 准确率和grounding评测脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model-output", type=Path, required=True, help="模型输出JSON路径")
    parser.add_argument("--dataset", type=Path, required=True, help="数据集JSON路径 (e.g., tvqa_val.json)")
    parser.add_argument("--output", type=Path, help="输出JSON结果文件 (可选)")
    parser.add_argument("--similarity-threshold", type=float, default=0.9, help="模糊匹配相似度阈值")
    parser.add_argument("--verbose", action="store_true", help="启用详细日志")
    parser.add_argument("--log-file", type=str, help="日志文件路径 (可选)")
    return parser.parse_args()

def main() -> None:
    """主函数：加载数据、计算准确率、打印/保存结果"""
    args = parse_args()
    logger = setup_logging(args.verbose, args.log_file)

    logger.info(f"加载模型输出: {args.model_output}")
    model_data = load_json(args.model_output)
    logger.info(f"加载数据集: {args.dataset}")
    dataset = load_json(args.dataset)

    logger.info(f"模型输出问题数: {len(model_data.get('results', []))}")
    logger.info(f"数据集问题数: {len(dataset)}")

    stats = calculate_accuracy_by_turns(model_data, dataset, args.similarity_threshold, logger)
    print_stats(stats, logger)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"结果保存至: {args.output}")

if __name__ == "__main__":
    main()