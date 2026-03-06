#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# =====================================
# Unified Local Eval Config (Edit Here)
# =====================================
# Required
DATASET="tvqa_plus"                         # tvqa | tvqa_plus
LLM_PATH="/path/to/your/local_llm"         # local model path for vLLM

# Core eval args
MAX_TURN=5
GPU_MEMORY_UTILIZATION=0.4

# API agents used by local pipeline
GROUNDING_MODEL="grok-4-fast-reasoning"
VISION_MODEL="gpt-4o"
GROUNDING_BASE_URL="https://api2.aigcbest.top/v1"
VISION_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
GROUNDING_API_KEY=""                        # fill your key
VISION_API_KEY=""                           # fill your key

# Optional grounding cache
GROUNDING_CACHE_JSON_PATH="../Tvqa_data/grounding_pairs_6000_samples.json"

# Optional explicit path overrides (leave empty to use dataset defaults)
QUESTIONS_PATH=""
SUBS_PATH=""
BASE_FRAME_DIR=""
BBOX_JSON_PATH=""
OUTPUT_FILENAME=""
DETAILED_OUTPUT_FILENAME=""

# =========================
# Dataset-specific defaults
# =========================
if [[ "$DATASET" == "tvqa" ]]; then
  DEFAULT_QUESTIONS_PATH="../Tvqa/house_met/tvqa_val_house_met.jsonl"
  DEFAULT_SUBS_PATH="../Tvqa/tvqa_subtitles.json"
  DEFAULT_BASE_FRAME_DIR="../Tvqa/house_met_frames"
  DEFAULT_BBOX_JSON_PATH=""
  DEFAULT_OUTPUT_FILENAME="./eval_tvqa_grok-4-fast-reasoning.json"
  DEFAULT_DETAILED_OUTPUT_FILENAME="./data_log_eval_tvqa_grok-4-fast-reasoning.json"
elif [[ "$DATASET" == "tvqa_plus" ]]; then
  DEFAULT_QUESTIONS_PATH="../Tvqa_data/tvqa_plus_val.json"
  DEFAULT_SUBS_PATH="../Tvqa_data/all_episodes_subtitles_by_clips.json"
  DEFAULT_BASE_FRAME_DIR="../Tvqa_data/bbt_frames"
  DEFAULT_BBOX_JSON_PATH="../Tvqa_data/clip_bbox_mapping.json"
  DEFAULT_OUTPUT_FILENAME="./eval_tvqa_plus_grok-4-fast-reasoning.json"
  DEFAULT_DETAILED_OUTPUT_FILENAME="./data_log_eval_tvqa_plus_grok-4-fast-reasoning.json"
else
  echo "[ERROR] DATASET must be 'tvqa' or 'tvqa_plus', got: $DATASET" >&2
  exit 1
fi

if [[ -z "$LLM_PATH" || "$LLM_PATH" == "/path/to/your/local_llm" ]]; then
  echo "[ERROR] Please set LLM_PATH at the top of scripts/eval_unified_local.sh" >&2
  exit 1
fi

if [[ -z "$GROUNDING_API_KEY" || -z "$VISION_API_KEY" ]]; then
  echo "[ERROR] Please set GROUNDING_API_KEY and VISION_API_KEY at the top of scripts/eval_unified_local.sh" >&2
  exit 1
fi

# Apply optional overrides
if [[ -z "$QUESTIONS_PATH" ]]; then QUESTIONS_PATH="$DEFAULT_QUESTIONS_PATH"; fi
if [[ -z "$SUBS_PATH" ]]; then SUBS_PATH="$DEFAULT_SUBS_PATH"; fi
if [[ -z "$BASE_FRAME_DIR" ]]; then BASE_FRAME_DIR="$DEFAULT_BASE_FRAME_DIR"; fi
if [[ -z "$BBOX_JSON_PATH" ]]; then BBOX_JSON_PATH="$DEFAULT_BBOX_JSON_PATH"; fi
if [[ -z "$OUTPUT_FILENAME" ]]; then OUTPUT_FILENAME="$DEFAULT_OUTPUT_FILENAME"; fi
if [[ -z "$DETAILED_OUTPUT_FILENAME" ]]; then DETAILED_OUTPUT_FILENAME="$DEFAULT_DETAILED_OUTPUT_FILENAME"; fi

mkdir -p "$(dirname "$OUTPUT_FILENAME")" "$(dirname "$DETAILED_OUTPUT_FILENAME")"

echo "=== Unified Local Evaluation ==="
echo "dataset: $DATASET"
echo "llm_path: $LLM_PATH"
echo "max_turn: $MAX_TURN"
echo "gpu_memory_utilization: $GPU_MEMORY_UTILIZATION"
echo "questions: $QUESTIONS_PATH"
echo "subtitles: $SUBS_PATH"
echo "frames: $BASE_FRAME_DIR"
echo "bbox: ${BBOX_JSON_PATH:-<none>}"
echo "summary_output: $OUTPUT_FILENAME"
echo "detail_output: $DETAILED_OUTPUT_FILENAME"
echo "grounding_model: $GROUNDING_MODEL"
echo "vision_model: $VISION_MODEL"

cmd=(
  python src/evaluation/lvagent/evaluate_local_unified.py
  --dataset "$DATASET"
  --llm-path "$LLM_PATH"
  --max_turn "$MAX_TURN"
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
  --questions-path "$QUESTIONS_PATH"
  --subs-path "$SUBS_PATH"
  --base-frame-dir "$BASE_FRAME_DIR"
  --output-filename "$OUTPUT_FILENAME"
  --detailed-output-filename "$DETAILED_OUTPUT_FILENAME"
  --grounding-model "$GROUNDING_MODEL"
  --vision-model "$VISION_MODEL"
  --grounding-base-url "$GROUNDING_BASE_URL"
  --vision-base-url "$VISION_BASE_URL"
  --grounding-api-key "$GROUNDING_API_KEY"
  --vision-api-key "$VISION_API_KEY"
  --grounding-cache-json-path "$GROUNDING_CACHE_JSON_PATH"
)

if [[ -n "$BBOX_JSON_PATH" ]]; then
  cmd+=(--bbox-json-path "$BBOX_JSON_PATH")
fi

"${cmd[@]}"

echo "=== Done ==="
echo "Summary JSON: $OUTPUT_FILENAME"
echo "Detailed JSON: $DETAILED_OUTPUT_FILENAME"
