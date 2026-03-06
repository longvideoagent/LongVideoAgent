# Evaluation

## Local Agent Evaluation (TVQA+)

This section covers local evaluation for the full LongVideoAgent pipeline on TVQA+, using a local main model (vLLM) and API-based grounding/vision agents.

### Script

- `src/evaluation/lvagent/tvqa_plus/evaluate_local.py`

### Usage

```bash
python src/evaluation/lvagent/tvqa_plus/evaluate_local.py \
  --checkpoint_step 100 \
  --max_turn 5 \
  --gpu_memory_utilization 0.4 \
  --model-dir-template "../videoagent-step-{checkpoint_step}" \
  --questions-path "../Tvqa_data/tvqa_plus_val_sampled_300.json" \
  --subs-path "../Tvqa_data/all_episodes_subtitles_by_clips.json" \
  --base-frame-dir "../Tvqa_data/bbt_frames" \
  --output-filename-template "./eval_videoagent-{checkpoint_step}.json" \
  --grounding-model "grok-4-fast-reasoning" \
  --vision-model "grok-4-fast-reasoning" \
  --grounding-base-url "https://api2.aigcbest.top/v1" \
  --vision-base-url "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

### Key Arguments

- `--model-dir-template`: Main model path template. Must contain `{checkpoint_step}`.
- `--questions-path`: TVQA+ questions JSON path.
- `--subs-path`: subtitles JSON path.
- `--base-frame-dir`: frame root directory.
- `--output-filename-template`: output JSON template. Must contain `{checkpoint_step}`.
- `--grounding-model`: API model name used by the grounding agent.
- `--vision-model`: API model name used by the vision agent.
- `--grounding-base-url`: grounding API endpoint.
- `--vision-base-url`: vision API endpoint.
- `--grounding-api-key`: optional; if omitted, reads env `qdd_api`.
- `--vision-api-key`: optional; if omitted, reads env `aliyun_api`.

### Notes

- This refactor keeps evaluation logic and prompts unchanged, and only improves configuration/CLI ergonomics.
- Input file and directory validation is performed at startup.

## Local Agent Evaluation (TVQA / LongTVQA)

This section covers local evaluation for TVQA with the LongTVQA dataset format from Hugging Face.

### Script

- `src/evaluation/lvagent/tvqa/evaluate_local.py`

### Usage

```bash
python src/evaluation/lvagent/tvqa/evaluate_local.py \
  --checkpoint_step 100 \
  --max_turn 5 \
  --gpu_memory_utilization 0.4 \
  --model-dir-template "../tvqa_qwen_3b-step-{checkpoint_step}" \
  --questions-path "../Tvqa_data/LongTVQA_val.jsonl" \
  --subs-path "../Tvqa_data/LongTVQA_subtitles_clip_level.jsonl" \
  --base-frame-dir "../Tvqa_data/frames" \
  --output-filename-template "./eval_tvqa-{checkpoint_step}.json" \
  --grounding-model "grok-4-fast-reasoning" \
  --vision-model "grok-4-fast-reasoning" \
  --grounding-base-url "https://api2.aigcbest.top/v1" \
  --vision-base-url "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

### Notes

- The evaluator keeps the original reasoning/prompt logic unchanged and only adds data-format compatibility.
- LongTVQA field mapping is handled automatically: `occur_clip -> vid_name`, `answer -> answer_idx`.
- Both JSON array and line-delimited JSONL inputs are supported for question files.

## Dataset Download & Preparation (LongTVQA / LongTVQA+)

Use the one-shot script below to download both Hugging Face datasets and prepare evaluator-ready files:

```bash
bash scripts/download_and_prepare_longtvqa.sh
```

Default outputs are generated under `../Tvqa_data`:

- `LongTVQA_val_normalized.jsonl`
- `LongTVQA_plus_val_normalized.json`
- `clip_bbox_mapping.json`

Notes:

- Downloader uses `hf` if available, otherwise falls back to `huggingface-cli`.
- You can override locations with env vars:
  - `DATA_ROOT` (raw downloaded dataset dir, default `../Tvqa_data/hf_datasets`)
  - `OUTPUT_DIR` (prepared output dir, default `../Tvqa_data`)

## Unified API Evaluation (TVQA / TVQA+)

This section covers a unified API evaluator that supports both datasets with one script.

### Script

- `src/evaluation/lvagent/evaluate_api_unified.py`

### Usage

Run on TVQA:

```bash
python src/evaluation/lvagent/evaluate_api_unified.py \
  --dataset tvqa \
  --checkpoint_step api \
  --max_turn 5 \
  --threads 30
```

Run on TVQA+:

```bash
python src/evaluation/lvagent/evaluate_api_unified.py \
  --dataset tvqa_plus \
  --checkpoint_step api \
  --max_turn 5 \
  --threads 30
```

### Key Arguments

- `--dataset`: `tvqa` or `tvqa_plus` (required).
- `--questions-path`: optional override for questions file path; dataset-specific default is used if omitted.
- `--subs-path`: optional override for subtitles file path; dataset-specific default is used if omitted.
- `--base-frame-dir`: optional override for frame root directory; dataset-specific default is used if omitted.
- `--bbox-json-path`: bbox JSON path (used for `tvqa_plus`; ignored by `tvqa`).
- `--output-filename`: summary output JSON filename.
- `--detailed-output-filename`: detailed conversation log JSON filename.
- `--grounding-model`, `--vision-model`, `--main-model`: model names for grounding/vision/main API calls.
- `--grounding-base-url`, `--vision-base-url`, `--main-base-url`: API endpoint base URLs.
- `--grounding-api-key`, `--vision-api-key`, `--main-api-key`: API keys (defaults: `qdd_api`, `aliyun_api`, `qdd_api` env vars).

### Notes

- The unified script keeps one evaluation pipeline and switches only dataset-specific input handling (question format, subtitle keying, frame/bbox access).
- Core multi-turn evaluation logic remains shared across TVQA and TVQA+.

## Unified Local Evaluation (TVQA / TVQA+)

This section covers a unified local evaluator that supports both datasets with one script, using local vLLM as the master agent.

### Script

- `src/evaluation/lvagent/evaluate_local_unified.py`

### Usage

Run on TVQA:

```bash
python src/evaluation/lvagent/evaluate_local_unified.py \
  --dataset tvqa \
  --llm-path "/path/to/your/local_llm" \
  --max_turn 5 \
  --gpu_memory_utilization 0.4
```

Full example (recommended for reproducible runs):

```bash
python src/evaluation/lvagent/evaluate_local_unified.py \
  --dataset tvqa_plus \
  --llm-path "/models/videoagent-step-100" \
  --max_turn 5 \
  --gpu_memory_utilization 0.4 \
  --questions-path "../Tvqa_data/tvqa_plus_val.json" \
  --subs-path "../Tvqa_data/all_episodes_subtitles_by_clips.json" \
  --base-frame-dir "../Tvqa_data/bbt_frames" \
  --bbox-json-path "../Tvqa_data/clip_bbox_mapping.json" \
  --output-filename "./results/tvqa_plus_step100_summary.json" \
  --detailed-output-filename "./results/tvqa_plus_step100_detail.json" \
  --grounding-model "grok-4-fast-reasoning" \
  --vision-model "gpt-4o" \
  --grounding-base-url "https://api2.aigcbest.top/v1" \
  --vision-base-url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
  --grounding-api-key "$qdd_api" \
  --vision-api-key "$aliyun_api" \
  --grounding-cache-json-path "../Tvqa_data/grounding_pairs_6000_samples.json"
```

Run on TVQA+:

```bash
python src/evaluation/lvagent/evaluate_local_unified.py \
  --dataset tvqa_plus \
  --llm-path "/path/to/your/local_llm" \
  --max_turn 5 \
  --gpu_memory_utilization 0.4
```

### Key Arguments

- `--dataset`: `tvqa` or `tvqa_plus` (required).
- `--llm-path`: local model directory path used by vLLM (required).
- `--questions-path`: optional override for questions file path; dataset-specific default is used if omitted.
- `--subs-path`: optional override for subtitles file path; dataset-specific default is used if omitted.
- `--base-frame-dir`: optional override for frame root directory; dataset-specific default is used if omitted.
- `--bbox-json-path`: bbox JSON path (used for `tvqa_plus`; ignored by `tvqa`).
- `--output-filename`: summary output JSON filename.
- `--detailed-output-filename`: detailed conversation log JSON filename.
- `--grounding-model`, `--vision-model`: model names for grounding/vision API calls.
- `--grounding-base-url`, `--vision-base-url`: API endpoint base URLs.
- `--grounding-api-key`, `--vision-api-key`: API keys (defaults: `qdd_api`, `aliyun_api` env vars).
- `--grounding-cache-json-path`: optional grounding cache JSON; unmatched questions fall back to grounding API.

## Baseline Evaluation

This section outlines the baseline evaluation script for the VideoQAgent project, located in `src/evaluation/baseline`. The script evaluates a model's performance on the TVQA dataset (or variants like TVQA+) using an API-based model for question-answering tasks.

### Overview

The `evaluate_api_tvqa.py` script processes video frames, subtitles, and questions to generate answers via an API-based model (e.g., Gemini). It supports input validation, multi-threaded processing, and result output. Dataset variants are handled by providing appropriate input files.

### Directory Structure

- `src/evaluation/baseline/`
  - `evaluate_api_tvqa.py`: Runs evaluation on TVQA or its variants using an API-based model.
  - `utils.py`: Contains utility functions, including `run_simple_qa`, for evaluation logic.

### Script

#### `evaluate_api_tvqa.py`

Evaluates a model on the TVQA dataset (or variants like TVQA+) by calling an API. The model name corresponds to the API provider's model (e.g., `gemini-2.5-pro-exp-03-25`).

**Usage**:

```bash
python src/evaluation/baseline/evaluate_api_tvqa.py \
  --questions-path ./tvqa_question.json \
  --subs-path ./substitle.json \
  --output-filename ./eval_gemini_tvqa.json \
  --base-frame-dir ../bbt_frames \
  --model gemini-2.5-pro-exp-03-25 \
  --threads 10
```

**Arguments**:

- `--questions-path`: Path to the TVQA questions JSON file (required).
- `--subs-path`: Path to the subtitles JSON file (required).
- `--output-filename`: Path for saving results (required).
- `--base-frame-dir`: Directory with video frame images (required).
- `--model`: API provider's model name (required).
- `--threads`: Number of threads for parallel processing (required; adjust based on API rate limits).

**Behavior**:

- Validates input paths (questions, subtitles, frames).
- Calls `run_simple_qa` from `utils.py` to perform evaluation via API.
- Uses multi-threaded API calls, then reorders results back to the original question index before computing accuracy.
- Saves results to the specified output file.
- Supports TVQA variants by adjusting the `--questions-path` input.

#### `utils.py`

Includes `run_simple_qa`, which handles API calls for question answering. It preserves question-answer alignment under multi-threading by sorting completed futures with their original indices before metric calculation. Configure the API base URL (e.g., `base_url`) in `utils.py` to match the API provider's endpoint.

### Prerequisites

- Ensure input files (questions JSON, subtitles JSON, frame directory) exist.
- Configure the API base URL in `utils.py` for the chosen model.
- The specified model must be supported by the API provider.
- Install Python dependencies (e.g., `argparse`, `os`).
- Set `--threads` based on the API provider's rate limits to avoid overloading.

### Notes

- Update the API base URL in `utils.py` to match your provider's endpoint.
- Use a thread count suitable for the API's maximum load to optimize performance.
- Ensure the output directory is writable for result storage.

For TVQA+ or other variants, provide the appropriate questions JSON file via `--questions-path`.
