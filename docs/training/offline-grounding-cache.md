# Offline Grounding Cache

For training data preprocessing, run a standalone offline grounding pass to generate an initial occur-clip cache (question -> clip), then inject this cache into your training prompt pipeline.

We strongly recommend this step because high-quality initial subtitles are important for downstream reasoning. If no cache is provided, the pipeline may fall back to randomly selecting one clip's subtitles, which makes later Master Agent reasoning much harder.

## Script

- `src/dataset/build_grounding_cache.py`

## Usage

```bash
python src/dataset/build_grounding_cache.py \
  --dataset tvqa_plus \
  --questions-path /path/to/train.json \
  --subs-path /path/to/all_episodes_subtitles_by_clips.json \
  --grounding-model "grok-4-fast-reasoning" \
  --grounding-base-url "https://api2.aigcbest.top/v1" \
  --output-dir /path/to/cache_dir \
  --threads 8
```

## Optional Arguments

- `--grounding-api-key`: if omitted, reads env `qdd_api`.
- `--output-filename`: custom cache filename; default is `grounding_cache_{dataset}_{model}.json`.
- `--max-samples`: run on first N samples for smoke test.
- `--overwrite`: force regenerate all entries.

## Output Format

The cache is a JSON object keyed by sample index. Each entry includes at least:

- `question`: question text
- `clip`: predicted occur clip label

This format is compatible with the cache lookup used in:

- `src/evaluation/lvagent/evaluate_api_unified.py` via `--grounding-cache-json-path`
