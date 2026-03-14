# Convert to Parquet

Use this script to build training parquet files (`train.parquet`, `val.parquet`) directly from LongTVQA / LongTVQA+ question files, with initial clip localization taken from offline grounding cache (question-text matching).

## Script

- `src/dataset/convert_tvqa_json_to_grpo_parquet.py`

## Usage

```bash
python src/dataset/convert_tvqa_json_to_grpo_parquet.py \
  --questions-path /path/to/LongTVQA_or_LongTVQA_plus_questions.jsonl_or_json \
  [--grounding-cache-json /path/to/grounding_cache_tvqa_plus_xxx.json] \
  --subtitles-dir /path/to/subtitles_dir \
  --output-dir /path/to/output \
  --seed 42
```

## Notes

- `--questions-path` supports both JSONL (for example, LongTVQA) and JSON (for example, LongTVQA+).
- `--subtitles-dir` must contain:
  - `LongTVQA_plus_subtitle_clip_level.json`
  - `LongTVQA_plus_subtitle_episode_level.json`
- `--grounding-cache-json` is optional. If it is missing, unavailable, or has no usable mapping for a question, the script randomly selects one initial clip as fallback; this usually degrades agent performance.
- To sample a subset before split, add `--subset-size N`.
- If `--subset-size` is omitted (or `0`), the full dataset is used.
