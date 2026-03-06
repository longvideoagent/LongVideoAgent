#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="${DATA_DIR:-../Tvqa_data}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_DIR}"
LONGTVQA_REPO="${LONGTVQA_REPO:-longvideoagent/LongTVQA}"
LONGTVQA_PLUS_REPO="${LONGTVQA_PLUS_REPO:-longvideoagent/LongTVQA_plus}"
TVQA_FRAMES_REPO="${TVQA_FRAMES_REPO:-Yuxuan0701/tvqa-frames}"
FRAMES_SHOW="${FRAMES_SHOW:-bbt}"

DATA_ROOT="${DATA_ROOT:-$DATA_DIR/hf_datasets}"
LONGTVQA_DIR="${LONGTVQA_DIR:-$DATA_ROOT/LongTVQA}"
LONGTVQA_PLUS_DIR="${LONGTVQA_PLUS_DIR:-$DATA_ROOT/LongTVQA_plus}"
TVQA_FRAMES_DIR="${TVQA_FRAMES_DIR:-$DATA_DIR/frames}"

echo "=== Download + Prepare LongTVQA Datasets ==="
echo "data_dir: $DATA_DIR"
echo "data_root: $DATA_ROOT"
echo "output_dir: $OUTPUT_DIR"
echo "LongTVQA dir: $LONGTVQA_DIR"
echo "LongTVQA+ dir: $LONGTVQA_PLUS_DIR"
echo "TVQA frames dir: $TVQA_FRAMES_DIR"
echo "Frames show: $FRAMES_SHOW"

if command -v hf >/dev/null 2>&1; then
  DL_BIN="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  DL_BIN="huggingface-cli"
else
  echo "[ERROR] Neither 'hf' nor 'huggingface-cli' is installed." >&2
  exit 1
fi

mkdir -p "$LONGTVQA_DIR" "$LONGTVQA_PLUS_DIR" "$TVQA_FRAMES_DIR" "$OUTPUT_DIR"

if [[ "$DL_BIN" == "hf" ]]; then
  hf download "$LONGTVQA_REPO" --repo-type dataset --local-dir "$LONGTVQA_DIR"
  hf download "$LONGTVQA_PLUS_REPO" --repo-type dataset --local-dir "$LONGTVQA_PLUS_DIR"
else
  huggingface-cli download "$LONGTVQA_REPO" --repo-type dataset --local-dir "$LONGTVQA_DIR"
  huggingface-cli download "$LONGTVQA_PLUS_REPO" --repo-type dataset --local-dir "$LONGTVQA_PLUS_DIR"
fi

python src/dataset/download_tvqa_frames.py \
  --repo-id "$TVQA_FRAMES_REPO" \
  --output-dir "$TVQA_FRAMES_DIR" \
  --show "$FRAMES_SHOW"

# Full list has six archives; keep only bbt enabled for now.
case "$FRAMES_SHOW" in
  bbt) selected_archive="bbt_frames.tar.gz" ;;
  castle) selected_archive="castle_frames.tar" ;;
  friends) selected_archive="friends_frames.tar" ;;
  grey) selected_archive="grey_frames.tar" ;;
  house) selected_archive="house_frames.tar.gz" ;;
  met) selected_archive="met_frames.tar.gz" ;;
  *)
    echo "[ERROR] Unsupported FRAMES_SHOW: $FRAMES_SHOW" >&2
    exit 1
    ;;
esac

FRAMES_ARCHIVES=(
  "$selected_archive"
  # "bbt_frames.tar.gz"
  # "castle_frames.tar"
  # "friends_frames.tar"
  # "grey_frames.tar"
  # "house_frames.tar.gz"
  # "met_frames.tar.gz"
)

python src/dataset/prepare_longtvqa_assets.py \
  --longtvqa-dir "$LONGTVQA_DIR" \
  --longtvqa-plus-dir "$LONGTVQA_PLUS_DIR" \
  --output-dir "$OUTPUT_DIR"

for archive in "${FRAMES_ARCHIVES[@]}"; do
  archive_path="$TVQA_FRAMES_DIR/$archive"
  if [[ ! -f "$archive_path" ]]; then
    echo "[WARN] Frame archive not found, skip extract: $archive_path"
    continue
  fi

  if [[ "$archive" == *.tar.gz ]]; then
    extracted_name="${archive%.tar.gz}"
    if [[ -d "$TVQA_FRAMES_DIR/$extracted_name" ]]; then
      echo "Skip extract (already exists): $TVQA_FRAMES_DIR/$extracted_name"
      continue
    fi
    tar -xzf "$archive_path" -C "$TVQA_FRAMES_DIR"
    echo "Extracted: $archive_path -> $TVQA_FRAMES_DIR"
  elif [[ "$archive" == *.tar ]]; then
    extracted_name="${archive%.tar}"
    if [[ -d "$TVQA_FRAMES_DIR/$extracted_name" ]]; then
      echo "Skip extract (already exists): $TVQA_FRAMES_DIR/$extracted_name"
      continue
    fi
    tar -xf "$archive_path" -C "$TVQA_FRAMES_DIR"
    echo "Extracted: $archive_path -> $TVQA_FRAMES_DIR"
  else
    echo "[WARN] Unsupported archive format, skip: $archive_path"
  fi
done

echo "=== Done ==="
echo "Prepared files in: $OUTPUT_DIR"
echo "- LongTVQA_val_normalized.jsonl"
echo "- LongTVQA_plus_val_normalized.json"
echo "- clip_bbox_mapping.json"
