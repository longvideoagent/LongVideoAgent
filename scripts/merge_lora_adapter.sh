#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Merge a LoRA adapter into a base model and export a HuggingFace model directory.

Usage:
  scripts/merge_lora_adapter.sh \
    --base_model <base_model_path_or_hf_id> \
    --adapter_path <adapter_dir_or_adapter.safetensors> \
    --output_dir <output_dir> \
    [--adapter_config <adapter_config.json>] \
    [--dtype auto|fp16|bf16|fp32] \
    [--device_map auto|cpu] \
    [--trust_remote_code]

Required:
  --base_model      Base model path or HuggingFace model ID.
  --adapter_path    PEFT adapter directory or single adapter safetensors file.
  --output_dir      Output directory for merged HuggingFace model.

Notes:
  - If --adapter_path is a file (for example adapter.safetensors), the script expects
    adapter_config.json in the same directory unless --adapter_config is provided.
  - Output is a full HuggingFace model folder that can be loaded with from_pretrained().
EOF
}

BASE_MODEL=""
ADAPTER_PATH=""
ADAPTER_CONFIG=""
OUTPUT_DIR=""
DTYPE="auto"
DEVICE_MAP="auto"
TRUST_REMOTE_CODE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base_model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --adapter_path)
      ADAPTER_PATH="$2"
      shift 2
      ;;
    --adapter_config)
      ADAPTER_CONFIG="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    --device_map)
      DEVICE_MAP="$2"
      shift 2
      ;;
    --trust_remote_code)
      TRUST_REMOTE_CODE="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$BASE_MODEL" || -z "$ADAPTER_PATH" || -z "$OUTPUT_DIR" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

if [[ "$DTYPE" != "auto" && "$DTYPE" != "fp16" && "$DTYPE" != "bf16" && "$DTYPE" != "fp32" ]]; then
  echo "Invalid --dtype: $DTYPE (use auto|fp16|bf16|fp32)" >&2
  exit 1
fi

if [[ "$DEVICE_MAP" != "auto" && "$DEVICE_MAP" != "cpu" ]]; then
  echo "Invalid --device_map: $DEVICE_MAP (use auto|cpu)" >&2
  exit 1
fi

TMP_ADAPTER_DIR=""
cleanup() {
  if [[ -n "$TMP_ADAPTER_DIR" && -d "$TMP_ADAPTER_DIR" ]]; then
    rm -rf "$TMP_ADAPTER_DIR"
  fi
}
trap cleanup EXIT

normalize_adapter_dir() {
  local input_path="$1"
  local config_override="$2"

  if [[ -d "$input_path" ]]; then
    if [[ -f "$input_path/adapter_model.safetensors" && -f "$input_path/adapter_config.json" ]]; then
      echo "$input_path"
      return 0
    fi

    if [[ -f "$input_path/adapter.safetensors" && -f "$input_path/adapter_config.json" ]]; then
      TMP_ADAPTER_DIR="$(mktemp -d)"
      cp "$input_path/adapter.safetensors" "$TMP_ADAPTER_DIR/adapter_model.safetensors"
      cp "$input_path/adapter_config.json" "$TMP_ADAPTER_DIR/adapter_config.json"
      echo "$TMP_ADAPTER_DIR"
      return 0
    fi

    echo "Adapter directory must contain adapter_model.safetensors (or adapter.safetensors) and adapter_config.json" >&2
    return 1
  fi

  if [[ -f "$input_path" ]]; then
    local cfg_path="$config_override"
    if [[ -z "$cfg_path" ]]; then
      cfg_path="$(dirname "$input_path")/adapter_config.json"
    fi

    if [[ ! -f "$cfg_path" ]]; then
      echo "adapter_config.json not found. Provide --adapter_config when --adapter_path is a file." >&2
      return 1
    fi

    TMP_ADAPTER_DIR="$(mktemp -d)"
    cp "$input_path" "$TMP_ADAPTER_DIR/adapter_model.safetensors"
    cp "$cfg_path" "$TMP_ADAPTER_DIR/adapter_config.json"
    echo "$TMP_ADAPTER_DIR"
    return 0
  fi

  echo "Invalid --adapter_path: $input_path" >&2
  return 1
}

ADAPTER_DIR="$(normalize_adapter_dir "$ADAPTER_PATH" "$ADAPTER_CONFIG")"
mkdir -p "$OUTPUT_DIR"

python3 - "$BASE_MODEL" "$ADAPTER_DIR" "$OUTPUT_DIR" "$DTYPE" "$DEVICE_MAP" "$TRUST_REMOTE_CODE" <<'PY'
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model, adapter_dir, output_dir, dtype_name, device_map, trust_remote_code_raw = sys.argv[1:]
trust_remote_code = trust_remote_code_raw.lower() == "true"

dtype_map = {
    "auto": "auto",
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
torch_dtype = dtype_map[dtype_name]

print(f"[merge] base_model={base_model}")
print(f"[merge] adapter_dir={adapter_dir}")
print(f"[merge] output_dir={output_dir}")
print(f"[merge] dtype={dtype_name}, device_map={device_map}, trust_remote_code={trust_remote_code}")

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch_dtype,
    device_map=device_map,
    low_cpu_mem_usage=True,
    trust_remote_code=trust_remote_code,
)
model = PeftModel.from_pretrained(
    model,
    adapter_dir,
    is_trainable=False,
    trust_remote_code=trust_remote_code,
)
model = model.merge_and_unload()
model.save_pretrained(output_dir, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
tokenizer.save_pretrained(output_dir)

print("[merge] done")
PY

echo "Merged HuggingFace model saved to: $OUTPUT_DIR"
