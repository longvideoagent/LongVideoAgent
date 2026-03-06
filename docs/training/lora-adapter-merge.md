# LoRA Adapter Merge

After LoRA training, you can merge the adapter into the base model and export a full HuggingFace model directory.

## Script

- `scripts/merge_lora_adapter.sh`

## Usage: adapter directory

Use this when the adapter path is a directory containing `adapter_model.safetensors` and `adapter_config.json`.

```bash
scripts/merge_lora_adapter.sh \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --adapter_path /path/to/lora_adapter \
  --output_dir /path/to/merged_hf_model
```

## Usage: single adapter file

Use this when the adapter is a single safetensors file.

```bash
scripts/merge_lora_adapter.sh \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --adapter_path /path/to/adapter.safetensors \
  --adapter_config /path/to/adapter_config.json \
  --output_dir /path/to/merged_hf_model
```

The output directory is a standard HuggingFace model folder that can be loaded with `from_pretrained`.
