#!/bin/bash
set -o pipefail

# ==================== Runtime Environment ====================
export RAY_TMPDIR=/home/rliuay/runtao/proj_videoqa
export VLLM_USE_V1=0
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# ==================== Experiment Settings ====================
export WAND_PROJECT="longvideoagent_train"
export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
export EXPERIMENT_NAME="${WAND_PROJECT}$(date +%Y%m%d_%H%M%S)"

export TRAIN_DATA_DIR='/home/rliuay/runtao/proj_videoqa/Vision-R1_bbox/data/tvqa_plus/train'
export TEST_DATA_DIR='/home/rliuay/runtao/proj_videoqa/Vision-R1_bbox/data/tvqa_plus/test'
TRAIN_FILE="${TRAIN_DATA_DIR}/train_newaction.parquet"
VAL_FILE="${TEST_DATA_DIR}/test.parquet"

VISION_BASE_FRAME_DIR="../Tvqa_data/bbt_frames"
VISION_BBOX_JSON_PATH="../Tvqa_data/clip_bbox_mapping.json"

# ==================== Agent Settings (Vision / Grounding) ====================
DO_VISION=true
MAX_TURNS=5
VISION_MODEL="grok-4-fast-non-reasoning"
GROUNDING_MODEL="grok-4-fast-reasoning"

# API keys: keep empty to fallback to environment variables in code.
AGENT_SHARED_API_KEY="${AGENT_SHARED_API_KEY:-}"
VISION_API_KEY="${VISION_API_KEY:-}"
GROUNDING_API_KEY="${GROUNDING_API_KEY:-}"
MAIN_API_KEY="${MAIN_API_KEY:-}"

# OpenAI-compatible base URLs for each agent client.
VISION_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
GROUNDING_BASE_URL="https://api2.aigcbest.top/v1"
MAIN_BASE_URL="https://api2.aigcbest.top/v1"

# ==================== Resource & Parallel Settings ====================
# Default to 8-GPU mask; override by exporting CUDA_VISIBLE_DEVICES before running.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
N_GPUS_PER_NODE=4
NNODES=1
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=2
VLLM_GPU_MEMORY_UTILIZATION=0.30
ROLLOUT_MAX_NUM_BATCHED_TOKENS=15001
ROLLOUT_LOG_PROB_MICRO_BSZ=2
REF_LOG_PROB_MICRO_BSZ=2

# ==================== Batch & Sequence Length Settings ====================
TRAIN_BSZ=16
VAL_BSZ=4
MAX_PROMPT_LEN=12800
MAX_RESP_LEN=1500
MAX_OBS_LEN=-1
MAX_START_LEN=9999

# ==================== Optimization Settings ====================
LR=5e-6
LORA_RANK=128
LORA_ALPHA=64

# ==================== RL Algorithm (GRPO) ====================
ADV_ESTIMATOR=grpo
ROLLOUT_N=8

# ==================== Training Schedule ====================
SAVE_FREQ=25
TEST_FREQ=5000
TOTAL_TRAINING_STEPS=401

echo "=== Training started ==="
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Train data: ${TRAIN_FILE}"
echo "Validation data: ${VAL_FILE}"
echo "Experiment name: ${EXPERIMENT_NAME}"

# Training config
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$TRAIN_BSZ \
    data.val_batch_size=$VAL_BSZ \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESP_LEN \
    +data.max_obs_length=$MAX_OBS_LEN \
    +data.max_start_length=$MAX_START_LEN \
    data.truncation='error' \
    +data.shuffle_train_dataloader=True \
    +data.video_id_key="vid_name" \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOG_PROB_MICRO_BSZ \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO_BSZ \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    algorithm.kl_ctrl.kl_coef=0.0001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=1 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=../$EXPERIMENT_NAME \
    trainer.val_only=false \
    trainer.val_before_train=false \
    +max_turns=$MAX_TURNS \
    +do_vision=$DO_VISION \
    +vision.base_frame_dir="$VISION_BASE_FRAME_DIR" \
    +vision.bbox_json_path="$VISION_BBOX_JSON_PATH" \
    +vision.vision_model="$VISION_MODEL" \
    +vision.grounding_model="$GROUNDING_MODEL" \
    +vision.grounding_temperature=0.6 \
    +vision.grounding_max_tokens=512 \
    +vision.api_key="$AGENT_SHARED_API_KEY" \
    +vision.vision_api="$VISION_API_KEY" \
    +vision.grounding_api="$GROUNDING_API_KEY" \
    +vision.main_api="$MAIN_API_KEY" \
    +vision.vision_base_url="$VISION_BASE_URL" \
    +vision.grounding_base_url="$GROUNDING_BASE_URL" \
    +vision.main_base_url="$MAIN_BASE_URL" \
    2>&1 | tee "${EXPERIMENT_NAME}.log"

echo "=== Training completed ==="
echo "Log file: ${EXPERIMENT_NAME}.log"
echo "Checkpoint location: verl_checkpoints/${EXPERIMENT_NAME}"
