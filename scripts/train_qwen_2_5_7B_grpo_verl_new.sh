#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_NEW_ROOT="${PROJECT_ROOT}/verl_new"
AGENT_LOOP_CONFIG="${PROJECT_ROOT}/videoagent/verl_ext/config/agent_loop.yaml"
REWARD_FN_PATH="${PROJECT_ROOT}/videoagent/verl_ext/reward.py"


# ==================== Runtime Environment ====================
export RAY_DISABLE_DOCKER_CPU_WARNING=1
# ==================== Experiment Settings ====================
export WAND_PROJECT="longvideoagent_train"
export BASE_MODEL="../model/Qwen2.5-7B-Instruct"
export EXPERIMENT_NAME="${WAND_PROJECT}_verl_new_$(date +%Y%m%d_%H%M%S)"

export TRAIN_DATA_DIR="/root/autodl-tmp/data/tvqa_plus"
export TEST_DATA_DIR="/root/autodl-tmp/data/tvqa_plus"
TRAIN_FILE="${TRAIN_DATA_DIR}/train.parquet"
VAL_FILE="${TEST_DATA_DIR}/val.parquet"

VISION_BASE_FRAME_DIR="../Tvqa_data/bbt_frames"
VISION_BBOX_JSON_PATH="../Tvqa_data/clip_bbox_mapping.json"
VISION_SUBS_PATH="../Tvqa_data/hf_datasets/LongTVQA_plus/LongTVQA_plus_subtitle_clip_level.json"

# ==================== Agent Settings (Vision / Grounding) ====================
MAX_TURNS=5
MAX_OBS_LEN=1500
VISION_MODEL="grok-4-fast-non-reasoning"
GROUNDING_MODEL="grok-4-fast-reasoning"

AGENT_SHARED_API_KEY="${AGENT_SHARED_API_KEY:-}"
export GROUNDING_API_KEY="${GROUNDING_API_KEY:-${QDD_API:-${qdd_api:-}}}"
export VISION_API_KEY="${VISION_API_KEY:-${QDD_API:-${qdd_api:-}}}"

VISION_BASE_URL="https://api2.aigcbest.top/v1"
GROUNDING_BASE_URL="https://api2.aigcbest.top/v1"

# ==================== Resource & Parallel Settings ====================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
N_GPUS_PER_NODE=1
NNODES=1
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
VLLM_GPU_MEMORY_UTILIZATION=0.30
ROLLOUT_MAX_NUM_BATCHED_TOKENS=15001
ROLLOUT_LOG_PROB_MICRO_BSZ=2
REF_LOG_PROB_MICRO_BSZ=2

# ==================== Batch & Sequence Length Settings ====================
TRAIN_BSZ=16
VAL_BSZ=4
MAX_PROMPT_LEN=12800
MAX_RESP_LEN=1500

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

echo "=== Training started (verl_new) ==="
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Train data: ${TRAIN_FILE}"
echo "Validation data: ${VAL_FILE}"
echo "Agent loop config: ${AGENT_LOOP_CONFIG}"
echo "Reward function: ${REWARD_FN_PATH}"
echo "Experiment name: ${EXPERIMENT_NAME}"



PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.return_raw_chat=True \
    data.train_batch_size=$TRAIN_BSZ \
    data.val_batch_size=$VAL_BSZ \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESP_LEN \
    data.truncation=error \
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
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOG_PROB_MICRO_BSZ \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_TURNS \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_TURNS \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=$MAX_OBS_LEN \
    actor_rollout_ref.rollout.agent.default_agent_loop=longvideoagent_multiturn \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$AGENT_LOOP_CONFIG \
    +actor_rollout_ref.rollout.custom={} \
    +actor_rollout_ref.rollout.custom={} \
    +actor_rollout_ref.rollout.custom.videoagent.max_turns=$MAX_TURNS \
    +actor_rollout_ref.rollout.custom.videoagent.max_obs_length=$MAX_OBS_LEN \
    +actor_rollout_ref.rollout.custom.videoagent.base_frame_dir="$VISION_BASE_FRAME_DIR" \
    +actor_rollout_ref.rollout.custom.videoagent.bbox_json_path="$VISION_BBOX_JSON_PATH" \
    +actor_rollout_ref.rollout.custom.videoagent.subs_path="$VISION_SUBS_PATH" \
    +actor_rollout_ref.rollout.custom.videoagent.vision_model="$VISION_MODEL" \
    +actor_rollout_ref.rollout.custom.videoagent.grounding_model="$GROUNDING_MODEL" \
    +actor_rollout_ref.rollout.custom.videoagent.grounding_temperature=0.6 \
    +actor_rollout_ref.rollout.custom.videoagent.grounding_max_tokens=512 \
    +actor_rollout_ref.rollout.custom.videoagent.api_key="$AGENT_SHARED_API_KEY" \
    +actor_rollout_ref.rollout.custom.videoagent.vision_api="$VISION_API_KEY" \
    +actor_rollout_ref.rollout.custom.videoagent.grounding_api="$GROUNDING_API_KEY" \
    +actor_rollout_ref.rollout.custom.videoagent.vision_base_url="$VISION_BASE_URL" \
    +actor_rollout_ref.rollout.custom.videoagent.grounding_base_url="$GROUNDING_BASE_URL" \
    +actor_rollout_ref.rollout.custom.videoagent.frame_start=1 \
    +actor_rollout_ref.rollout.custom.videoagent.frame_end=180 \
    +actor_rollout_ref.rollout.custom.videoagent.frame_step=15 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO_BSZ \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    algorithm.kl_ctrl.kl_coef=0.0001 \
    reward.custom_reward_function.path=$REWARD_FN_PATH \
    reward.custom_reward_function.name=compute_score \
    reward.reward_manager.name=naive \
    trainer.critic_warmup=0 \
    'trainer.logger=["console"]' \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=1 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="${PROJECT_ROOT}/outputs/${EXPERIMENT_NAME}" \
    trainer.val_only=false \
    trainer.val_before_train=false \
    "$@" \
    2>&1 | tee "${PROJECT_ROOT}/${EXPERIMENT_NAME}.log"

echo "=== Training completed (verl_new) ==="
echo "Log file: ${PROJECT_ROOT}/${EXPERIMENT_NAME}.log"
