# LongVideoAgent `verl_new` Migration Notes

This repository now includes a PPO/GRPO training path based on `verl_new` official extension points.

## What changed

- Added custom agent loop: `videoagent/verl_ext/agent_loop.py`
- Added custom reward function: `videoagent/verl_ext/reward.py`
- Added agent loop registration config: `videoagent/verl_ext/config/agent_loop.yaml`
- Added new launch script: `scripts/train_qwen_2_5_7B_grpo_verl_new.sh`
- Dataset conversion now writes `agent_name=longvideoagent_multiturn` for direct routing.

## Config mapping (old -> new)

- `max_turns` -> `actor_rollout_ref.rollout.multi_turn.max_assistant_turns` and `max_user_turns`
- `vision.*` -> `actor_rollout_ref.rollout.custom.videoagent.*`
- `reward_model.*` custom logic -> `reward.custom_reward_function.path/name`
- `info_mask`-style masking -> `response_mask` in custom agent loop (tool observations are `0`)

## Launch

```bash
bash scripts/train_qwen_2_5_7B_grpo_verl_new.sh
```

The script runs from `verl_new/` and injects `${PROJECT_ROOT}` into `PYTHONPATH` so `videoagent.verl_ext.*` can be imported by Ray workers.

