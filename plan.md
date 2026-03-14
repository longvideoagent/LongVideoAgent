# LongVideoAgent 升级到 verl_new 的迁移计划

## 0. 目标

将当前基于 `verl/` 的 LongVideoAgent 训练链路升级到 `verl_new/verl/`，
并满足下面两个要求：

1. 功能行为不变
2. 尽量不继续 fork `verl` 核心 trainer，而是迁移到 `verl_new` 的官方扩展点

---

## 1. 当前必须保留的功能基线

迁移后，以下行为必须保持一致：

### 1.1 多轮 agent rollout 行为不变

- 模型输出三类动作标签：
  - `<search>...</search>`
  - `<request_grounding>...</request_grounding>`
  - `<answer>...</answer>`
- 每轮根据动作执行外部逻辑：
  - `search` -> 视觉模型读取帧
  - `request_grounding` -> grounding 模型定位 clip
  - `answer` -> 终止轨迹
- 支持最大轮数限制
- 支持将 observation 继续拼回上下文，进入下一轮生成

对应现有实现：

- [videoagent/action_generation.py](/home/ziyao/LongVideoAgent/videoagent/action_generation.py)
- [verl/trainer/ppo/ray_trainer.py](/home/ziyao/LongVideoAgent/verl/trainer/ppo/ray_trainer.py#L1146)
- [verl/trainer/dapo/dapo_ray_trainer.py](/home/ziyao/LongVideoAgent/verl/trainer/dapo/dapo_ray_trainer.py#L112)

### 1.2 loss mask 语义不变

- 只有模型自己生成的 token 参与 actor loss
- 工具/环境返回的 observation token 不参与 actor loss
- 当前通过 `info_mask -> loss_mask` 实现

对应现有实现：

- [videoagent/action_generation.py](/home/ziyao/LongVideoAgent/videoagent/action_generation.py#L589)
- [verl/trainer/ppo/ray_trainer.py](/home/ziyao/LongVideoAgent/verl/trainer/ppo/ray_trainer.py#L1482)

### 1.3 reward 行为不变

- `tvqa_plus_vision` 保持现有 reward 逻辑
- 多轮样本保留 tool-use bonus 语义
- 其他 QA 数据源继续兼容
- 返回额外统计信息的能力保留

对应现有实现：

- [videoagent/reward.py](/home/ziyao/LongVideoAgent/videoagent/reward.py)

### 1.4 数据字段约定不变

- 继续依赖 `extra_info`
- 继续支持 `vid_name` / `predicted_vid` / `video_id`
- 继续支持 `choices` / `original_question`
- 继续保留 `reward_model.ground_truth`

对应现有实现：

- [videoagent/action_generation.py](/home/ziyao/LongVideoAgent/videoagent/action_generation.py#L457)

---

## 2. 迁移原则

### 2.1 不再继续改 trainer 主流程

旧版中直接改了：

- [verl/trainer/main_ppo.py](/home/ziyao/LongVideoAgent/verl/trainer/main_ppo.py)
- [verl/trainer/ppo/ray_trainer.py](/home/ziyao/LongVideoAgent/verl/trainer/ppo/ray_trainer.py)
- [recipe/dapo/main_dapo.py](/home/ziyao/LongVideoAgent/recipe/dapo/main_dapo.py)
- [verl/trainer/dapo/dapo_ray_trainer.py](/home/ziyao/LongVideoAgent/verl/trainer/dapo/dapo_ray_trainer.py)

升级后应尽量改成：

- 自定义 AgentLoop
- 自定义 reward function / reward manager
- 必要时自定义 AgentLoopManager
- 尽量不 patch `verl_new/verl/trainer/ppo/ray_trainer.py`

### 2.2 先迁 PPO 主线，再考虑 DAPO

原因：

- `verl_new` 已经原生支持 async multi-turn + reward loop
- 旧 DAPO 路径在新版本中不是一比一对应
- 先跑通 PPO/GRPO，可快速验证核心功能未变

### 2.3 先做“兼容迁移”，再做“结构优化”

第一阶段允许保留部分旧逻辑封装。
第二阶段再逐步把配置和接口整理成新版本原生风格。

---

## 3. 新版本里的承接点

### 3.1 多轮 rollout 官方入口

新版本已有：

- [verl_new/verl/trainer/ppo/ray_trainer.py](/home/ziyao/LongVideoAgent/verl_new/verl/trainer/ppo/ray_trainer.py#L807)
- [verl_new/verl/experimental/agent_loop/agent_loop.py](/home/ziyao/LongVideoAgent/verl_new/verl/experimental/agent_loop/agent_loop.py)
- [verl_new/verl/trainer/config/rollout/rollout.yaml](/home/ziyao/LongVideoAgent/verl_new/verl/trainer/config/rollout/rollout.yaml#L159)

### 3.2 reward 官方入口

新版本已有：

- [verl_new/verl/experimental/reward_loop/reward_loop.py](/home/ziyao/LongVideoAgent/verl_new/verl/experimental/reward_loop/reward_loop.py)
- [verl_new/verl/trainer/ppo/reward.py](/home/ziyao/LongVideoAgent/verl_new/verl/trainer/ppo/reward.py)

### 3.3 可插拔 manager

新版本支持：

- `agent_loop_manager_class`

见：

- [verl_new/verl/workers/config/rollout.py](/home/ziyao/LongVideoAgent/verl_new/verl/workers/config/rollout.py#L70)
- [verl_new/verl/trainer/ppo/ray_trainer.py](/home/ziyao/LongVideoAgent/verl_new/verl/trainer/ppo/ray_trainer.py#L823)

---

## 4. 分阶段计划

## Phase 1: 建立迁移基线

### 任务

- 梳理旧实现与新扩展点映射
- 冻结一份“功能对齐基线”

### 产出

- 一份字段/行为映射表
- 一组 golden samples

### 要做的事

- 从当前训练数据里抽取 20~50 条代表样本
- 固定：
  - prompt
  - extra_info
  - ground_truth
  - video_id
  - max_turns
- 在旧链路上保存：
  - 每轮动作
  - 每轮 observation
  - 最终 response
  - reward
  - num_turns
  - loss mask coverage

### 验收标准

- 基线样本可重复回放
- 后续每阶段都能对比

---

## Phase 2: 提取 VideoAgent 业务逻辑为独立扩展模块

### 任务

把当前直接塞进 trainer 的业务逻辑抽出来。

### 新建建议目录

- `videoagent/verl_ext/agent_loop.py`
- `videoagent/verl_ext/reward.py`
- `videoagent/verl_ext/config/`
- `videoagent/verl_ext/__init__.py`

### 要迁移的内容

#### A. rollout 逻辑

来源：

- [videoagent/action_generation.py](/home/ziyao/LongVideoAgent/videoagent/action_generation.py)

迁移目标：

- 封装为 `AgentLoopBase` 子类
- 每个 sample 独立运行多轮逻辑
- 不再在 trainer 里手工调 `VisionLLMGenerationManager`

#### B. reward 逻辑

来源：

- [videoagent/reward.py](/home/ziyao/LongVideoAgent/videoagent/reward.py)

迁移目标：

- 封装为新 reward loop 可调用的 custom reward function / custom reward manager
- 保留 `tvqa_plus_vision` 逻辑
- 保留额外 reward info 输出

### 验收标准

- 新模块不依赖旧版 `verl/trainer/ppo/ray_trainer.py`
- 旧业务逻辑可以单元测试调用

---

## Phase 3: 在 verl_new 上接入自定义 AgentLoop

### 任务

让 `verl_new` 通过官方 agent loop 跑 LongVideoAgent。

### 做法

- 采用 `verl_new` 的 async rollout 流程
- 在配置里注册自定义 agent loop
- 先不要自定义 AgentLoopManager，优先走官方默认 manager

### 关键点

- 用 `raw_prompt` 作为输入，而不是旧版 dataset 里预先生成的 `input_ids`
- observation 通过官方 multi-turn/agent-loop 的 response_mask 语义表达
- 不再自己维护 `info_mask`

参考新版本能力：

- [verl_new/verl/experimental/agent_loop/agent_loop.py](/home/ziyao/LongVideoAgent/verl_new/verl/experimental/agent_loop/agent_loop.py#L569)
- [verl_new/verl/experimental/agent_loop/agent_loop.py](/home/ziyao/LongVideoAgent/verl_new/verl/experimental/agent_loop/agent_loop.py#L789)

### 验收标准

- 同一批基线样本上：
  - 每轮动作类型一致率高
  - 终止条件一致
  - num_turns 接近
  - response 结构一致

---

## Phase 4: 在 verl_new 上接入 reward loop

### 任务

把旧 reward 接到新 reward 体系，不再改 `main_ppo.py`。

### 做法

- 通过 `reward.custom_reward_function` 或自定义 reward manager 接入
- 如果需要，把旧 `CustomRewardManager` 改写成新 reward loop manager 风格
- 让 agent loop 透传 `tool_extra_fields` / `reward_extra_info`

参考：

- [verl_new/verl/experimental/reward_loop/reward_loop.py](/home/ziyao/LongVideoAgent/verl_new/verl/experimental/reward_loop/reward_loop.py#L38)
- [verl_new/verl/experimental/reward_loop/reward_manager/naive.py](/home/ziyao/LongVideoAgent/verl_new/verl/experimental/reward_loop/reward_manager/naive.py)

### 验收标准

- reward 数值与旧版对齐
- extra info 字段完整
- validation 指标计算结果一致

---

## Phase 5: 配置迁移

### 任务

把旧配置改成新版本原生结构。

### 需要迁移的旧字段

- 顶层 `max_turns`
- 顶层 `vision.*`
- 旧的 `reward_model.*` 兼容字段
- 旧脚本里的 `+data.max_obs_length`、`+data.max_start_length`

### 建议迁移方向

- `max_turns` -> `actor_rollout_ref.rollout.multi_turn.max_assistant_turns` / `max_user_turns`
- `vision.*` -> 放入自定义 agent loop config
- reward 相关 -> 放入 `reward.*`
- 视频业务自定义参数 -> 放入自定义 config block，不污染 verl 主 config

### 验收标准

- 训练脚本不再依赖旧版魔改字段
- 新脚本只依赖 `verl_new` 官方 config + 你自己的扩展 config

---

## Phase 6: PPO 主线回归测试

### 任务

验证“功能不变”。

### 必做测试

- 单样本 rollout 对比
- 小 batch rollout 对比
- reward 对比
- actor loss mask 覆盖率对比
- validation 对比
- checkpoint save/load 对比

### 建议对比项

- `response`
- `num_turns`
- `reward`
- `loss_mask.sum()`
- `state_tokens/coverage`
- `val accuracy`

### 验收标准

- 关键指标误差在可接受范围内
- 无明显行为回归

---

## Phase 7: DAPO 迁移

### 任务

在 PPO 主线稳定后，再处理 DAPO。

### 原因

- 旧 DAPO 同样耦合了 `VisionLLMGenerationManager`
- 新版本未必保留旧 DAPO 入口结构
- 提前迁会放大调试面

### 做法

- 先复用 Phase 2/3/4 已提取的 AgentLoop 与 reward
- 再找 `verl_new` 中最接近 DAPO 的实现入口挂接

### 验收标准

- DAPO 不再复制一套 VideoAgent 逻辑
- 与 PPO 共用同一套 agent/reward 扩展

---

## 5. 风险点

### 高风险

- 旧版 `info_mask` 与新版本 `response_mask/loss_mask` 语义是否完全一致
- 旧版 dataset 先 tokenize，新版 dataset 以 `raw_prompt` 为主，token 边界可能变化
- grounding / vision API 调用并发后，轨迹顺序和 timing 可能影响行为
- reward 中对 `solution_str` 的 decode 边界可能因新拼接方式变化

### 中风险

- DAPO 路径和 PPO 路径行为分叉
- 多模态 processor 在新 agent loop 中的输入结构变化
- tool/observation token 是否会触发不同的 truncate 行为

### 低风险

- config 字段迁移
- 日志指标名字变化
- checkpoint 路径差异

---

## 6. 建议的执行顺序

1. 做基线样本与行为快照
2. 提取 `videoagent` 扩展模块
3. 在 `verl_new` 上接入自定义 AgentLoop
4. 接入 reward loop
5. 跑 PPO 小规模回归
6. 再迁 DAPO
7. 最后清理旧版 trainer patch

---

## 7. 完成标准

满足以下条件才算升级完成：

- `verl_new` 上 PPO/GRPO 可训练
- LongVideoAgent 多轮 agent 行为保持不变
- reward 行为保持不变
- observation token 不参与 actor loss
- validation 指标不回退
- 不再依赖旧版 fork trainer 主流程
- DAPO 如需保留，也复用同一套扩展实现
