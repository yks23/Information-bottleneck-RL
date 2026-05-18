# Information Bottleneck RLVR AlphaZero 实验框架

这份代码实现的是一个面向 RLVR 的 AlphaZero-style 实验框架。当前正式实验环境是标准 `6x7 Connect4`，`TicTacToe` 只用于 smoke test。框架的核心目标是研究：在 rollout search、verifiable reward、reward combination、reward noise、self-play 参数共享等因素变化时，策略学习过程中的信息量和性能指标如何演化。

## 当前实验在做什么

正式训练脚本会在 4 张 GPU 上并行跑一组 ablation：

- `baseline`: rollout=32，RLVR reward，无噪声，共享参数 self-play。
- `noise`: rollout=32，在 RLVR combined reward 上加入 `N(0, c * I)` 噪声，当前覆盖 `c=0.01/0.05/0.10`。
- `nonshared`: rollout=32，actor 和 opponent 不共享参数，用冻结 opponent snapshot 做对照。
- `multi-seed`: 每个设置按 seed 串行排队，方便统计稳定性。

训练日志同时写到本地和 W&B：

- 本地 metrics: `runs/<run_id>/metrics.jsonl`
- checkpoints: `runs/<run_id>/checkpoints`
- W&B project: `information-bottleneck-alphazero`
- 当前 status: `scripts/training_status.sh`

## 代码结构

主要代码在 `src/ibrlvr`：

- `envs/tictactoe.py`: TicTacToe 环境，主要用于 smoke test。
- `envs/connect4.py`: Connect4 / Connect4-small 环境，正式实验使用 `6x7 Connect4`。
- `envs/__init__.py`: 环境工厂，按配置名创建初始状态和 validation states。
- `model.py`: policy/value 网络，输入为棋盘 observation，输出 policy logits 和 scalar value。
- `mcts.py`: PUCT-style MCTS，使用当前网络做 leaf evaluation 和 prior。
- `rewards.py`: RLVR verifier 层，负责产生可验证 reward components、combined reward 和 noisy reward。
- `metrics.py`: entropy、信息量积分、reward combination 信息量等统计函数。
- `replay.py`: self-play transition replay buffer。
- `logging.py`: 本地 `metrics.jsonl` 与 W&B 双记录。
- `cli/train.py`: 训练主循环，把环境、MCTS、RLVR reward、replay、优化和评估串起来。

配置和脚本：

- `configs/connect4_gpu_rollout32.yaml`: 正式 baseline。
- `configs/connect4_gpu_noise.yaml`: reward noise ablation。
- `configs/connect4_gpu_nonshared.yaml`: shared vs non-shared self-play ablation。
- `configs/connect4_small_cpu_*.yaml`: CPU 快速调试。
- `configs/tictactoe_*.yaml`: smoke test。
- `scripts/run_gpu_formal_training.sh`: 后台启动 4-GPU 正式实验。
- `scripts/run_gpu_ablation_matrix.sh`: 4 个 GPU worker 的 ablation 队列。
- `scripts/training_status.sh`: 查看 launcher、worker、GPU 和最新 metrics。
- `scripts/export_run_summary.py`: 从 `metrics.jsonl` 导出 summary CSV。

## 训练框架逻辑

每个训练 step 做以下事情：

1. 创建若干个 self-play episodes。
2. 对每个非终局状态运行 MCTS，rollout 数由 `mcts.num_rollouts` 控制，正式实验为 `32`。
3. MCTS 返回 action visit distribution，作为 policy target。
4. 环境执行采样到的 action，直到 episode 终局。
5. 终局状态交给 RLVR verifier，得到 correctness reward、spec reward、combined reward 和 noisy reward。
6. 根据终局胜负，把 verifier reward 变成每个历史状态的 value target。
7. transition 写入 replay buffer。
8. 从 replay buffer 采样 batch，训练 policy/value 网络。
9. 定期在 validation states 和 train starts 上评估。
10. 每步记录 metrics，本地和 W&B 同步。

这个过程是 AlphaZero-style 的：policy improvement 来自 MCTS visit distribution，value learning 来自 self-play terminal outcome。但 reward 不是 learned reward model，而是 RLVR verifier 产出的可验证 reward。

## RLVR 是如何实现的

RLVR 逻辑集中在 `src/ibrlvr/rewards.py`。

`RLVRVerifier` 当前包含两个 verifiable reward source：

- `correctness`: 终局是否非平局。Connect4 中只要有明确 winner 就记为 `1.0`，平局为 `0.0`。
- `spec`: 终局赢家是否满足中心列控制规则。这个规则用于提供第二个可验证 reward 来源，避免 reward combination 只是接口占位。

组合 reward 为：

```text
combined = correctness_weight * correctness + format_weight * spec
```

配置字段在 `reward` 下：

```yaml
reward:
  correctness_weight: 1.0
  format_weight: 0.25
  noise_c: 0.0
```

如果开启噪声，则训练 value target 使用：

```text
noisy = combined + Normal(0, sqrt(noise_c * I))
```

其中 `I` 是 rollout correctness entropy 的累计积分。

对应 metrics：

- `rlvr/correctness_verifier`
- `rlvr/spec_verifier`
- `rlvr/combined_reward`
- `reward/correct`
- `reward/spec`
- `reward/combined`
- `reward/noisy`
- `noise/c`
- `noise/std`

## 你的 6 个目标如何落到代码和指标

### 1. rollout=32 的正确率 p、信息量 H(p)、积分 I

实现位置：

- MCTS rollout 数：`configs/connect4_gpu_*.yaml` 的 `mcts.num_rollouts: 32`
- rollout 统计：`src/ibrlvr/mcts.py`
- 信息量累计：`src/ibrlvr/metrics.py` 的 `InformationAccumulator`
- 训练调用：`src/ibrlvr/cli/train.py`

核心指标：

- `rollout/p`: MCTS simulations 中 value > 0 的比例，作为 rollout correctness probability。
- `info/H_p_bits`: binary entropy `H(p)`，单位 bits。
- `info/H_p_nats`: binary entropy `H(p)`，单位 nats。
- `info/I_cumulative_bits`: `sum_i H(p_i)`。
- `info/I_cumulative_nats`: nats 单位累计。

当前定义是按 train step 聚合：每个 step 内多个 episodes 和多个 states 的 rollout/p 先求均值，再更新一次信息积分。

### 2. rollout sequence 的 per-token avg entropy 和 seq-level cumulative entropy

这里的 token 对应一步 action decision，entropy 定义在 action space 的 MCTS visit distribution 上。

实现位置：

- action distribution entropy: `src/ibrlvr/cli/train.py`
- entropy 函数: `src/ibrlvr/metrics.py` 的 `categorical_entropy`

核心指标：

- `entropy/action_avg`: 一个 episode 内每个 action distribution entropy 的平均值，再对本 step 的 episodes 求平均。
- `entropy/seq_cumulative`: 一个 episode 的 action entropy 总和，再对本 step 的 episodes 求平均。
- `mcts/root_visit_entropy`: 当前 root visit distribution 的 entropy。

这些指标用来观察 policy 是否 entropy collapse，或者搜索分布是否逐渐变尖。

### 3. validation set 和 training set performance

实现位置：

- 评估函数：`src/ibrlvr/cli/train.py` 的 `evaluate`
- validation states：`src/ibrlvr/envs/connect4.py` 的 `validation_states`

核心指标：

- `val/success_rate`
- `val/avg_reward`
- `val/avg_episode_length`
- `train/success_rate`
- `train/avg_reward`
- `train/avg_episode_length`

validation set 使用固定局面，training set 从初始局面开始，用当前 policy 贪心下棋。它们用于观察训练性能、泛化到固定验证局面、以及 train/val gap。

后续版本还支持更严格的固定带标签局面评估。训练启动时会生成两组固定局面：

- `train_positions`: 从训练分布随机抽取但固定下来的局面。
- `valid_positions`: 用不同随机种子生成、不会参与训练更新的固定局面。

每个局面用 depth-limited negamax/heuristic solver 标注 reference action 和 reference value。新增指标：

- `train/position_policy_acc`
- `train/position_top3_acc`
- `train/position_value_mse`
- `train/position_policy_nll`
- `val/position_policy_acc`
- `val/position_top3_acc`
- `val/position_value_mse`
- `val/position_policy_nll`

这些指标比单纯 `success_rate` 更适合判断 policy 是否学到 reference move，以及 value head 是否拟合固定局面的 reference value。

### 4. combination reward 的信息量总和

实现位置：

- RLVR reward components: `src/ibrlvr/rewards.py`
- 信息量统计：`src/ibrlvr/metrics.py` 的 `reward_information`
- 训练聚合：`src/ibrlvr/cli/train.py`

当前有两个 reward source：

- `correctness`: 终局是否成功。
- `spec`: 终局赢家是否满足中心控制 verifier。

核心指标：

- `reward_info/H_correct_bits`: correctness reward 的 Bernoulli entropy。
- `reward_info/H_spec_bits`: spec reward 的 Bernoulli entropy。
- `reward_info/H_combined_bits`: combined reward 离散分布 entropy。
- `reward_info/H_sum_components_bits`: component entropy 之和。

这样可以比较：

- 单个 verifier 各自携带多少信息。
- combined reward 的 entropy 是否等于、低于或高于 component entropy 的简单和。
- reward components 是否高度相关，导致组合后信息量没有线性增加。

### 5. noise reward 的影响

实现位置：

- 噪声注入：`src/ibrlvr/rewards.py`
- noise configs: `configs/connect4_gpu_noise.yaml`
- ablation 队列：`scripts/run_gpu_ablation_matrix.sh`

噪声形式：

```text
reward_noise ~ Normal(0, sqrt(c * I))
noisy_reward = combined_reward + reward_noise
```

核心指标：

- `noise/c`: 当前噪声系数。
- `noise/std`: 当前噪声标准差。
- `reward/noisy`: 加噪后的 reward 均值。
- `loss/value`: value head 对 noisy target 的拟合损失。
- `val/success_rate`: 噪声是否伤害验证表现。
- `entropy/action_avg` 和 `entropy/seq_cumulative`: 噪声是否延缓或加速 entropy collapse。

当前正式队列会跑 `c=0.01/0.05/0.10`。

### 6. self-play vs not self-play

这里的差异定义为：两个玩家是否共享同一套当前参数。

实现位置：

- 训练主循环：`src/ibrlvr/cli/train.py`
- 配置：`self_play.shared_params`
- non-shared config: `configs/connect4_gpu_nonshared.yaml`

两种模式：

- `self_play.shared_params: true`: 标准 AlphaZero self-play。player 1 和 player -1 都用当前模型。
- `self_play.shared_params: false`: player 1 用当前模型，player -1 用 frozen opponent snapshot。只有当前模型控制的 positions 会进入训练 target。opponent 每 `opponent_update_interval` step 同步一次。

核心指标：

- `selfplay/shared_params`: `1.0` 表示 shared，`0.0` 表示 non-shared。
- `rollout/p`
- `entropy/*`
- `train/*`
- `val/*`
- `loss/*`

这个 ablation 用来观察参数共享是否影响搜索分布稳定性、训练收敛和验证表现。

## 如何启动和监控

快速 smoke test：

```bash
PYTHONPATH=src WANDB_MODE=disabled python tests/run_tests.py
PYTHONPATH=src WANDB_MODE=disabled python -m ibrlvr.cli.train --config configs/tictactoe_smoke.yaml
```

单个正式 baseline：

```bash
PYTHONPATH=src WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 \
python -m ibrlvr.cli.train --config configs/connect4_gpu_rollout32.yaml
```

4-GPU 正式 ablation：

```bash
WANDB_MODE=online STEPS=3000 EPISODES_PER_STEP=8 bash scripts/run_gpu_formal_training.sh
```

查看训练状态：

```bash
scripts/training_status.sh
tail -f runs/gpu_ablation_logs/launcher.log
nvidia-smi
```

导出汇总表：

```bash
python scripts/export_run_summary.py --runs-dir runs --out runs/summary.csv
```

## 配置说明

正式 GPU 配置的关键字段：

```yaml
env:
  name: connect4

mcts:
  num_rollouts: 32
  c_puct: 1.5
  temperature: 1.0

training:
  steps: 3000
  episodes_per_step: 8
  batch_size: 256
  eval_interval: 50
  checkpoint_interval: 500

reward:
  correctness_weight: 1.0
  format_weight: 0.25
  noise_c: 0.0

self_play:
  shared_params: true
  opponent_update_interval: 500
```

`run_gpu_ablation_matrix.sh` 会覆盖部分字段，比如 seed、run_name、noise_c、shared_params 和 steps。

## 输出文件如何读

每个 run 的目录类似：

```text
runs/connect4_baseline_rollout32_seed0_<timestamp>/
  config.json
  metrics.jsonl
  checkpoints/
  wandb/
```

`metrics.jsonl` 每行是一条 step-level JSON。典型字段包括：

- 搜索和信息：`rollout/p`, `info/H_p_bits`, `info/I_cumulative_bits`
- 行为 entropy：`entropy/action_avg`, `entropy/seq_cumulative`
- reward：`rlvr/*`, `reward/*`, `reward_info/*`, `noise/*`
- 性能：`train/*`, `val/*`
- 训练：`loss/policy`, `loss/value`, `loss/total`, `replay/size`
- self-play：`selfplay/shared_params`

## 当前实现的边界

- MCTS 是 Python 单进程实现，GPU 主要加速网络 forward 和 batch training，所以单 run GPU 利用率不会很高。当前通过 4-GPU 多 run 并行提高整体吞吐。
- `rollout/p` 是基于 MCTS leaf/value 的成功比例统计，不是独立随机 rollout 的真实胜率。
- Connect4 的 `spec` verifier 是一个可验证的中心控制规则，用于研究组合 reward 信息量；如果后续换成数学题、代码题或真实 RLVR 任务，可以把它替换成格式检查、单元测试、proof checker 等 verifier。
- 当前 validation set 是固定局面集合，不是大规模 holdout 数据集；如果要做更严格泛化，需要扩展 `validation_states` 或引入外部 position set。

## 最小修改指南

增加新的 RLVR verifier：

1. 在环境 terminal state 上提供可验证事实或 checker。
2. 修改 `src/ibrlvr/rewards.py`，增加 reward component。
3. 修改 `src/ibrlvr/metrics.py` 或训练聚合逻辑，记录新 component 的 entropy。
4. 在 config 中增加权重。
5. 在 W&B 中比较 `reward_info/*`、`val/*`、`entropy/*` 的变化。

增加新的环境：

1. 实现 `legal_actions`, `legal_mask`, `step`, `winner`, `is_terminal`, `terminal_value`, `observation`, `spec_reward`。
2. 在 `src/ibrlvr/envs/__init__.py` 注册 `make_initial_state` 和 `make_validation_states`。
3. 新增 config，确认 `PolicyValueNet` 的输入维度会从 observation shape 自动推断。
4. 先跑 smoke，再放入 GPU ablation 队列。
