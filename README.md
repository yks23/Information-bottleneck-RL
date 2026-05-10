# Outcome Entropy RL

一个可直接执行的研究脚手架：验证 **稀疏奖励学习的瓶颈不是 sparse 本身，而是 outcome 分布塌缩（低熵）**。

覆盖三条实验线：
1. **Self-play sparse reward**（OpenSpiel）
2. **Static sparse reward**（MiniGrid/Gymnasium + SB3）
3. **LLM RL / GRPO / RLVR**（先做离线 group outcome 分析）

---

## 1) 项目结构

```text
configs/
  selfplay/
  sparse/
  llm/
src/
  common/
    entropy.py          # 离散熵/二元熵/group 熵 + group_stats
    logging_utils.py    # JSONL/CSV 日志
    plotting.py         # 基础画图
    metrics.py
  selfplay/
    smoke_test_openspiel.py
    run_openspiel_selfplay.py
  sparse/
    smoke_test_minigrid.py
    run_minigrid_sb3.py
    eval_sparse.py
    curriculum.py
  llm/
    offline_sampling.py
    reward_functions.py
    analyze_groups.py
    run_grpo.py
scripts/
  setup_env.sh
  run_all_smoke_tests.sh
```

---

## 2) 环境配置（按顺序执行）

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

或直接：

```bash
bash scripts/setup_env.sh
```

> 如果是 CPU-only 机器，`bitsandbytes` 可能失败，可从 `requirements.txt` 删除后重装。

---

## 3) 一键 smoke test

```bash
bash scripts/run_all_smoke_tests.sh
```

该脚本会依次运行：
- OpenSpiel TicTacToe 终局回报检查
- MiniGrid + PPO 最小训练
- LLM 离线 group 采样（dummy verifier）

---

## 4) 如何“看到统计信息量”（Outcome Entropy）

### 4.1 Self-play 统计

```bash
python src/selfplay/run_openspiel_selfplay.py --game tic_tac_toe --episodes 500 --out logs/selfplay/random_baseline.csv
```

输出字段：
- `p_win`, `p_draw`, `p_loss`
- `outcome_entropy`
- `eval_win_rate`

### 4.2 LLM group-relative 统计

```bash
python src/llm/offline_sampling.py --num_prompts 200 --group_size 8 --out logs/llm/offline_groups.csv
python -c "from src.llm.analyze_groups import summarize_group_types; print(summarize_group_types('logs/llm/offline_groups.csv'))"
```

可快速观察：
- `all_fail / mixed / all_correct` 比例
- mixed 比例越高，GRPO 相对优势信号越可能有效

### 4.3 核心熵函数

`src/common/entropy.py` 提供：
- `discrete_entropy(values)`
- `binary_entropy_from_successes(successes)`
- `group_binary_entropy(rewards)`
- `group_stats(rewards)`

---

## 5) 当前里程碑状态（Week 1）

已完成：
- 目录结构与依赖脚手架
- 三条线 smoke test / skeleton
- 共享熵统计和日志工具

下一步建议：
- 在 `run_minigrid_sb3.py` 接入定期 eval callback，落地 `success_rate + outcome_entropy` 曲线
- 在 `run_grpo.py` 接入 TRL `GRPOTrainer` 并记录 `mixed_group_ratio`
- 增加 `results/figures/` 自动出图脚本
