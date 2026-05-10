# Outcome Entropy RL

研究目标：验证**稀疏奖励真正的瓶颈是 outcome 分布塌缩（低熵）**，而不只是 sparse reward 本身。

## 1) 两类实验线

### A. 经典 RL（Self-play + Static Sparse）
- **Self-play**: OpenSpiel（如 TicTacToe / ConnectFour）
- **Static sparse**: MiniGrid + SB3（PPO/A2C）
- 关键指标：`success/win/draw/loss` 分布、`outcome_entropy`

### B. LLM RL（GRPO / RLVR）
- 离线 group 采样 + verifier 统计 mixed/all-fail/all-correct 比例
- 在线训练框架建议：
  - **TRL GRPOTrainer**（本仓已给 skeleton）
  - **VERL**（推荐作为更工程化的大规模 LLM RL 训练框架进行对照验证）
- 关键指标：`mixed_group_ratio`、`avg_group_entropy`、`effective_update_fraction`

---

## 2) 环境安装

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

或：

```bash
bash scripts/setup_env.sh
```

> 备注：CPU-only 机器若 `bitsandbytes` 安装失败，可从 `requirements.txt` 去掉后重装。

---

## 3) 怎么运行

### 3.1 一键 smoke tests
```bash
bash scripts/run_all_smoke_tests.sh
```

### 3.2 Self-play 统计（OpenSpiel）
```bash
python src/selfplay/run_openspiel_selfplay.py --game tic_tac_toe --episodes 500 --out logs/selfplay/random_baseline.csv
```

### 3.3 LLM group outcome 统计（离线）
```bash
python src/llm/offline_sampling.py --num_prompts 200 --group_size 8 --out logs/llm/offline_groups.csv
python src/llm/analyze_groups.py --csv logs/llm/offline_groups.csv
```

---

## 4) 目录

```text
src/common/      # 熵与日志工具
src/selfplay/    # OpenSpiel
src/sparse/      # MiniGrid + SB3
src/llm/         # offline sampling / verifier / GRPO skeleton
scripts/         # 安装与 smoke tests
logs/, results/  # 输出目录
```
