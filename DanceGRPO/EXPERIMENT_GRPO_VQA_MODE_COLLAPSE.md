# GRPO + VQA Judge 实验报告：模式坍缩分析

## 实验概览

- **实验名称**: GRPO 微调 SD1.5 生成 "a giraffe and two flamingos"
- **日期**: 2026-05-27
- **GPU**: 1× NVIDIA H100 80GB
- **结果**: 训练至 epoch ~70 后发生模式坍缩（model collapse），模型完全丢失 "giraffe"，只生成 flamingo

---

## 1. 实验 Setting

### 基础模型与数据

| 组件 | 配置 |
|------|------|
| Base Model | `runwayml/stable-diffusion-v1-5` |
| LoRA | rank=64, alpha=128, target: to_q/to_k/to_v/to_out.0 |
| VLM Judge | `Qwen2.5-VL-7B-Instruct` (bf16, ~15GB VRAM) |
| 采样器 | DDIM, 50 steps, eta=1.0 (全随机) |
| CFG | guidance_scale=5.0 |

### 训练超参数

| 参数 | 值 |
|------|-----|
| num_epochs | 500 |
| num_generations | 16 (每个 prompt 每 epoch 生成 16 张图) |
| sample.batch_size | 1 |
| train.batch_size | 1 |
| gradient_accumulation_steps | 4 |
| 有效 batch size | 4 (每个 optimizer step 累积 4×49=196 步) |
| learning_rate | 1e-4 |
| clip_range (PPO) | 1e-4 |
| adv_clip_max | 5 |
| max_grad_norm | 1.0 |
| timestep_fraction | 1.0 (训练 49 个 timestep) |

### Prompt 与 VQA

- Prompt: `"a giraffe and two flamingos"`
- VQA 问题（4 题，全部答对才 reward=1）:

| # | 问题 | 正确答案 | 类型 |
|---|------|---------|------|
| Q1 | How many giraffes are in the image? | one / 1 | count |
| Q2 | Are there any giraffes in the image? | Yes | binary |
| Q3 | How many flamingos are in the image? | two / 2 | count |
| Q4 | Are there any flamingos in the image? | Yes | binary |

### 配置文件

`DanceGRPO/fastvideo/config_sd/single1.py`:
```python
def single1():
    config = base.get_config()
    config.run_name = "grpo_giraffe_flamingos"
    config.num_epochs = 500
    config.save_freq = 100
    config.lora_rank = 64
    config.num_generations = 16
    config.prompt_file = "/workspace/oyly/prompt_giraffe_flamingos.jsonl"
    config.vlm_model_id = "/workspace/models/Qwen2.5-VL-7B-Instruct"
    config.sample.batch_size = 1
    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = 4
    config.train.learning_rate = 1e-4
    config.train.clip_range = 1e-4
    config.train.adv_clip_max = 5
    return config
```

---

## 2. 涉及的代码文件

| 文件 | 作用 |
|------|------|
| `fastvideo/train_grpo_sd_vlm.py` | 主训练脚本（~656 行） |
| `fastvideo/config_sd/base.py` | 基础配置模板 |
| `fastvideo/config_sd/single1.py` | 本实验配置 |
| `fastvideo/models/stable_diffusion/pipeline_with_logprob.py` | DDIM 采样 + log_prob 追踪（来自 DDPO-pytorch） |
| `fastvideo/models/stable_diffusion/ddim_with_logprob.py` | 单步 DDIM + 高斯 log_prob 计算 |
| `prompt_giraffe_flamingos.jsonl` | Prompt + VQA 数据 |

### 关键代码模块

#### VLM Reward Model (`train_grpo_sd_vlm.py` lines 96-191)

```python
class VLMRewardModel:
    def _ask_vlm(self, pil, query, max_new_tokens=20):
        """发送单个问题到 VLM，只解码生成的 token"""
        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": query},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil],
                                return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False)
        # 关键: 只解码新生成的 token，不包含 prompt
        input_len = inputs['input_ids'].shape[1]
        answer = self.processor.decode(output[0][input_len:],
                                       skip_special_tokens=True).strip()
        return answer

    def _check_vqa_answer(self, vlm_answer, ground_truth):
        """检查 VLM 回答是否匹配 ground truth"""
        gt = ground_truth.lower().strip()
        ans = vlm_answer.lower().strip().rstrip('.')

        # Yes/No 问题: 数字 >0 视为 Yes, 0 视为 No
        if gt in ("yes", "no"):
            if ans.startswith(gt):
                return True
            first_token = ans.split()[0] if ans else ans
            if first_token.lstrip('-').isdigit():
                return (int(first_token) > 0) == (gt == "yes")
            return False

        # 数字问题: "1", "1 giraffe", "one" 都接受
        num_words = {"zero":0, "one":1, "two":2, "three":3, "four":4,
                     "five":5, "six":6, "seven":7, "eight":8, "nine":9, "ten":10}
        if gt in num_words:
            gt_num = num_words[gt]
            first_token = ans.split()[0] if ans else ans
            first_token = first_token.rstrip(',.;:!')
            if first_token.isdigit() and int(first_token) == gt_num:
                return True
            if gt in ans:
                return True
            return False

        return gt in ans

    def judge_batch(self, images, prompt, vqa_list=None):
        """批量评分: VQA 全对 → reward=1.0, 否则 0.0"""
        rewards = torch.zeros(len(images), device=self.device)
        for i, pil in enumerate(images):
            all_correct = True
            for question, ground_truth in vqa_list:
                query = (f'Question: {question}\n'
                         f'Answer with only the exact number, a single word, '
                         f'or "Yes"/"No".')
                answer = self._ask_vlm(pil, query, max_new_tokens=20)
                if not self._check_vqa_answer(answer, ground_truth):
                    all_correct = False
                    break
            rewards[i] = 1.0 if all_correct else 0.0  # ← 二值 reward
        return rewards
```

#### GRPO 训练循环核心 (`train_grpo_sd_vlm.py` lines 534-621)

```python
# 1. 优势计算: 每 16 张图一组做 z-score 归一化
n = len(samples["rewards"]) // config.num_generations  # = 1
for i in range(n):
    group_rewards = samples["rewards"][i*16 : (i+1)*16]
    group_mean = group_rewards.mean()
    group_std = group_rewards.std() + 1e-8
    advantages[i*16:(i+1)*16] = (group_rewards - group_mean) / group_std

# 2. PPO 损失
ratio = torch.exp(log_prob_new - log_prob_old)  # π_new / π_old
unclipped_loss = -advantages * ratio
clipped_loss = -advantages * clip(ratio, 1-ε, 1+ε)  # ε=1e-4
loss = mean(max(unclipped_loss, clipped_loss))

# 3. 只有 approx_kl 记录 (不加入 loss)
info["approx_kl"] = 0.5 * mean((log_prob_new - log_prob_old) ** 2)
```

**注意**: 原始 DanceGRPO 对 KL 只做追踪，不加入 loss 约束。这是 GRPO 「无需 critic」的一个设计选择，但也意味着模型可能无限漂离 base distribution。

---

## 3. 实验结果

### 3.1 训练曲线

实验跑了 100 个 epoch 后被手动停止（已完全坍缩，无意义继续）。

```
Epoch  0: reward=0.1250  p=0.1250  (2/16 正确)
Epoch 10: reward=0.0000  p=0.0000
Epoch 20: reward=0.3125  p=0.3125
Epoch 30: reward=0.2500  p=0.2500
Epoch 40: reward=0.1875  p=0.1875
Epoch 50: reward=0.3125  p=0.3125
Epoch 60: reward=0.0625  p=0.0625
Epoch 70: reward=0.0000  p=0.0000  ← 开始频繁出现全零
Epoch 78: reward=0.0625  p=0.0625  ← 最后一次非零
Epoch 79+: reward=0.0000 p=0.0000  ← 彻底坍缩，连续 20+ epoch 全零
```

**特征**: reward 始终在 0.0-0.375 之间波动，从未持续提升。约 epoch 70 后完全崩塌为 0。

### 3.2 生成图片的 VLM 诊断

每 20 epoch 保存 4×4 网格图，用同一 VLM（Qwen2.5-VL-7B）描述图片内容：

| Epoch | VLM 描述 | 分析 |
|-------|---------|------|
| 0 | "Giraffe: 1-5, Flamingo: 2-5" | 两种动物都在，数量随机 |
| 20 | "Giraffe: 1-2, Flamingo: 2-4" | 两者仍在，质量无明显提升 |
| 40 | "Flamingo: 2-6, Giraffe: 偶尔缺失" | 开始出现只有 flamingo 的图 |
| 60 | "Flamingo statues/birds, 少量有 giraffe" | 大部分丢失 giraffe，Flamingo 主导 |
| 80 | **"All images: flamingos only. No giraffes."** | 完全坍缩，只剩 flamingo |

### 3.3 坍缩时间线

```
Phase 1 (Epoch 0-40): 正常探索，p 在 0.0-0.375 间波动
                     图片多样，两种动物都存在

Phase 2 (Epoch 40-70): 漂移期，模型逐渐偏向 flamingo
                       giraffe 出现频率下降，样本多样性降低

Phase 3 (Epoch 70+):   完全坍缩
                       所有 16 张图都只有 flamingo
                       reward=0, advantage=0, 梯度为零
                       模型永远无法自行恢复
```

---

## 4. 坍缩原因分析

### 4.1 根本原因: 二值稀疏 Reward + 无约束 Policy Drift

这是最核心的问题。VQA reward 机制为**全部 4 题答对才得 1，否则得 0**。

GRPO 算法每 epoch 的处理流程:

```
16 张图 → VLM 评分 → reward ∈ {0, 1} (只有 1-5 个正样本)
  → Group z-score: 正样本 z≈+3.87, 负样本 z≈-0.26
  → PPO 更新: 正样本强正梯度, 负样本弱负梯度
  → 模型朝「幸运噪声种子」方向微调
```

**致命缺陷**: 15 个负样本的惩罚完全相同 (z≈-0.26), **不区分「差一点」(3/4 正确) 和「完全错误」(0/4 正确)**。

具体而言:
- 图片有 giraffe + flamingo 但数量错 (→ 2/4 答对, reward=0) → z ≈ -0.26
- 图片只有 flamingo (→ ~2/4 答对, reward=0) → z ≈ -0.26
- 图片什么都没有 (→ 0/4 答对, reward=0) → z ≈ -0.26

**梯度完全相同。** 模型丢失 giraffe 时得不到任何惩罚信号。

### 4.2 促成因素

**a) 无 KL 约束**

原版 DanceGRPO 不将 KL divergence 加入 loss:

```python
# 只记录，不约束
info["approx_kl"] = 0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
# 不参与 loss
```

LoRA 权重可以不受限制地漂离 `runwayml/stable-diffusion-v1-5` 的原始分布。一旦漂到只生成 flamingo 的区域，无法自行返回。

**b) PPO clip_range 过紧 (1e-4)**

`ratio = exp(logπ_new - logπ_old)` 被限制在 [0.9999, 1.0001]。模型每次更新幅度极小，但方向不受约束。小步慢漂 + 无 KL 约束 = 慢性坍缩。

**c) 随机噪声种子**

每 epoch 使用新的 `torch.randn` 作为初始 latent，eta=1.0 的 DDIM 加入随机噪声。正样本的出现高度依赖噪声种子的随机运气，模型学到的是「追逐幸运噪声」而非「稳定的生成策略」。

### 4.3 坍缩的不可逆性

一旦所有 16 张图都只生成 flamingo:
- 4 个 VQA 中，giraffe 相关的 2 题全错，flamingo 相关的 2 题可能对可能错
- 即使 flamingo 数量偶尔正确 (2/4), reward 仍为 0 (因为 giraffe 两题全错)
- 16 张 reward 全零 → group_mean=0, group_std≈0
- z-score = (0-0)/ε ≈ 0 → **所有样本的 advantage 均为零**
- loss = 0 → **梯度消失**

模型进入吸收态 (absorbing state)，永远出不来了。

### 4.4 与之前实验的对比

之前用同一个代码、相同超参数跑过 Exp1 (giraffe+flamingo) 和 Exp2 (lion+elephant), 跑了 400+ epoch 没有坍缩。

**差异在于 VQA judge**:
- **之前**: VQA judge 有 bug — 从 prompt 文本中提取了 "Yes" 和 "one" 作为答案 → 模型随便生成什么都可能被误判为正确 → reward 几乎总是 1 → 不存在坍缩
- **现在**: VQA judge 修好 — 模型必须真正生成正确的动物和数量 → reward 极稀疏 → 坍缩

### 4.5 为什么「只坍缩到 flamingo」而非 giraffe？

可能原因:
1. SD1.5 的 flamingo 视觉 prior 更强 (颜色鲜艳、形态独特)
2. 火烈鸟数量 "two" 比长颈鹿数量 "one" 更易被 VLM 识别
3. 或纯粹随机漂移 — 哪个方向先到吸收态就先坍缩到哪个

---

## 5. 可能的解决方案

### 方案 A: 部分分 Reward (最小改)

将 `judge_batch` 中的二值 reward 改为按比例给分:

```python
# 当前 (二值):
rewards[i] = 1.0 if all_correct else 0.0

# 改为 (连续):
rewards[i] = correct_count / len(vqa_list)  # 0.0, 0.25, 0.5, 0.75, 1.0
```

**效果**:
- 3/4 正确 → reward=0.75 vs 0/4 → reward=0.0 → z-score 差异显著 → 有梯度引导
- 模型丢失 giraffe 时，reward 从 0.5+ 降到 0-0.25 → 立即有负梯度信号
- 即使坍缩到只画 flamingo (reward≈0.25-0.5), advantage 非零 → 仍有梯度可以恢复

### 方案 B: 加入 KL Penalty

在 loss 中加入 KL 项:

```python
kl_penalty = 0.01 * approx_kl
loss = ppo_loss + kl_penalty
```

### 方案 C: 增大 num_generations

增加到 32 或 64 → 更多样本 → 正样本比例更稳定 → 不易坍缩。代价是每 epoch 计算量加倍。

### 方案 D: 降低 Learning Rate

1e-4 → 5e-6 或 1e-5 → 模型漂移更慢 → 更多时间在「非坍缩区」探索。但本质上不解决无约束漂移问题。

### 组合建议

**A + B**: 部分分 reward (提供梯度信号) + KL penalty (防止漂太远)。这是最可靠的方向。

---

## 6. 代码文件完整清单

### 服务器路径

| 路径 | 说明 |
|------|------|
| `/workspace/oyly/DanceGRPO/fastvideo/train_grpo_sd_vlm.py` | 主训练脚本 |
| `/workspace/oyly/DanceGRPO/fastvideo/config_sd/base.py` | 基础配置 |
| `/workspace/oyly/DanceGRPO/fastvideo/config_sd/single1.py` | Exp1 配置 |
| `/workspace/oyly/DanceGRPO/fastvideo/models/stable_diffusion/pipeline_with_logprob.py` | 采样 pipeline |
| `/workspace/oyly/DanceGRPO/fastvideo/models/stable_diffusion/ddim_with_logprob.py` | DDIM step + log_prob |
| `/workspace/oyly/prompt_giraffe_flamingos.jsonl` | Prompt + VQA 数据 |
| `/workspace/oyly/logs/exp1_v2.log` | 训练日志 |
| `/workspace/oyly/checkpoints/previews/grpo_giraffe_flamingos_*/` | 生成的预览图 |

### 关键函数索引 (`train_grpo_sd_vlm.py`)

| 函数/类 | 行号 | 功能 |
|---------|------|------|
| `PromptDataset` | 42-70 | 数据加载 |
| `VLMRewardModel._ask_vlm` | 107-121 | VLM 单次问答 |
| `VLMRewardModel._check_vqa_answer` | 123-156 | 答案匹配 |
| `VLMRewardModel.judge_batch` | 158-191 | 批量评分 |
| `main` (sampling) | 404-472 | DDIM 采样 + VLM 评分 |
| `main` (GRPO advantage) | 534-545 | 组内 z-score 优势计算 |
| `main` (training) | 549-633 | PPO loss + 梯度更新 |
