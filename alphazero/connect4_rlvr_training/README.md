# Information Bottleneck AlphaZero Experiments

中文说明见 [README.zh.md](README.zh.md)。

This repository implements a compact AlphaZero-style training stack with the
diagnostics needed for rollout correctness, information accumulation, action
entropy, train/validation performance, reward-combination information, reward
noise, and shared vs non-shared self-play.

The reward path is RLVR-style: rewards come from deterministic verifiers rather
than a learned reward model. For Connect4, `correctness` verifies terminal
non-draw success and `spec` verifies a center-control rule for the terminal
winner; the configured weighted combination is the scalar value target, with
optional Gaussian noise scaled by cumulative information.

## Quickstart

CPU smoke test:

```bash
WANDB_MODE=disabled python -m ibrlvr.cli.train --config configs/tictactoe_smoke.yaml
```

Formal rollout-32 baseline:

```bash
WANDB_MODE=offline python -m ibrlvr.cli.train --config configs/connect4_gpu_rollout32.yaml
```

4x4090 formal ablation matrix:

```bash
WANDB_MODE=offline bash scripts/run_gpu_formal_training.sh
scripts/training_status.sh
```

The formal configs use standard 6x7 Connect4 with rollout=32. TicTacToe and
Connect4-small configs are kept for smoke tests and quick debugging only.

The run writes local metrics to `runs/<run_id>/metrics.jsonl` and checkpoints to
`runs/<run_id>/checkpoints`. If `wandb` is enabled, the same metrics are logged
to the configured project.

## Core Metrics

- `rollout/p`: success probability over `num_rollouts` MCTS simulations.
- `info/H_p_bits` and `info/I_cumulative_bits`: binary entropy of rollout
  correctness and its cumulative sum over train steps.
- `entropy/action_avg` and `entropy/seq_cumulative`: entropy over action-space
  distributions along generated sequences.
- `train/*` and `val/*`: performance on generated training episodes and fixed
  validation positions.
- `reward_info/*`: information in correctness reward, format/spec reward,
  combined reward, and component sum.
- `rlvr/*`: raw verifier component means and the combined verifiable reward.
- `noise/*`: injected Gaussian reward noise diagnostics.
- `selfplay/*`: whether actor and opponent share parameters.

## Notes

- `self_play.shared_params=true` is standard AlphaZero-style self-play.
- `self_play.shared_params=false` uses the current model for player `1` and a
  frozen opponent snapshot for player `-1`; only current-model positions update
  the current model.
- `reward.noise_c` injects Gaussian reward noise with standard deviation derived
  from the cumulative information integral.
