"""
Single-Question Diffusion RLVR Experiment
==========================================
Task: Generate "a red square" via DDPO
Metrics:
  - reward: rule-based red square score
  - H(p): batch μ variance (policy entropy proxy)
  - I = ∑H: cumulative information
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from peft import get_peft_model, LoraConfig
from PIL import Image

from diffusers import DDIMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from dataclasses import dataclass, field
from typing import Callable, Optional


# ============================================================
# 1. Single Prompt
# ============================================================
SINGLE_PROMPT = "a red square"


# ============================================================
# 2. Rule-Based Reward: arbitrary red square (any position/size)
# ============================================================
def _detect_red_squares(arr):
    """
    Detect any red square region in the image.
    Returns list of (score, bb_h, bb_w, min_y, max_y, min_x, max_x) for each detected region.
    """
    h, w = arr.shape[:2]
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    is_red = (r > 0.5) & (g < 0.35) & (b < 0.35)

    visited = np.zeros_like(is_red, dtype=bool)
    regions = []

    for y in range(h):
        for x in range(w):
            if visited[y, x] or not is_red[y, x]:
                continue
            component = []
            queue = [(y, x)]
            min_y, max_y, min_x, max_x = y, y, x, x

            while queue:
                cy, cx = queue.pop()
                if visited[cy, cx]:
                    continue
                visited[cy, cx] = True
                if not is_red[cy, cx]:
                    continue
                component.append((cy, cx))
                min_y, max_y = min(min_y, cy), max(max_y, cy)
                min_x, max_x = min(min_x, cx), max(max_x, cx)
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = cy+dy, cx+dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                        queue.append((ny, nx))

            if not component:
                continue
            bb_h = max_y - min_y + 1
            bb_w = max_x - min_x + 1
            area = len(component)
            bbox_area = bb_h * bb_w
            fill_ratio = area / max(bbox_area, 1)
            aspect_ratio = bb_w / max(bb_h, 1)
            squareness = max(0.0, 1.0 - abs(1.0 - aspect_ratio))
            pixels = np.array([arr[cy, cx] for cy, cx in component])
            red_score = float(pixels[:, 0].mean() - pixels[:, 1].mean() - pixels[:, 2].mean())
            if bbox_area < (h * w * 0.005):
                continue
            score = red_score * 0.5 + squareness * 0.3 + fill_ratio * 0.2
            regions.append((score, bb_h, bb_w, min_y, max_y, min_x, max_x))

    return regions


def compute_reward(images, binary=False):
    """
    Reward = best detected red square anywhere in the image.
    Uses connected-component detection + squareness + border darkness.
    If binary=True, threshold at 0.5 → {0.0, 1.0}.
    """
    rewards = []
    for img in images:
        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]

        r, g, b = arr[:, :, 0]/255.0, arr[:, :, 1]/255.0, arr[:, :, 2]/255.0
        is_bg = (r < 0.15) & (g < 0.15) & (b < 0.15)
        bg_ratio = is_bg.sum() / (h * w)
        bg_penalty = 0.1 * bg_ratio

        regions = _detect_red_squares(arr / 255.0)

        if regions:
            best_score, bb_h, bb_w, min_y, max_y, min_x, max_x = max(regions, key=lambda x: x[0])
            border_size = max(2, min(bb_h, bb_w) // 4)
            border_y1, border_y2 = max(0, min_y - border_size), min(h, max_y + border_size + 1)
            border_x1, border_x2 = max(0, min_x - border_size), min(w, max_x + border_size + 1)
            border = arr[border_y1:border_y2, border_x1:border_x2].reshape(-1, 3)
            border_dark = ((border < 40).all(axis=1)).mean()
            reward = best_score * 0.7 + border_dark * 0.3 + bg_penalty
        else:
            reward = -0.5

        rewards.append(float(np.clip(reward, -1.0, 1.0)))
    if binary:
        rewards = [1.0 if r > 0.5 else 0.0 for r in rewards]
    return rewards


# ============================================================
# 3. Entropy H(p) based on correctness rate
# ============================================================
REWARD_THRESHOLD = 0.5  # reward > threshold counts as "correct"


class EntropyTracker:
    """
    Tracks H(p) = binary entropy of correctness rate p per epoch.
    p = fraction of rollouts with reward > threshold.
    H(p) = -(p log2 p + (1-p) log2 (1-p))
    I = ∑ H(p_i)
    """
    def __init__(self):
        self.history = []
        self.p_history = []
        self.step = 0

    def update(self, rewards, threshold=REWARD_THRESHOLD):
        correct = sum(1 for r in rewards if r > threshold)
        p = correct / len(rewards)
        if p == 0.0 or p == 1.0:
            h = 0.0
        else:
            h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        self.history.append(h)
        self.p_history.append(p)
        self.step += 1
        return h, p

    def compute_integral_I(self):
        return np.cumsum(self.history)

    def summary(self):
        cumsum = self.compute_integral_I()
        return {
            "H_final": self.history[-1] if self.history else 0.0,
            "H_max": max(self.history) if self.history else 0.0,
            "H_initial": self.history[0] if self.history else 0.0,
            "I_final": cumsum[-1] if len(cumsum) > 0 else 0.0,
            "I_steps": cumsum.tolist(),
            "H_per_step": self.history,
            "p_per_step": self.p_history,
        }


# ============================================================
# 4. Standard UNet2DConditionModel
# ============================================================


# ============================================================
# 5. DDPO Trainer (simplified, single-file, no trl dependency)
# ============================================================
@dataclass
class DDPOConfig:
    num_epochs: int = 50
    sample_batch_size: int = 16
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lr: float = 1e-4
    unet_lora_rank: int = 4
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    tracker_project_name: str = "diffusion-rlvr"
    log_with: Optional[str] = "wandb"
    run_name: str = "single_question"
    seed: int = 42
    ckpt_dir: str = "/workspace/oyly/checkpoints"
    ckpt_every: int = 50
    binary_reward: bool = False


class MinimalDDPOTrainer:
    """
    Minimal DDPO implementation for single-prompt RLVR.
    Based on: https://github.com/nicklashansen/ddpo-pytorch
    """

    def __init__(
        self,
        config: DDPOConfig,
        reward_fn: Callable,
        entropy_tracker: EntropyTracker,
        model_id: str = "/workspace/models/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.config = config
        self.reward_fn = reward_fn
        self.entropy_tracker = entropy_tracker
        self.device = device

        print(f"[Init] Loading SD1.5 from {model_id} ...")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", use_safetensors=False).to(device)
        self.noise_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(self.config.num_inference_steps)

        # Freeze most params
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        # Only train LoRA params on UNet
        self.unet_lora_params = self._setup_lora(rank=config.unet_lora_rank)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.unet_lora_params),
            lr=config.lr,
            weight_decay=0.0,
        )
        print("[Init] Done.")

    def _setup_lora(self, rank=4):
        """Attach LoRA layers to UNet attention blocks."""
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            inference_mode=False,
        )
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.print_trainable_parameters()
        return self.unet.parameters()

    def save_checkpoint(self, epoch):
        ckpt_path = os.path.join(self.config.ckpt_dir, f"{self.config.run_name}_ep{epoch}")
        os.makedirs(ckpt_path, exist_ok=True)
        self.unet.save_pretrained(ckpt_path)
        print(f"[Checkpoint] saved LoRA → {ckpt_path}")

    def _ddim_step_stochastic(self, noise_pred, t, prev_t, sample, alphas_cumprod, eta=1.0, variance_noise=None):
        """
        Manual DDIM step with eta (η). Returns (prev_sample, mean, std).
        alphas_cumprod passed in to support cross-device calls.
        """
        alpha_prod_t = alphas_cumprod[t]
        alpha_prod_t_prev = alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=sample.device)
        beta_prod_t = 1 - alpha_prod_t

        pred_x0 = (sample - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std = eta * variance.clamp(min=1e-20).sqrt()
        pred_dir = (1 - alpha_prod_t_prev - std**2).clamp(min=0.0).sqrt() * noise_pred
        mean = alpha_prod_t_prev.sqrt() * pred_x0 + pred_dir

        if variance_noise is None:
            variance_noise = torch.randn_like(sample)
        prev_sample = mean + std * variance_noise
        return prev_sample, mean, std

    @torch.no_grad()
    def _rollout_one_gpu(self, mb_prompts, device, eta=1.0):
        """Rollout a single micro-batch on a specific GPU."""
        alphas = self.noise_scheduler.alphas_cumprod.to(device)
        text_inputs = self.tokenizer(
            mb_prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_emb = self.text_encoder(text_inputs.input_ids.to(device))[0]
        uncond_emb = self.text_encoder(
            self.tokenizer([""] * len(mb_prompts), padding="max_length",
                           max_length=self.tokenizer.model_max_length,
                           return_tensors="pt").input_ids.to(device)
        )[0]

        latents = torch.randn(
            (len(mb_prompts), self.unet.config.in_channels, 64, 64),
            device=device, dtype=torch.float32,
        )

        mb_trajectory = []
        all_timesteps = list(self.noise_scheduler.timesteps)
        # Sample 5 random steps to store (reduces CPU memory from 50 to 5)
        save_steps = sorted(np.random.choice(len(all_timesteps), size=min(5, len(all_timesteps)), replace=False))

        for i, t in enumerate(all_timesteps):
            prev_t = all_timesteps[i + 1].item() if i + 1 < len(all_timesteps) else -1

            noise_cond = self.unet(latents, t, encoder_hidden_states=text_emb).sample
            noise_uncond = self.unet(latents, t, encoder_hidden_states=uncond_emb).sample
            noise_pred = noise_uncond + self.config.guidance_scale * (noise_cond - noise_uncond)

            prev_sample, _mean, _std = self._ddim_step_stochastic(
                noise_pred, int(t), int(prev_t), latents, alphas, eta=eta,
            )

            if i in save_steps:
                mb_trajectory.append({
                    "latent": latents.clone().cpu(),
                    "t": int(t),
                    "prev_t": int(prev_t),
                    "prev_sample": prev_sample.clone().cpu(),
                })

            latents = prev_sample

        latents_dec = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents_dec.to(self.vae.dtype)).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        imgs_np = (imgs.cpu().float().permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
        return [Image.fromarray(img) for img in imgs_np], mb_trajectory

    @torch.no_grad()
    def _rollout(self, prompts, num_inference_steps=50, micro_batch=16, eta=1.0):
        """
        Collect rollouts from current policy (no grad), using stochastic DDIM (eta=1).
        """
        batch_size = len(prompts)
        all_images = []
        all_trajectories = []

        for start in range(0, batch_size, micro_batch):
            end = min(start + micro_batch, batch_size)
            mb_prompts = prompts[start:end]
            imgs, traj = self._rollout_one_gpu(mb_prompts, self.device, eta=eta)
            all_images.extend(imgs)
            all_trajectories.append(traj)

        return all_images, all_trajectories, None

    def _compute_advantages(self, rewards):
        """GRPO-style group normalization within batch."""
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        if r.std() > 1e-8:
            return ((r - r.mean()) / (r.std() + 1e-8)).tolist()
        return (r - r.mean()).tolist()

    def _ddim_step_for_training(self, noise_pred, t, prev_t, sample, eta=1.0):
        """Compute mean and std for training (on correct device)."""
        alphas = self.noise_scheduler.alphas_cumprod.to(sample.device)
        _prev, mean, std = self._ddim_step_stochastic(
            noise_pred, int(t), int(prev_t), sample, alphas, eta=eta,
        )
        return mean, std

    def training_step(self, prompt, epoch):
        """
        DDPO/REINFORCE training step:
          1. Rollout (no grad) → images + trajectory
          2. Compute rewards
          3. Re-run UNet WITH grad on stored states → REINFORCE loss
          4. Gradient step
        """
        self.unet.train()

        # --- Phase 1: Rollout (no grad) ---
        prompts = [prompt] * self.config.sample_batch_size
        images, trajectory, text_emb = self._rollout(
            prompts, self.config.num_inference_steps
        )

        # --- Phase 2: Reward + Entropy ---
        rewards = self.reward_fn(images, binary=self.config.binary_reward)
        entropy_val, p_correct = self.entropy_tracker.update(rewards)
        advantages = self._compute_advantages(rewards)

        # --- Phase 3: REINFORCE loss (proper DDPO Gaussian log_prob) ---
        self.optimizer.zero_grad()
        total_loss = 0.0
        total_log_prob = 0.0

        num_mb = len(trajectory)
        n_mb = min(4, num_mb)
        n_step = 3
        for _ in range(n_mb):
            mb_idx = np.random.randint(0, num_mb)
            mb_traj = trajectory[mb_idx]
            num_steps = len(mb_traj)
            mb_size = mb_traj[0]["latent"].shape[0]

            mb_prompts = [prompt] * mb_size
            mb_text_inputs = self.tokenizer(
                mb_prompts, padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            mb_text_emb = self.text_encoder(mb_text_inputs.input_ids.to(self.device))[0]
            mb_uncond_emb = self.text_encoder(
                self.tokenizer([""] * mb_size, padding="max_length",
                               max_length=self.tokenizer.model_max_length,
                               return_tensors="pt").input_ids.to(self.device)
            )[0]

            step_indices = np.random.choice(num_steps, size=min(n_step, num_steps), replace=False)
            for idx in step_indices:
                step = mb_traj[idx]
                latent = step["latent"].to(self.device)
                t = step["t"]
                prev_t = step["prev_t"]
                prev_sample = step["prev_sample"].to(self.device)

                # Re-run UNet with grad and rebuild CFG noise (matches rollout)
                noise_cond = self.unet(latent, t, encoder_hidden_states=mb_text_emb).sample
                noise_uncond = self.unet(latent, t, encoder_hidden_states=mb_uncond_emb).sample
                noise_pred = noise_uncond + self.config.guidance_scale * (noise_cond - noise_uncond)

                # Compute mean & std under new policy (on correct device)
                mean, std = self._ddim_step_for_training(
                    noise_pred, t, prev_t, latent, eta=1.0,
                )

                # Gaussian log_prob: -((x - μ)^2 / 2σ^2) summed over latent dims
                log_prob = -((prev_sample - mean) ** 2) / (2 * std ** 2 + 1e-8)
                log_prob = log_prob.mean(dim=[1, 2, 3])  # per-sample scalar

                mb_start = mb_idx * mb_size
                mb_end = min(mb_start + mb_size, len(advantages))
                adv_tensor = torch.tensor(advantages[mb_start:mb_end], device=self.device, dtype=torch.float32)

                loss = -(adv_tensor * log_prob).mean()
                (loss / (n_mb * n_step)).backward()
                total_loss += loss.item()
                total_log_prob += log_prob.mean().item()

        self.optimizer.step()

        return {
            "reward": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "entropy_H": entropy_val,
            "p_correct": p_correct,
            "loss": total_loss / (n_mb * n_step),
            "log_prob": total_log_prob / (n_mb * n_step),
        }

    def train(self):
        """Main training loop."""
        if self.config.log_with == "wandb":
            wandb.init(
                project=self.config.tracker_project_name,
                name=self.config.run_name,
                config=vars(self.config),
            )

        for epoch in range(self.config.num_epochs):
            metrics = self.training_step(SINGLE_PROMPT, epoch)

            # Compute I = cumulative sum of H
            I_val = self.entropy_tracker.compute_integral_I()
            metrics["I_cumulative"] = I_val[-1] if len(I_val) > 0 else 0.0

            if (epoch + 1) % 5 == 0:
                summary = self.entropy_tracker.summary()
                metrics.update({
                    "I_final": summary["I_final"],
                    "H_initial": summary["H_initial"],
                    "H_max": summary["H_max"],
                })

            if self.config.log_with == "wandb":
                wandb.log(metrics, step=epoch)

            if (epoch + 1) % 10 == 0:
                print(f"[Epoch {epoch+1:03d}] reward={metrics['reward']:.4f} ± {metrics['reward_std']:.4f}  "
                      f"p={metrics['p_correct']:.4f}  H={metrics['entropy_H']:.6f}  I={metrics['I_cumulative']:.4f}")

            if (epoch + 1) % self.config.ckpt_every == 0:
                self.save_checkpoint(epoch + 1)

        if self.config.log_with == "wandb":
            wandb.finish()

        print("\n=== Final Summary ===")
        summary = self.entropy_tracker.summary()
        print(f"  I_final = {summary['I_final']:.4f}")
        print(f"  H_initial = {summary['H_initial']:.6f}")
        print(f"  H_final = {summary['H_final']:.6f}")
        print(f"  H_max = {summary['H_max']:.6f}")
        print("  I steps:", [f"{x:.4f}" for x in summary["I_steps"][::max(1, len(summary['I_steps'])//10)]])

        return summary


# ============================================================
# 6. Main
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="/workspace/models/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"),
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--sample_batch_size", type=int, default=16)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default="single_question")
    parser.add_argument("--ckpt_dir", type=str, default="/workspace/oyly/checkpoints")
    parser.add_argument("--ckpt_every", type=int, default=50)
    parser.add_argument("--binary_reward", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = DDPOConfig(
        num_epochs=args.num_epochs,
        sample_batch_size=args.sample_batch_size,
        unet_lora_rank=args.lora_rank,
        lr=args.lr,
        seed=args.seed,
        log_with=None if args.no_wandb else "wandb",
        run_name=args.run_name,
        ckpt_dir=args.ckpt_dir,
        ckpt_every=args.ckpt_every,
        binary_reward=args.binary_reward,
    )

    entropy_tracker = EntropyTracker()

    trainer = MinimalDDPOTrainer(
        config=config,
        reward_fn=compute_reward,
        entropy_tracker=entropy_tracker,
        model_id=args.model_id,
    )

    summary = trainer.train()
