# Based on train_grpo_sd_vlm.py — adds LoRA checkpoint resume support
# Usage: python -m fastvideo.train_grpo_sd_vlm_resume --config fastvideo/config_sd/resume1.py
#
# New config fields:
#   config.lora_checkpoint = "/path/to/checkpoint_dir"  (contains adapter_config.json + adapter_model.safetensors)
#   config.start_epoch = 400  (epoch to resume from; if 0, auto-parsed from checkpoint dir name)

from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from peft import get_peft_model, LoraConfig, PeftModel, set_peft_model_state_dict
import numpy as np
from fastvideo.models.stable_diffusion.pipeline_with_logprob import pipeline_with_logprob
from fastvideo.models.stable_diffusion.ddim_with_logprob import ddim_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
import torch.distributed as dist
import functools
import random
import itertools
import re
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from safetensors.torch import save_file
import json

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

SD1_5_PATH = "/workspace/models/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
VLM_MODEL_ID = "/workspace/models/Qwen2.5-VL-7B-Instruct"


class PromptDataset(Dataset):
    def __init__(self, file_path):
        self.prompts = []
        self.vqa_map = {}
        if file_path.endswith('.jsonl'):
            with open(file_path, "r", encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        p_text = item["prompt"]
                        self.prompts.append(p_text)
                        self.vqa_map[p_text] = item.get("vqa_list", None)
        elif file_path.endswith('.json'):
            with open(file_path, "r", encoding='utf-8') as f:
                self.prompts = json.load(f)
            for p in self.prompts:
                self.vqa_map[p] = None
        else:
            with open(file_path, "r", encoding='utf-8') as f:
                self.prompts = [line.strip() for line in f if line.strip()]
            for p in self.prompts:
                self.vqa_map[p] = None

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        return {"prompt": prompt, "vqa_list": self.vqa_map.get(prompt)}


class RewardTracker:
    def __init__(self):
        self.reward_mean_history = []
        self.p_history = []
        self.entropy_history = []
        self.variance_history = []

    def update(self, rewards):
        mean_reward = np.mean(rewards)
        variance = np.var(rewards)
        p = mean_reward
        if p == 0.0 or p == 1.0:
            h = 0.0
        else:
            h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        self.reward_mean_history.append(mean_reward)
        self.p_history.append(p)
        self.entropy_history.append(h)
        self.variance_history.append(variance)
        return mean_reward, p, h, variance

    @property
    def entropy_cumulative(self):
        return float(np.sum(self.entropy_history)) if self.entropy_history else 0.0

    @property
    def variance_cumulative(self):
        return float(np.sum(self.variance_history)) if self.variance_history else 0.0


class VLMRewardModel:
    def __init__(self, model_id, device="cuda"):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        print(f"[VLM Judge] Loading {model_id} ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="eager",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = device
        print("[VLM Judge] Ready.")

    def _ask_vlm(self, pil, query, max_new_tokens=20):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": query},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=self.processor.tokenizer.pad_token_id)
        input_len = inputs['input_ids'].shape[1]
        answer = self.processor.decode(output[0][input_len:], skip_special_tokens=True).strip()
        return answer

    def _check_vqa_answer(self, vlm_answer, ground_truth):
        gt = ground_truth.lower().strip()
        ans = vlm_answer.lower().strip().rstrip('.')

        num_words = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }

        if gt in ("yes", "no"):
            if ans.startswith(gt):
                return True
            first_token = ans.split()[0] if ans else ans
            if first_token.lstrip('-').isdigit():
                return (int(first_token) > 0) == (gt == "yes")
            return False

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

    @torch.no_grad()
    def judge_batch(self, images, prompt, vqa_list=None):
        rewards = torch.zeros(len(images), device=self.device)

        if vqa_list is None:
            query = (
                f'Examine this image critically. Does it accurately contain: "{prompt}"?\n'
                f'If ANY element is missing, wrong color, wrong count, or incorrect position, answer NO.\n'
                f'Answer ONLY "YES" if ALL elements are correct and clearly visible.'
            )
            for i, pil in enumerate(images):
                answer = self._ask_vlm(pil, query, max_new_tokens=10)
                rewards[i] = 1.0 if "YES" in answer.upper() else 0.0
            return rewards

        for i, pil in enumerate(images):
            correct_count = 0
            for question, ground_truth in vqa_list:
                query = (
                    f'Question: {question}\n'
                    f'Answer with only the exact number, a single word, or "Yes"/"No".'
                )
                answer = self._ask_vlm(pil, query, max_new_tokens=20)
                if self._check_vqa_answer(answer, ground_truth):
                    correct_count += 1
            rewards[i] = correct_count / len(vqa_list)

        return rewards


def _parse_epoch_from_dir(ckpt_dir):
    """Extract epoch number from checkpoint dir name like '..._ep400'."""
    m = re.search(r'_ep(\d+)$', os.path.basename(ckpt_dir.rstrip('/')))
    if m:
        return int(m.group(1))
    return 0


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "fastvideo/config_sd/base.py", "Training configuration.")

logger = get_logger(__name__)


def main(_):
    config = FLAGS.config

    lora_checkpoint = getattr(config, 'lora_checkpoint', '')
    start_epoch = getattr(config, 'start_epoch', 0)

    if lora_checkpoint and start_epoch == 0:
        start_epoch = _parse_epoch_from_dir(lora_checkpoint) + 1
        print(f"[Resume] Auto-detected start_epoch={start_epoch} from {lora_checkpoint}")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    num_train_timesteps = int((config.sample.num_steps - 1) * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="grpo-sd-vlm",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )

    logger.info(f"\n{config}")
    set_seed(config.seed, device_specific=True)

    vlm_reward_model = VLMRewardModel(VLM_MODEL_ID, device=accelerator.device)

    pipeline = StableDiffusionPipeline.from_pretrained(
        SD1_5_PATH, local_files_only=True,
    )
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1, disable=not accelerator.is_local_main_process,
        leave=False, desc="Timestep", dynamic_ncols=True,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    # ── LoRA: either load from checkpoint or create fresh ──
    if config.use_lora:
        if lora_checkpoint:
            print(f"[Resume] Loading LoRA adapter from {lora_checkpoint}")
            pipeline.unet = PeftModel.from_pretrained(
                pipeline.unet, lora_checkpoint, is_trainable=True,
            )
            print(f"[Resume] LoRA adapter loaded. Resuming from epoch {start_epoch}.")
        else:
            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_rank * 2,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                inference_mode=False,
            )
            pipeline.unet = get_peft_model(pipeline.unet, lora_config)
        pipeline.unet.print_trainable_parameters()
        unet = pipeline.unet
    else:
        unet = pipeline.unet

    # ── Checkpoint hooks ──
    if config.use_lora:
        def save_model_hook(models, weights, output_dir):
            peft_model = accelerator.unwrap_model(pipeline.unet)
            peft_model.save_pretrained(output_dir)
            weights.pop()

        def load_model_hook(models, input_dir):
            tmp_unet = UNet2DConditionModel.from_pretrained(SD1_5_PATH, subfolder="unet", local_files_only=True)
            tmp_unet = PeftModel.from_pretrained(tmp_unet, input_dir)
            models[0].load_state_dict(tmp_unet.state_dict())
            del tmp_unet
            models.pop()
    else:
        def save_model_hook(models, weights, output_dir):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
            weights.pop()

        def load_model_hook(models, input_dir):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
            models.pop()

    def gather_tensor(tensor):
        if not dist.is_initialized():
            return tensor
        world_size = dist.get_world_size()
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, tensor)
        return torch.cat(gathered_tensors, dim=0)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam.")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    reward_tracker = RewardTracker()

    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""], return_tensors="pt", padding="max_length",
            truncation=True, max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    unet, optimizer = accelerator.prepare(unet, optimizer)
    executor = futures.ThreadPoolExecutor(max_workers=2)

    samples_per_epoch = config.sample.batch_size * accelerator.num_processes * config.num_generations
    total_train_batch_size = config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    if lora_checkpoint:
        logger.info(f"  Resumed from LoRA checkpoint: {lora_checkpoint}")
        logger.info(f"  Starting at epoch {start_epoch}")
    logger.info(f"  Num Epochs = {config.num_epochs} (will run epochs {start_epoch} to {config.num_epochs - 1})")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Total samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size = {total_train_batch_size}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0

    dataset = PromptDataset(config.prompt_file)

    def identity_collate(batch):
        return batch

    if dist.is_initialized():
        sampler = DistributedSampler(
            dataset, num_replicas=dist.get_world_size(),
            rank=dist.get_rank(), seed=123543, shuffle=config.sample.shuffle_dataset,
        )
        loader = DataLoader(dataset, batch_size=1, sampler=sampler,
                            pin_memory=True, drop_last=True, collate_fn=identity_collate)
    else:
        loader = DataLoader(dataset, batch_size=1,
                            shuffle=config.sample.shuffle_dataset,
                            pin_memory=True, drop_last=True, collate_fn=identity_collate)

    global_step = 0
    loader_iter = itertools.cycle(loader)
    latent_pool = torch.randn(
        (config.num_generations, 4, 64, 64), device=accelerator.device, dtype=inference_dtype,
    )
    for epoch in range(start_epoch, config.num_epochs):
        prompts_raw = next(loader_iter)
        if isinstance(prompts_raw[0], dict):
            prompt_texts = [p["prompt"] for p in prompts_raw]
            vqa_data = [p.get("vqa_list") for p in prompts_raw]
        else:
            prompt_texts = list(prompts_raw)
            vqa_data = [None] * len(prompts_raw)

        #################### SAMPLING ####################
        pipeline.unet.eval()
        samples = []

        expanded_prompts = []
        expanded_vqa = []
        for p_text, vqa in zip(prompt_texts, vqa_data):
            expanded_prompts.extend([p_text] * config.num_generations)
            expanded_vqa.extend([vqa] * config.num_generations)

        all_latents = []
        all_log_probs = []
        all_rewards = []
        all_prompts_embed = []
        all_pil_images = []

        global_input_latents = latent_pool

        sample_bs = config.sample.batch_size
        for i in range(0, len(expanded_prompts), sample_bs):
            current_batch = expanded_prompts[i:i + sample_bs]
            current_vqa = expanded_vqa[i]

            prompt_ids = pipeline.tokenizer(
                current_batch, return_tensors="pt", padding="max_length",
                truncation=True, max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]
            gen_idx = i % config.num_generations
            input_latents = global_input_latents[gen_idx:gen_idx + sample_bs].clone()

            with torch.no_grad():
                with autocast():
                    images, _, latents, log_probs = pipeline_with_logprob(
                        pipeline, prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        eta=config.sample.eta, output_type="pt",
                        latents=input_latents,
                    )

            pil_images = []
            for j, image in enumerate(images):
                pil = Image.fromarray(
                    (image.to(torch.float32).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((512, 512))
                pil_images.append(pil)

            vlm_rewards = vlm_reward_model.judge_batch(pil_images, current_batch[0], vqa_list=current_vqa)

            all_pil_images.extend(pil_images)

            latents = torch.stack(latents, dim=1).detach()
            log_probs = torch.stack(log_probs, dim=1).detach()

            all_latents.append(latents)
            all_log_probs.append(log_probs)
            all_rewards.append(vlm_rewards)
            all_prompts_embed.append(prompt_embeds)

            torch.cuda.empty_cache()

        all_latents = torch.cat(all_latents, dim=0)
        all_log_probs = torch.cat(all_log_probs, dim=0)
        all_rewards = torch.cat(all_rewards, dim=0).to(torch.float32)
        all_prompts_embed = torch.cat(all_prompts_embed, dim=0)
        timesteps = pipeline.scheduler.timesteps.repeat(
            config.sample.batch_size * config.num_generations, 1
        )

        time.sleep(0)

        samples = {
            "prompt_embeds": all_prompts_embed,
            "timesteps": timesteps[:, :-1],
            "latents": all_latents[:, :-1][:, :-1],
            "next_latents": all_latents[:, 1:][:, :-1],
            "log_probs": all_log_probs[:, :-1],
            "rewards": all_rewards,
        }

        all_rewards_world = gather_tensor(all_rewards)
        all_rewards_np = [float(r) for r in all_rewards_world.cpu()]
        mean_reward, p_correct, entropy_h, reward_var = reward_tracker.update(all_rewards_np)

        accelerator.log(
            {
                "reward": all_rewards_world,
                "epoch": epoch,
                "reward_mean": mean_reward,
                "reward_std": all_rewards_world.std(),
                "p_correct": p_correct,
                "entropy_H": entropy_h,
                "entropy_cumulative": reward_tracker.entropy_cumulative,
                "reward_variance": reward_var,
                "variance_cumulative": reward_tracker.variance_cumulative,
            },
            step=global_step,
        )

        preview_freq = 20
        if epoch % preview_freq == 0 and accelerator.is_main_process:
            preview_dir = os.path.join(config.ckpt_dir, "previews", config.run_name)
            os.makedirs(preview_dir, exist_ok=True)
            grid_size = int(config.num_generations ** 0.5)
            grid = Image.new("RGB", (grid_size * 512, grid_size * 512))
            for idx, pil in enumerate(all_pil_images):
                r = all_rewards[idx].item()
                row, col = idx // grid_size, idx % grid_size
                grid.paste(pil, (col * 512, row * 512))
            grid.save(os.path.join(preview_dir, f"epoch_{epoch:04d}.png"))
            for idx, pil in enumerate(all_pil_images):
                r = all_rewards[idx].item()
                pil.save(os.path.join(preview_dir, f"epoch_{epoch:04d}_{idx:02d}_r{r:.0f}.png"))
            accelerator.print(f"[Preview] Saved images to {preview_dir}/epoch_{epoch:04d}*")

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f"[Epoch {epoch:03d}] reward={all_rewards_world.mean().item():.4f} "
                  f"var={reward_var:.4f}  var_cum={reward_tracker.variance_cumulative:.4f}  "
                  f"p={p_correct:.4f}  H={entropy_h:.4f}  H_cum={reward_tracker.entropy_cumulative:.4f}")

        # ── GRPO Advantage (per-prompt group) ──
        n = len(samples["rewards"]) // config.num_generations
        advantages = torch.zeros_like(samples["rewards"])
        for i in range(n):
            start_idx = i * config.num_generations
            end_idx = (i + 1) * config.num_generations
            group_rewards = samples["rewards"][start_idx:end_idx]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
        samples["advantages"] = advantages
        samples["final_advantages"] = advantages

        total_batch_size, num_timesteps = samples["timesteps"].shape

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            perms = torch.stack([
                torch.randperm(num_timesteps, device=accelerator.device)
                for _ in range(total_batch_size)
            ])
            for key in ["timesteps", "latents", "next_latents", "log_probs"]:
                samples[key] = samples[key][
                    torch.arange(total_batch_size, device=accelerator.device)[:, None], perms,
                ]

            samples_batched = {
                k: v.reshape(-1, config.train.batch_size, *v.shape[1:])
                for k, v in samples.items()
            }
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            pipeline.unet.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0, disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    embeds = torch.cat([train_neg_prompt_embeds, sample["prompt_embeds"]])
                else:
                    embeds = sample["prompt_embeds"]

                for j in tqdm(
                    range(num_train_timesteps), desc="Timestep",
                    position=1, leave=False, disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(unet):
                        with autocast():
                            if config.train.cfg:
                                noise_pred = unet(
                                    torch.cat([sample["latents"][:, j]] * 2),
                                    torch.cat([sample["timesteps"][:, j]] * 2),
                                    embeds,
                                ).sample
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)
                            else:
                                noise_pred = unet(
                                    sample["latents"][:, j], sample["timesteps"][:, j], embeds,
                                ).sample

                            _, log_prob = ddim_step_with_logprob(
                                pipeline.scheduler, noise_pred,
                                sample["timesteps"][:, j], sample["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample["next_latents"][:, j],
                            )

                        advantages = torch.clamp(sample["final_advantages"], -config.train.adv_clip_max, config.train.adv_clip_max)
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(ratio, 1.0 - config.train.clip_range, 1.0 + config.train.clip_range)
                        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                        info["approx_kl"].append(0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2))
                        info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float()))
                        info["loss"].append(loss)

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        assert (j == num_train_timesteps - 1) and (i + 1) % config.train.gradient_accumulation_steps == 0
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({
                            "epoch": epoch, "inner_epoch": inner_epoch,
                            "reward_variance": reward_var,
                            "entropy_H": entropy_h, "p_correct": p_correct,
                        })
                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

        # ── Save checkpoint ──
        if epoch != 0 and epoch % config.save_freq == 0:
            if accelerator.is_main_process:
                base_ckpt_dir = config.ckpt_dir
                os.makedirs(base_ckpt_dir, exist_ok=True)
                ckpt_epoch_dir = os.path.join(base_ckpt_dir, f"{config.run_name}_ep{epoch}")
                os.makedirs(ckpt_epoch_dir, exist_ok=True)
                peft_model = accelerator.unwrap_model(pipeline.unet)
                peft_model.save_pretrained(ckpt_epoch_dir)
                accelerator.print(f"Saved checkpoint → {ckpt_epoch_dir}")
            if dist.is_initialized():
                dist.barrier()

    if accelerator.is_main_process:
        print("\n=== Final Summary ===")
        print(f"  variance_cumulative = {reward_tracker.variance_cumulative:.4f}")
        print(f"  entropy_cumulative = {reward_tracker.entropy_cumulative:.4f}")
        print(f"  reward_mean_per_step = {[f'{m:.4f}' for m in reward_tracker.reward_mean_history[::max(1, len(reward_tracker.reward_mean_history)//10)]]}")
        print(f"  p_per_step = {[f'{p:.4f}' for p in reward_tracker.p_history[::max(1, len(reward_tracker.p_history)//10)]]}")


if __name__ == "__main__":
    app.run(main)
