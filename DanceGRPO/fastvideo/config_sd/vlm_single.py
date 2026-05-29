import ml_collections
import os
base = __import__("fastvideo.config_sd.base", fromlist=["get_config"])


def vlm_single():
    config = base.get_config()

    config.run_name = "grpo_vlm_ac4_r16"
    config.num_epochs = 200
    config.save_freq = 50

    config.lora_rank = 16
    config.num_generations = 16
    config.prompt_file = "/workspace/oyly/geneval2_prompts_ac4.json"
    config.vlm_model_id = "/workspace/models/Qwen2.5-VL-7B-Instruct"
    config.ckpt_dir = "/workspace/oyly/checkpoints"

    config.sample.batch_size = 4
    config.sample.num_batches_per_epoch = 1

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 4
    config.train.learning_rate = 1e-4
    config.train.num_inner_epochs = 1
    config.train.clip_range = 1e-4
    config.train.adv_clip_max = 5

    return config


def get_config(name):
    return globals()[name]()
