import ml_collections
base = __import__("fastvideo.config_sd.base", fromlist=["get_config"])

def resume2():
    config = base.get_config()
    config.run_name = "grpo_lion_elephants_resume"
    config.num_epochs = 800
    config.save_freq = 100
    config.lora_rank = 64
    config.num_generations = 16
    config.prompt_file = "/workspace/oyly/prompt_lion_elephants.jsonl"
    config.vlm_model_id = "/workspace/models/Qwen2.5-VL-7B-Instruct"
    config.ckpt_dir = "/workspace/oyly/checkpoints"
    config.sample.batch_size = 1
    config.sample.num_steps = 25
    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = 4
    config.train.learning_rate = 1e-4
    config.train.clip_range = 1e-4
    config.train.adv_clip_max = 5
    # Resume from Exp2 ep400 checkpoint
    config.lora_checkpoint = "/workspace/oyly/checkpoints/grpo_lion_elephants_2026.05.27_14.03.10_ep400"
    config.start_epoch = 0  # auto-detect from checkpoint dir name → 401
    return config

def get_config(name):
    return globals()[name]()
