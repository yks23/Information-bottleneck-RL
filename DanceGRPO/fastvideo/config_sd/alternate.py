import ml_collections
base = __import__("fastvideo.config_sd.base", fromlist=["get_config"])

def alternate():
    config = base.get_config()
    config.run_name = "grpo_alternate"
    config.num_epochs = 1000
    config.save_freq = 100
    config.lora_rank = 64
    config.num_generations = 16
    config.prompt_file = "/workspace/oyly/prompt_both.jsonl"
    config.vlm_model_id = "/workspace/models/Qwen2.5-VL-7B-Instruct"
    config.ckpt_dir = "/workspace/oyly/checkpoints"
    config.sample.batch_size = 1
    config.sample.shuffle_dataset = False  # deterministic alternation
    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = 4
    config.train.learning_rate = 1e-4
    config.train.clip_range = 1e-4
    config.train.adv_clip_max = 5
    return config

def get_config(name):
    return globals()[name]()
