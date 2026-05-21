"""Rollout with checkpoint: load LoRA, generate images, compute rewards."""
import torch, numpy as np, os, sys
from diffusers import DDIMScheduler, UNet2DConditionModel, AutoencoderKL
from peft import PeftModel
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image

sys.path.insert(0, "/workspace/oyly")
from single_question_diffusion_rlvr import compute_reward

MODEL = "/workspace/models/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
CKPT = "/workspace/oyly/checkpoints/sq_v9_fixed_ep50"
DEVICE = "cuda"
N = 16
STEPS = 50
CFG = 7.5

print("Loading base SD1.5...")
tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder").to(DEVICE)
vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae").to(DEVICE)
unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet", use_safetensors=False).to(DEVICE)
scheduler = DDIMScheduler.from_pretrained(MODEL, subfolder="scheduler")
scheduler.set_timesteps(STEPS)
text_encoder.requires_grad_(False)
vae.requires_grad_(False)

print("Loading LoRA checkpoint...")
unet = PeftModel.from_pretrained(unet, CKPT)
unet.eval()
print(f"  LoRA loaded from {CKPT}")

print("Encoding prompts...")
text_inputs = tokenizer(["a red square"] * N, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
text_emb = text_encoder(text_inputs.input_ids.to(DEVICE))[0]
uncond_emb = text_encoder(tokenizer([""] * N, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(DEVICE))[0]

print("Generating...")
latents = torch.randn((N, unet.config.in_channels, 64, 64), device=DEVICE, dtype=torch.float32)

with torch.no_grad():
    for t in scheduler.timesteps:
        nc = unet(latents, t, encoder_hidden_states=text_emb).sample
        nu = unet(latents, t, encoder_hidden_states=uncond_emb).sample
        noise_pred = nu + CFG * (nc - nu)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

imgs = vae.decode(latents / 0.18215).sample
imgs = (imgs / 2 + 0.5).clamp(0, 1)
imgs_np = (imgs.cpu().float().permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)

pil_imgs = [Image.fromarray(x) for x in imgs_np]
rewards = compute_reward(pil_imgs)

OUT = "/workspace/oyly/debug_imgs/ckpt_ep50"
os.makedirs(OUT, exist_ok=True)

# Save individual images
for i, (img, r) in enumerate(zip(pil_imgs, rewards)):
    img.save(f"{OUT}/img_{i:02d}_r{r:.2f}.png")
    print(f"  img_{i:02d}.png  reward={r:.4f}")

# Save grid (4x4)
grid_img = Image.new("RGB", (4 * 512, 4 * 512))
for i, img in enumerate(pil_imgs):
    row, col = i // 4, i % 4
    img_resized = img.resize((512, 512))
    grid_img.paste(img_resized, (col * 512, row * 512))
grid_img.save(f"{OUT}/grid.png")

print(f"\n=== Summary ===")
print(f"Mean reward: {np.mean(rewards):.4f}")
print(f"Correct (>0.5): {sum(1 for r in rewards if r > 0.5)}/{N}")
print(f"Best reward:  {max(rewards):.4f}")
print(f"Worst reward: {min(rewards):.4f}")
print(f"Saved to {OUT}/")
