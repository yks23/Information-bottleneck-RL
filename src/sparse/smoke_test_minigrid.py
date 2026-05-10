import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO


env = gym.make("MiniGrid-DoorKey-8x8-v0")
env = ImgObsWrapper(env)

model = PPO("CnnPolicy", env, verbose=1, n_steps=128, batch_size=64)
model.learn(total_timesteps=1000)

obs, info = env.reset()
done = False
total_reward = 0.0

while not done:
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

print("Episode return:", total_reward)
