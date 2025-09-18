import gymnasium as gym
# import shimmy
import ale_py

env = gym.make("ALE/Alien-v5", render_mode="human")
observation, info = env.reset()

for _ in range(500):
    action = env.action_space.sample()  # 随机动作
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
