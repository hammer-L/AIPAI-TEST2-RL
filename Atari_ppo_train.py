#!/usr/bin/env python
# coding: utf-8

# In[16]:


import gymnasium as gym
import ale_py
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3 import PPO
import os

# 生成向量化环境
env = make_vec_env(
    "ALE/Alien-v5",
    n_envs=4,
    wrapper_class=AtariWrapper,
    vec_env_cls=None  # 默认 DummyVecEnv
)

env.envs[0].spec


# In[27]:


tensorboard_path = r'./runs/ppo_alien'
model_path = r"./runs/ppo_alien/ppo_alien2.zip"

model = PPO(
    'CnnPolicy',
    env,
    verbose=0,
    tensorboard_log = os.path.join(tensorboard_path, 'tensorboard'),
    n_steps=128,
    batch_size = 256,
    n_epochs = 4,
    learning_rate = 2e-4,
    gamma = 0.99,
    clip_range = 0.1,
    ent_coef = 0.01,
)

# model.learn(total_timesteps = 20_000_000)

# model.save(model_path)
# del model

# 加载模型
# model = PPO.load(model_path, env=env)


# In[3]:


# !jupyter nbconvert --to script atari_ppo.ipynb


# In[26]:


base_env = gym.make("ALE/Alien-v5", render_mode="human")
eval_env = AtariWrapper(base_env)
model = PPO.load("./runs/ppo_alien/ppo_alien2.zip")
num_epochs = 4

while (num_epochs):
    num_epochs -= 1
    obs, _ = eval_env.reset()
    done, truncated = False, False

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        eval_env.render()


# In[ ]:




