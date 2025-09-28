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


# In[ ]:


model = PPO(
    'CnnPolicy',
    env,
    verbose=1,
    tensorboard_log = os.path.join('./runs/ppo/ppo_alien', 'tensorboard'),
    n_steps=128,
    batch_size = 256,
    n_epochs = 4,
    learning_rate = 2e-4,
    gamma = 0.99,
    clip_range = 0.1,
    ent_coef = 0.01, 
)

model.learn(total_timesteps = 10_000)

model.save(r"./runs/ppo/ppo_alien-v5")
del model

# 加载模型
model = PPO.load("./runs/ppo/ppo_alien-v5", env=env)


# In[ ]:




