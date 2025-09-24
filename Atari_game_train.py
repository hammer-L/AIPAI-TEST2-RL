#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gymnasium as gym
import ale_py

# In[2]:


import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation
from tqdm import tqdm
from gymnasium import RewardWrapper

"""
step1: 预处理环境。
why: atari的obs是210*160*3，应该先处理
how: 1.转灰度 2.压缩 3.堆叠最近的4帧画面(这样能同时展示s和a)
"""
#train
# env = gym.make("ALE/Alien-v5",frameskip=1)

# render
env = gym.make("ALE/Breakout-v5",frameskip=4)

#print
obs, info = env.reset()
# print ('obs', obs.shape)
# print ('info', info)

# AtaAtariPreprocessing函数详解：
# - frame_skip=4 一个动作保持4帧 -> 1.可以减少计算量 2.让动作效果更加明显， 这里atari已经内置了
# - grayscale_obs=True 转化为灰度图像
# - scale_obs=True 把像素值从【0,255】-> 【0,1】
env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True,frame_skip=1)

# 表示每次obs由最近的4帧图形拼接而成
# shape（4,210,160）
env = FrameStackObservation(env, stack_size=4)

# print 
obs, info = env.reset()
# print ('obs', obs.shape)
# print ('info', info)
# print (env.action_space.n)
# print(env.spec)              # 查看环境配置

class RewardShaping(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.penalty = -5
        self.lives = None

    def reset(self, **Kwargs):
        obs, info = self.env.reset(**Kwargs)
        self.lives = info.get("lives", 0)  # 默认 0 避免 None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_lives = info.get("lives", self.lives)
        if current_lives < self.lives:  # 掉血
            reward += self.penalty
        self.lives = current_lives
        return obs, reward, terminated, truncated, info

env = RewardShaping(env)
obs, info = env.reset()


# ### EnvSpec 逐项解读（重点说明含义与影响）
# 
# - id='ALE/Alien-v5'
# 环境标识：Atari 的 Alien（版本 v5）。
# 
# - entry_point='ale_py.env:AtariEnv'
# 真正创建底层环境的类是 ALE（Arcade Learning Environment）的 AtariEnv。
# 
# - kwargs={...}（底层 env 的参数）
# 
# - game: 'alien'：游戏名。
# 
# - repeat_action_probability: 0.25：sticky actions（黏性动作）概率为 0.25，表示有 25% 概率重复上一个动作，增加环境随机性。
# 
# - full_action_space: False：使用 minimal action set 而不是所有动作。
# 
# - frameskip: 1：底层 ALE 本身的 frameskip=1（非常重要——不是最终跳帧的意思；wrapper 可能会再做跳帧）。
# 
# - max_num_frames_per_episode: 108000：每个 episode 最多 108000 帧（Atari 的标准——约 30 分钟）。
# 
# - render_mode: 'human'：渲染模式。
# 
# - max_episode_steps=None
# 表示没有被 TimeLimit（Gym 的 step 上限封装器）包裹，所以 env.spec 的 max_episode_steps 是 None。不过底层的 max_num_frames_per_episode 仍然存在（上面那项）。
# 
# - additional_wrappers=(WrapperSpec(...), WrapperSpec(...))
# 这非常关键 —— Gymnasium 在 make("ALE/Alien-v5") 时自动为你套了两个 wrapper（你通常不需要再手动套一次）：
# 
# **AtariPreprocessing 的 kwargs:**
# 
# - noop_max=30：reset 时会随机做 0–30 个 NOOP，用于打乱起始状态。
# 
# - frame_skip=4：AtariPreprocessing 会把每个动作重复执行 4 帧 —— 这就是常说的跳帧（把 60 FPS 降为 15 FPS 的效果）。
# 
# - screen_size=84：会把屏幕缩到 84×84。
# 
# - terminal_on_life_loss=False：失去一条命不会把 episode 标记为 terminated（很多实现可选这个行为）。
# 
# - grayscale_obs=True：转灰度。
# 
# - grayscale_newaxis=False：灰度不会被加成单独的最后轴（意味着单帧是 2D (H,W) 而不是 (H,W,1)）。
# 
# - scale_obs=True：把像素归一化到 [0,1]（仍为 uint8、0–255）。
# 
# **FrameStackObservation 的 kwargs:**
# 
# - stack_size=4：堆叠最近 4 帧 → 最终 observation 包含 4 帧历史。
# 
# - padding_type='reset'：在 episode 开始时，空的前帧用 reset 的观测填充（而不是用 0）。
# 
# - 注意：顺序通常是先 AtariPreprocessing（做灰度/resize/跳帧），再 FrameStackObservation（在预处理后的帧上做堆叠）。
# 
# - vector_entry_point='ale_py.vector_env:AtariVectorEnv'
# - 环境支持 vectorized（并行）版本，用于同时跑多个 env。

# In[3]:


"""
class DQN
input : state s and action a
        in other words, 4-frame obs (N, 4, 84, 84)
return : q(s,a)
"""
class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), ## output=(N, 32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size=4, stride=2), ## output=(N, 64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), ## output = (N, 64, 7, 7)
            nn.ReLU(),
            nn.Flatten(), ## 64*7*7 = 3136
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim) # 对应每个a的q值
        )

    def forward(self, x):
        y = self.net(x)
        return y

"""
test DQN
input: a dummy tensor to test DQN
"""
def test_DQN():
    print ("===test DQN===")
    dqn = DQN(env.action_space.n)
    dummy_x = torch.randn((1,4,84,84))
    print (dummy_x.shape)
    y = dqn(dummy_x)
    print (y.shape)
    print (y)
    print ("===test DQN===\n")

# test_DQN()


"""
class ReplayBuffer
why: 1.打破数据的事件关联性 2.增强样本的利用率 3.提高训练的稳定性
how: 把经验放在deque容器buffer中， 需要的时候随机采样batch_size个
"""
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity) # 双端队列deque, 如果满了， 最老的经验会被删除

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) #从buffer中采样 batch_size 个样本

        # *是解包操作unpacking, 这是把batch解包后， 按列组合
        state, action, reward, next_state, done = map(np.array, zip(*batch)) 
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


"""
关键参数赋值
"""
epsilon_start, epsilon_end, epsilon_decay = 1, 0.05, 2000000
gamma = 0.99
batch_size = 32 # 一次梯度下降用这么多数量的数据
update_target = 8000 # 更新target的频率

num_episodes = 10000
returns = []



"""
def train()
封装整个训练过程
"""
from torch.utils.tensorboard import SummaryWriter # 用 tensorboard 记录训练过程
def train(env):
    device = torch.device('cuda')
    action_dim = env.action_space.n
    global_step = 0
    writer = SummaryWriter(log_dir='runs/breakout/dqn1')

    #qnet and target_net
    qnet = DQN(action_dim).to(device)
    target_net = DQN(action_dim).to(device)

    # qnet.load_state_dict(torch.load("runs/dqn_ckpt_4000.pth", map_location=device))
    
    # 将qnet 的参数传入 target_net
    # why两个net ： 1.虽然两个都是估计q(s,a) 
    #               2.先固定target_net，对损失函数求偏导的时候就不会太复杂，然后再把更新后的参数赋给targetnet
    #               3. 这样还可以稳定训练，减少 Q 值振荡
    target_net.load_state_dict(qnet.state_dict())

    # 创建实例
    optimizer = optim.Adam(qnet.parameters(), lr=1e-4)
    buffer = ReplayBuffer()

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.array(state)
        episode_loss = 0

        # max = num_episodes * steps
        # steps 最大为 108000 / 4, 根据 env.spec()查看得 'max_num_frames_per_episode': 108000
        # 上面有设置 skip_frame = 4, 
        episode_reward = 0

        done = False
        while not done:

            # 用 epsilon-greeedy 策略采样episode， 刚开始多exploration, 后来多exploitation
            # global_step = 0 -> epsilon = 1.0
            # global_step = 正无穷 -> epsilon = 0.1
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp( -global_step / epsilon_decay)
            # epsilon = 0.1

            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 4, 84, 84)
                    q_values = qnet(s)
                    action = q_values.argmax(1).item() # q_value (1,18) -> argmax(1)指定在action space 维度

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state)
            done = terminated or truncated

            # 放入经验 buffer 中
            buffer.push(state, action, reward, next_state, done)

            #更新状态
            state = next_state
            episode_reward += reward
            global_step += 1


            # 用经验回放来训练
            if len(buffer) > 10*batch_size:
                s, a, r, ns, d = buffer.sample(batch_size)

                #放在gpu上
                s = torch.tensor(s, dtype=torch.float32, device=device)  
                ns = torch.tensor(ns, dtype=torch.float32, device=device)
                a = torch.tensor(a, dtype=torch.long, device=device)     
                r = torch.tensor(r, dtype=torch.float32, device=device)  
                d = torch.tensor(d, dtype=torch.float32, device=device)

                # gather(dim, idx): dim=1表示在行上， 选取idx列的数据
                # a = [0, 2, 1],  a.unsqueeze(1) = [[0], [2], [1]]
                # 本质是从每个 batch 中选出采样到 qvalue
                q_values = qnet(s).gather(1, a.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    # 在计算 target_q 的时候要锁住梯度计算， 不然会很复杂
                    max_next_q = target_net(ns).max(1)[0]
                    target_q = r + gamma * (1 - d) * max_next_q

                loss = nn.MSELoss()(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                episode_loss += loss.item() # 用于记录整个episode的loss

            if global_step % update_target == 0:
                target_net.load_state_dict(qnet.state_dict())

        # 记录训练过程
        writer.add_scalar("Reward", episode_reward, episode)
        writer.add_scalar("Loss", episode_loss, episode)
        writer.add_scalar("Epsilon", epsilon, episode)
        
        returns.append(episode_reward)
        # print (f"Episode {episode}, Return {episode_reward}, Epsilon {epsilon:.3f}")

        # save 权重， 不然一个意外就白跑了半天
        if (episode + 1) % 1000 == 0:
            torch.save(qnet.state_dict(), f'runs/breakout/dqn1{episode+1}.pth')
                


# In[4]:


train(env)


# In[7]:


"""
用render展示训练成果
"""

# device = torch.device('cuda')
# action_dim = env.action_space.n
# global_step = 0
# # writer = SummaryWriter(log_dir='runs/dqn2')

# #qnet and target_net
# qnet = DQN(action_dim).to(device)
# target_net = DQN(action_dim).to(device)
# qnet.load_state_dict(torch.load("runs/dqn_ckpt_4000.pth", map_location=device))
# target_net.load_state_dict(qnet.state_dict())

# num_episode = 100

# for episode in range(num_episodes):
#     state, _ = env.reset()
#     state = np.array(state)
#     done = False

#     while not done:
#         s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 4, 84, 84)
#         q_values = qnet(s)
#         action = q_values.argmax(1).item()

#         next_state, reward, terminated, truncated, _ = env.step(action)
#         next_state = np.array(next_state)
#         done = terminated or truncated

#         state = next_state


# In[ ]:




