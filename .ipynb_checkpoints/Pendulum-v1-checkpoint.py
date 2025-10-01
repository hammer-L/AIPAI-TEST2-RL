#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gymnasium as gym
env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)

"""test"""
# obs, info = env.reset()
# for _ in range(200):
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     env.render()
#     if terminated or truncated:
#         obs, info = env.reset()
print ("env.spec", env.spec)
print ()
print ('env.obs', env.observation_space.shape) # x, y, w
print ('env.act', env.action_space.shape) # T

state_dim = env.observation_space.shape[0]
print ('state_dim', state_dim)
action_bound = env.action_space.high[0]
print ('actspace.high', env.action_space.high)
print ('action_bound', action_bound)


# ![pendulum](./figures/pendulum.png "pendulum")
# 

# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


import random
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils

"""
用 DDPG 算法求解倒立摆问题。
"""
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(), 
        )
        self.action_bound = action_bound 

    def forward(self, x):
        """
        为什么可以直接 *action_bound： 因为最后一步用的tanh当激活函数，
        把整个空间压缩到 （-1， 1），这样可以保证不会越界。
        """
        return self.model(x) * self.action_bound

class QNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), #Q(s,a) 即包含s也包含a， 因此要把两个拼接起来
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, a):
        cat = torch.cat([x,a], dim=1)
        return self.model(cat)

class DDPG:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 sigma, actor_lr, critic_lr, tau, gamma, device):

        # 初始化四个网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = QNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_critic = QNet(state_dim, hidden_dim, action_dim).to(device)

        # 将两个 target net 共享权重
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 优化器选择 adam
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 剩余参数赋值
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络 soft update 参数
        self.action_dim = action_dim
        self.device = device

    def take_action(self, state):
        # 将参数放在 gpu 上
        # print ('state', state)  state (array([-0.9728001 , -0.23164637,  0.94702655], dtype=float32), {})
        if isinstance(state, tuple):
            obs = state[0]
        else:
            obs = state
        obs = np.array(obs, dtype=np.float32)

        # 转成 tensor 并加 batch 维度
        state = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        action = self.actor(state).item()
        
        # 给动作加上高斯噪声，增加随机性
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update(self, net, target_net):
        """
        net.parameters()： 返回一个生成器iterator， 只包含可训练参数。
        net.state_dict(): 返回一个字典， 包含模型所有的参数
        why 用parameters().data: 对data的修改不会影响autograd。
        """
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # q_target 和 q_critic 计算
        next_qvalue = self.target_critic(next_states, self.target_actor(next_states))
        q_target = rewards + self.gamma * next_qvalue * (1 - dones)
        q_critic = self.critic(states, actions)
        
        # critic训练过程： critic_loss = TD error^2
        critic_loss = F.mse_loss(q_critic, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # actor训练过程： actor_loss = -critic(states, actions)
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # 软更新到 target网络
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)


# In[4]:


# 超参赋值
actor_lr = 3e-4
critic_lr = 3e-3
num_episodes = 2000
hidden_dim = 64
gamma = 0.98
tau = 0.005  # 软更新参数
buffer_size = 20000
minimal_size = 1000
batch_size = 64
sigma = 0.01  # 高斯噪声标准差
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 使用 rl_utils中的 ReplayBuffer class 和 train_off_policy_agent
# 引用自 https://github.com/boyu-ai/Hands-on-RL
replay_buffer = rl_utils.ReplayBuffer(buffer_size)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]
# 创建agent
agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)

return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)


# In[ ]:


保存训练模型的参数
torch.save({
    'actor': agent.actor.state_dict(),
    'critic': agent.critic.state_dict(),
    'optimizer_actor': agent.actor_opt.state_dict(),
    'optimizer_critic': agent.critic_opt.state_dict(),
    'return_list': return_list
}, f"./runs/pendulum/ckpt2.pth")


# In[ ]:


# 展示训练成果

env = gym.make("Pendulum-v1", render_mode="human", g=9.81)

for ep in range(10):
    ep -= 1
    agent_test = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)
    
    # load
    checkpoint = torch.load(f"./runs/pendulum/ckpt2.pth", weights_only=False)
    agent_test.actor.load_state_dict(checkpoint['actor'])
    agent_test.critic.load_state_dict(checkpoint['critic'])
    agent_test.actor_opt.load_state_dict(checkpoint['optimizer_actor'])
    agent_test.critic_opt.load_state_dict(checkpoint['optimizer_critic'])
    return_list = checkpoint['return_list']

    state, info = env.reset()
    done = False
    while not done:
        action = agent.take_action(state)
        # next_state, reward, done, _ = env.step(action)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state

"""
结果非常好， 不到一秒就倒立了。
"""


# In[ ]:


import imageio
import gymnasium as gym
import torch
# 1. 创建环境（注意 render_mode 要改成 rgb_array）
env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)

# 2. 加载 agent
agent_test = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)
checkpoint = torch.load(f"./runs/pendulum/ckpt2.pth", map_location=device, weights_only=False)
agent_test.actor.load_state_dict(checkpoint['actor'])
agent_test.critic.load_state_dict(checkpoint['critic'])
agent_test.actor_opt.load_state_dict(checkpoint['optimizer_actor'])
agent_test.critic_opt.load_state_dict(checkpoint['optimizer_critic'])

# 3. 运行一个 episode 并收集帧
frames = []
state, info = env.reset()
done = False
while not done:
    action = agent_test.take_action(state)   # 注意这里应该是 agent_test 而不是 agent
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    state = next_state
    
    # 保存当前帧
    frame = env.render()
    frames.append(frame)

# 4. 保存为 GIF
imageio.mimsave("pendulum.gif", frames, fps=30)
env.close()


# In[ ]:




