

# AIPAI-TEST2-RL

This is a repo for AIPAI's test



### 老虎机:

- 老虎机的程序在 ```BernoulliBandit.ipynb ```中。
- 使用了 epsilon-greedy, decaying-epsilon-greedy, UCB 三种算法求解。



### Gym-taxi

- 程序在 ```Gym-Taxi.ipynb```中。
- 问题回答穿插在 jupyter 文档中。



### Atari-Alien-v5

- 用了两种方法求解。DQN 版本在```Gym-Taxi.ipynb```中，```Atari_game_train.py```是用 jupyter 直接转换为py文件的，只是用来训练。PPO版本在```Atari_game_train.py```中，用了stable baseline库。

- 模型的```state_dict```保存在```./runs/alien_dqn```中。

- 反思与记录：

  ​	1.第一个5000轮，loss没有收敛，render看了一下训练成果，发现agent会卡在右上角不动，推测可能是训练轮数太少了，
  到后面epsilon又变成0.1，导致探索性不足，q值趋于不好的稳定。
  ​	2.第二次，使用了上面第4000轮的权重，调整了一下epsilon从0.75-0.1，调整了一下lr=1.5e-4，但是loss发散了，发散的很严重，
    赶紧停掉了，重新设计一下参数。
  ​    3.第三次：lr=0.9e-4， epsilon_start, epsilon_end, epsilon_decay = 0.7, 0.1, 6000000， num_episodes = 5000结果非常烂...agent完全不会走路了已经，开局吃完上面的星星就开始乱走路，并且有非常严重的走路的偏好（喜欢往右上角走）

  ​	4.第四次：添加了一个reward wrapper做reward shaping, 掉血给惩罚， 教智能体先学会躲避怪物才能多活命， 才能右更多的时间和机会探索环境。此外还将epsilon改成指数下降，这样在前期纯随机的时间会减少，使训练更自然。结果也很一般，loss几乎收敛了，但是reward还是不是很高而且抖动严重。智能体有躲避怪兽的意识，但还是不够智能。

   5.换用PPO，效果好多了，一次生命的平均reward可以到1200（虽然中间突然policy崩了，但后面拉回来了）。

  

- 反思DQN有哪些问题，为什么表现不好:
	
  - 使用DQN有一些特征:1.loss几乎收敛  2.但是reward依旧很低  3.agent经常会卡在右上角
  - 推测原因：1.loss收敛，但是reward低，应该是收敛到了不好的q值，而后期探索又变低，agent很难探索到一个比较好的策略。  2.agent卡在右上角：可能是agent为了苟活采取了保守策略，然后在experience buffer 中不断积累重复经验，并且Q值存在over-estimation bias，通往右上角的Q值被高估， agent会稳定往这边走。3.dqn的policy的bias比较大（用的 one-step TD 来估计Q值， 并且Qnet和Q*本身也有bias），因此会产生有偏（bias大）且坚定（var小）的动作。
  
- 反思PPO为什么好：
  
  - **1.稳定的更新策略：**PPO 通过 clip ratio 约束更新幅度，相当于对 KL 散度做了个软限制，在 Atari 这种高维离散空间里，这种稳定性非常关键。
  - 2.**样本利用率更高:**PPO 在保持 On-policy 性质的前提下，允许在同一批样本上多次梯度更新(importance sampling)。在 Atari 游戏这种 reward 稀疏、环境复杂的场景里，能显著提高学习效率。
  - 3.**优势函数估计（GAE）降低方差:**Atari 游戏奖励很稀疏，直接用 MC 或 TD 估计都会很抖动,PPO 引入 GAE (Generalized Advantage Estimation)，在偏差与方差之间做平衡，使得梯度估计更平滑，更容易学到有效策略。



### 论文阅读

- 问题回答放在```论文阅读.pdf```中。
- 前面是问题回答部分，后面顺手附上了学习笔记。



### 自定义

- 选择第一项，探索Pendulum-v1环境， 程序在 ```Pendulum-v1```中。
- 使用了DDPG方法。
- 借用了```rl_utils.py```工具箱。来源[hands-on rl](https://github.com/boyu-ai/Hands-on-RL)。