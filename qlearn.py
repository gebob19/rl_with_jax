#%%
import gym
import numpy as np 
import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt 
from torch.distributions.categorical import Categorical
import copy 
from numpngw import write_apng

#%%
env = gym.make('CartPole-v1')
# env = gym.make('LunarLander-v2')

T = 200
num_episodes = 5000

update_target_every = 2
batch_size = 64
gamma = 0.99 
tau = 0.999
max_replay_buffer_size = 50000

eval_every = 200 

replay_buffer = []
rollout_rewards = []

# loss_func = nn.MSELoss()
loss_func = nn.SmoothL1Loss()

dim_in = env.observation_space.shape[0]
dim_out = env.action_space.n
qnet = nn.Sequential(
    nn.Linear(dim_in, 32), 
    nn.ReLU(), 
    nn.Linear(32, 32), 
    nn.ReLU(), 
    nn.Linear(32, dim_out),
)
optim = torch.optim.Adam(qnet.parameters(), lr=1e-3)

qnet_target = copy.deepcopy(qnet)

for epi_i in range(num_episodes):
    # rollout 
    obs = env.reset()
    for _ in range(T):
        # a = env.action_space.sample() # random policy rollout 
        
        tobs = torch.from_numpy(obs).float()
        with torch.no_grad():
            a_logits = qnet(tobs)
        a = Categorical(logits=a_logits).sample().numpy()

        obs2, r, done, _ = env.step(a)

        t = (obs, a, obs2, r, done) # add to buffer 
        replay_buffer.append(t)

        if done: break 
        obs = obs2

    # optim 
    idxs = np.random.randint(len(replay_buffer), size=(batch_size,))
    trans = [replay_buffer[i] for i in idxs]

    def extract(trans, i): return torch.from_numpy(np.array([t[i] for t in trans]))
    obst = extract(trans, 0).float()
    at = extract(trans, 1)
    obs2t = extract(trans, 2).float()
    rt = extract(trans, 3).float()
    donet = extract(trans, 4).float()

    for i in range(len(idxs)):
        dql = qnet(obst[i]).max(-1)[-1] # idxs of max 
        qtarget = qnet_target(obs2t[i])[dql]
        y = rt[i] + (1 - donet[i]) * gamma * qtarget
        y = y.detach() # stop grad 

        ypred = qnet(obst[i])[at[i]]

        loss = loss_func(ypred, y)
        
        optim.zero_grad()
        loss.backward()
        for param in qnet.parameters(): # gradient clipping 
            param.grad.data.clamp_(-1, 1)
        optim.step()

        # # polyak avging -- worse perf. than hard update
        # for sp, tp in zip(qnet.parameters(), qnet_target.parameters()):
        #     tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

    # hard updates 
    if (epi_i+1) % update_target_every == 0:
        qnet_target = copy.deepcopy(qnet)

    if (epi_i+1) % eval_every == 0:
        eval_reward = 0 
        obs = env.reset()
        while True: 
            tobs = torch.from_numpy(obs).float()
            a_space = qnet(tobs)
            a = a_space.argmax(-1)
            
            obs2, r, done, _ = env.step(a.numpy())
            
            eval_reward += r 
            obs = obs2

            if done: break 

        print(f'Total Reward @ step {epi_i}: {eval_reward}')
        rollout_rewards.append(eval_reward)


    if len(replay_buffer) > max_replay_buffer_size: 
        print('cleaning up replay buffer...')
        cut_idx = int(max_replay_buffer_size * 0.5) # remove half of exps
        replay_buffer = replay_buffer[cut_idx:]

# %%
imgs = []
eval_reward = 0 
obs = env.reset()
while True: 
    img = env.render(mode='rgb_array')
    imgs.append(img)

    tobs = torch.from_numpy(obs).float()
    a_space = qnet(tobs)
    a = a_space.argmax(-1)
    
    obs2, r, done, _ = env.step(a.numpy())
    
    eval_reward += r 
    obs = obs2

    if done: break 

print(f'Total Reward: {eval_reward:.2f} #frames: {len(imgs)}')
print('writing...')
write_apng('dqn_cartpole.png', imgs, delay=20)

# %%
plt.plot(rollout_rewards)
plt.show()

# %%
# %%
# %%
