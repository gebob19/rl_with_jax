#%%
import gym
import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt 
from torch.distributions.categorical import Categorical 
from torch.distributions.normal import Normal

import pybullet as p 
import pybullet_envs
from numpngw import write_apng
import numpy as np 
import matplotlib.pyplot as plt 

# env = gym.make('Pendulum-v0')
env = gym.make('CartPoleContinuousBulletEnv-v0')
# env = gym.make('AntBulletEnv-v0')
env.action_space, env.observation_space

#%%
n_action = env.action_space.shape[0]
import torch.nn.functional as F 

class Policy(nn.Module):
    def __init__(self, dobs, daction, dhidden=100):
        super().__init__()
        self.fc = nn.Linear(dobs, dhidden)
        self.mu_head = nn.Linear(dhidden, daction)
        self.sigma_head = nn.Linear(dhidden, daction)

    def forward(self, x):
        x = F.relu(self.fc(x))
        # tanh = (-1, 1) -- 2 * --> (-2, 2)
        mu = 10. * torch.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return (mu, sigma)

policy = Policy(4, 1)
optim = torch.optim.SGD(policy.parameters(), lr=1e-3)
optim.zero_grad()
T = 200

num_episodes = 5000
batch_size = 64
eval_every = 100
render_eval = False 

rollout_rewards = []
for epi_i in range(num_episodes):

    log_action_probs = []
    rewards = []
    obs = env.reset()

    for _ in range(T):
        tobs = torch.from_numpy(obs).float().flatten()
        mu, sigma = policy(tobs)
        assert not (mu.isnan() or sigma.isnan())

        a_space = Normal(mu, sigma)
        
        a = a_space.sample()
        prob = a_space.log_prob(a)

        if n_action == 1: 
            a = a[None]
        
        obs2, r, done, _ = env.step(a.numpy())

        rewards.append(torch.tensor(r).float())
        log_action_probs.append(prob)

        if done: break
        obs = obs2

    gamma = 0.99
    for i in range(len(rewards) - 1)[::-1]:
        rewards[i] = rewards[i] + gamma * rewards[i+1]
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) #/ rewards.std()
    if n_action > 1: 
        rewards = rewards.unsqueeze(-1)

    log_action_probs = torch.stack(log_action_probs)
    
    rollout_loss = -(log_action_probs * rewards).sum() # this is key
    rollout_loss.backward()

    if (epi_i+1) % batch_size == 0:
        optim.step()
        optim.zero_grad()

    if (epi_i+1) % eval_every == 0:
        eval_reward = 0 
        obs = env.reset()
        while True: 
            if render_eval: env.render()

            tobs = torch.from_numpy(obs).float().flatten()
            mu, sigma = policy(tobs)
            a_space = Normal(mu, sigma)
            a = a_space.sample()

            if n_action == 1: 
                a = a[None]
            
            obs2, r, done, _ = env.step(a.numpy())
            
            eval_reward += r 
            obs = obs2

            if done: break 
        if render_eval: env.close()

        print(f'Reward @ episode {epi_i}: {eval_reward:.2f}')
        rollout_rewards.append(eval_reward)

# %%
imgs = []
obs = env.reset()
eval_reward = 0 
while True: 
    img = env.render(mode='rgb_array')
    imgs.append(img)

    tobs = torch.from_numpy(obs).float()
    a_space = policy(tobs)
    mean, log_std = a_space[:n_action], a_space[n_action:]
    a_space = Normal(mean, torch.exp(log_std))
    a = a_space.sample()

    if n_action == 1: 
        a = a[None]
    
    obs2, r, done, _ = env.step(a.numpy())
    
    eval_reward += r 
    obs = obs2
    if done: break 

print(f'Total Reward: {eval_reward:.2f} #frames: {len(imgs)}')
print('writing...')
write_apng('demo.png', imgs, delay=20)

# %%
plt.plot(rollout_rewards)
plt.show()

# %%
