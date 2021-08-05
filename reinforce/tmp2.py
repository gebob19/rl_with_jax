#%%
import jax 
import jax.numpy as jnp 
import haiku as hk 
import distrax

import torch 
from torch.distributions.categorical import Categorical 

import gym 
import numpy as np 

from functools import partial
import time 

#%%
env = gym.make('CartPole-v0')

import torch.nn as nn 
policy = nn.Sequential(
    nn.Linear(env.observation_space.shape[0], 32),
    nn.ReLU(), 
    nn.Linear(32, 32), 
    nn.ReLU(), 
    nn.Linear(32, env.action_space.n), 
    nn.Softmax(dim=-1),
)
optim = torch.optim.SGD(policy.parameters(), lr=1e-3)
optim.zero_grad()

def _policy_fcn(obs):
    a_probs = hk.Sequential([
        hk.Linear(32), jax.nn.relu,
        hk.Linear(32), jax.nn.relu,
        hk.Linear(env.action_space.n), jax.nn.softmax
    ])(obs)
    return a_probs 

policy_fcn = hk.transform(_policy_fcn)
policy_fcn = hk.without_apply_rng(policy_fcn)
p_frwd = jax.jit(policy_fcn.apply)

seed = 0
rng = jax.random.PRNGKey(seed)
obs = env.reset() # dummy input 
p_params = policy_fcn.init(rng, obs) 

#%%
def torch_policy(obs):
    a_space = policy(torch.from_numpy(obs).float())
    a_space = Categorical(a_space)
    a = a_space.sample()
    log_prob = a_space.log_prob(a)
    return a, log_prob

def torch_rollout():
    np.random.seed(seed)
    env.seed(seed)
    obs = env.reset()

    log_probs = []
    rewards = []
    while True: 
        ## -- 
        a, log_prob = torch_policy(obs) 
        a = a.numpy()
        ## -- 

        a = np.random.choice(env.action_space.n)
        obs2, r, done, _ = env.step(a)
        if done: break 
        obs = obs2 

        log_probs.append(log_prob)
        rewards.append(r)
    
    ## --
    log_probs = torch.stack(log_probs)
    r = torch.tensor(rewards)
    loss = -(log_probs * r).sum()
    ## --
    return loss 

@jax.jit    
def jax_policy(p_params, obs, key):
    a_probs = p_frwd(p_params, obs)
    a_probs = distrax.Categorical(probs=a_probs)
    a, log_prob = a_probs.sample_and_log_prob(seed=key)        
    return a, log_prob

def jax_rollout(p_params, rng):
    np.random.seed(seed)
    env.seed(seed)
    obs = env.reset()

    log_probs = []
    rewards = []
    while True: 
        ## -- 
        rng, key = jax.random.split(rng, 2) 
        a, log_prob = jax_policy(p_params, obs, key)
        a = a.astype(int)
        ## --         

        a = np.random.choice(env.action_space.n)
        obs2, r, done, _ = env.step(a)
        if done: break 
        obs = obs2 

        log_probs.append(log_prob)
        rewards.append(r)
    
    ## --
    log_prob = jnp.stack(log_probs)
    r = np.stack(rewards)
    loss = -(log_prob * r).sum()
    ## --
    return loss 

## only sample policy (log_prob is computed in loss)
@jax.jit    
def jax_policy2(p_params, obs, key):
    a_probs = p_frwd(p_params, obs)
    a_probs = distrax.Categorical(probs=a_probs)
    a = a_probs.sample(seed=key)
    return a

def jax_rollout2(p_params, rng):
    np.random.seed(seed)
    env.seed(seed)
    obs = env.reset()

    observ, action, rew = [], [], []
    while True: 
        ## --
        rng, key = jax.random.split(rng, 2) 
        a = jax_policy2(p_params, obs, key)
        a = a.astype(int)
        ## --
        
        a = np.random.choice(env.action_space.n)
        obs2, r, done, _ = env.step(a)
        
        observ.append(obs)
        action.append(a)
        rew.append(r)
        
        if done: break 
        obs = obs2 

    obs = jnp.stack(observ)
    a = jnp.stack(action)
    r = jnp.stack(rew)
    return obs, a, r

def jax_loss(p_params, obs, a, r):
    a_probs = p_frwd(p_params, obs)
    log_prob = distrax.Categorical(probs=a_probs).log_prob(a.astype(int))
    loss = -(log_prob * r).sum()
    return loss 

def batch_jax_loss(params, obs, a, r):
    return jax.vmap(partial(jax_loss, params))(obs, a, r).sum()

rng = jax.random.PRNGKey(seed)

#%%
#### PYTORCH 
times = []
for _ in range(50):
    start = time.time()
    
    loss = torch_rollout() 
    loss.backward() 

    times.append(time.time() - start)

# 0.03423449039459228
print(f'PYTORCH TIME: {np.mean(times)}')

#%%
#### JAX 
# rollout_loss fcn 
jax_rollout_jitgrad = jax.jit(jax.value_and_grad(jax_rollout))

times = []
for _ in range(50):
    rng = jax.random.PRNGKey(seed)
    
    start = time.time()
    
    rng, key = jax.random.split(rng, 2)  
    loss, grad = jax_rollout_jitgrad(p_params, key)
    loss.block_until_ready()

    times.append(time.time() - start)

# 0.21324730396270752
print(f'JAX (rollout_loss) TIME: {np.mean(times)}')

#%%
#### JAX 
# rollout fcn & loss fcn
jit_jax_rollout2 = jax.jit(jax_rollout2)
jax_loss_jit = jax.jit(jax.value_and_grad(batch_jax_loss))

times = []
for _ in range(50):
    rng = jax.random.PRNGKey(seed)
    start = time.time()
    
    rng, key = jax.random.split(rng, 2)  
    # batch = jit_jax_rollout2(p_params, key)
    batch = jax_rollout2(p_params, key)

    loss, grad = jax_loss_jit(p_params, *batch)
    loss.block_until_ready()

    times.append(time.time() - start)

# 0.10453275203704834 with jit_jax_rollout2
# 0.07715171337127685 with **no-jit**-rollout2
print(f'JAX (rollout -> loss) TIME: {np.mean(times)}')

#%%