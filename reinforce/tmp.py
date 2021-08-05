#%%
import jax 
import jax.numpy as jnp 
import haiku as hk 
import distrax

import torch 
from torch.distributions.categorical import Categorical 

import gym 
import numpy as np 

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

@jax.jit    
def jax_policy(p_params, obs, key):
    a_probs = p_frwd(p_params, obs)
    a_probs = distrax.Categorical(probs=a_probs)
    a, log_prob = a_probs.sample_and_log_prob(seed=key)        
    return a, log_prob

#%%
import time 
times = []
for _ in range(100):
    env.seed(seed)
    np.random.seed(seed)
    step = 0 
    obs = env.reset()
    start = time.time()

    log_probs = []
    rewards = []
    while True: 
        ## test 1
        # p_frwd(p_params, obs) ## 0.003257927894592285 
        # policy(torch.from_numpy(obs).float()) ## 0.007266004085540772
        
        # ## test 2
        # rng, key = jax.random.split(rng, 2) ## 0.02236743688583374
        # a, log_prob = jax_policy(p_params, obs, key)
        # a = a.item()
        
        # a, log_prob = torch_policy(obs) ## 0.022551815509796142
        # a = a.numpy()

        a = np.random.choice(env.action_space.n)
        obs2, r, done, _ = env.step(a)
        if done: break 
        obs = obs2 
        step += 1 

        log_probs.append(log_prob)
        rewards.append(r)

    times.append(time.time() - start)
    break 

print(np.mean(times))

#%%
def torch_rollout(env):
    np.random.seed(seed)
    env.seed(seed)
    obs = env.reset()
    log_probs = []
    rewards = []
    while True: 
        a, log_prob = torch_policy(obs) ## 0.022551815509796142
        a = a.numpy()

        a = np.random.choice(env.action_space.n)
        obs2, r, done, _ = env.step(a)
        if done: break 
        obs = obs2 

        log_probs.append(log_prob)
        rewards.append(r)
    
    log_probs = torch.stack(log_probs)
    r = torch.tensor(rewards)
    loss = -(log_probs * r).sum()
    return loss 

def jax_rollout(p_params, rng):
    np.random.seed(seed)
    env.seed(seed)
    obs = env.reset()

    log_probs = []
    rewards = []
    for _ in range(200):
        ## test 
        rng, key = jax.random.split(rng, 2) 
        a, log_prob = jax_policy(p_params, obs, key)
        a = a.astype(int)
        ## rollout
        
        ## for fair comparison 
        a = np.random.choice(env.action_space.n)
        obs2, r, done, _ = env.step(a)
        if done: break 
        obs = obs2 

        log_probs.append(log_prob)
        rewards.append(r)
    
    # compute loss 
    log_prob = jnp.stack(log_probs)
    r = np.stack(rewards)
    loss = -(log_prob * r).sum()
    return loss 

def jax_rollout22(p_params, rng):
    np.random.seed(seed)
    env.seed(seed)
    obs = env.reset()

    log_probs = []
    rewards = []
    for _ in range(200):
        rng, key = jax.random.split(rng, 2) 
        a, log_prob = jax_policy(p_params, obs, key)
        a = a.astype(int)
        
        a = np.random.choice(env.action_space.n)
        obs2, r, done, _ = env.step(a)
        if done: break 
        obs = obs2 

        log_probs.append(log_prob)
        rewards.append(r)
    
    return log_probs, rewards

@jax.jit
def jax_loss33(p_params, rng):
    log_prob, r = jax_rollout22(p_params, rng)    
    log_prob = jnp.stack(log_prob)
    r = jnp.stack(r)
    loss = -(log_prob * r).sum()
    return loss 

@jax.jit    
def jax_policy2(p_params, obs, key):
    a_probs = p_frwd(p_params, obs)
    a_probs = distrax.Categorical(probs=a_probs)
    a = a_probs.sample(seed=key)
    return a

def jax_rollout2(p_params, env, rng):
    np.random.seed(seed)
    env.seed(seed)
    obs = env.reset()

    observ, action, rew = [], [], []
    while True: 
        ## test 2
        rng, key = jax.random.split(rng, 2) 
        a = jax_policy2(p_params, obs, key)
        a = a.item()
        
        a = np.random.choice(env.action_space.n)
        obs2, r, done, _ = env.step(a)
        if done: break 

        observ.append(obs)
        action.append(a)
        rew.append(r)

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

from functools import partial
def batch_jax_loss(params, obs, a, r):
    return jax.vmap(partial(jax_loss, params), 0)(obs, a, r).sum()

jax_vg = jax.value_and_grad(jax_rollout)

jax_loss_jit = jax.jit(jax.value_and_grad(batch_jax_loss))
jax_loss_jit0 = jax.jit(jax_vg)

jax_loss_jit33 = jax.jit(jax.value_and_grad(jax_loss33))

rng = jax.random.PRNGKey(seed)

import time 
times = []
for _ in range(50):
    start = time.time()
    
    # loss = torch_rollout(env) # loss = 0.027692785263061525
    # ## loss & grad = 0.026107568740844727
    # loss.backward() 

    # rng, key = jax.random.split(rng, 2)  
    # loss = jax_rollout(p_params, env, key) # loss = 0.029589149951934814

    # # ## loss & grad = 0.17592523574829103
    # rng, key = jax.random.split(rng, 2)  
    # loss, grad = jax_vg(p_params, key) 

    ## = 0.2892731046676636
    rng, key = jax.random.split(rng, 2)  
    loss, grad = jax_loss_jit33(p_params, key)

    ## loss & grad = 0.23618555545806885
    # rng, key = jax.random.split(rng, 2)  
    # loss, grad = jax_loss_jit0(p_params, key) 
    # loss, grad = jax_loss_jit0(p_params, key) 

    # # ## loss & grad 2 = 0.8245581483840942
    # rng, key = jax.random.split(rng, 2)  
    # batch = jax_rollout2(p_params, env, key)
    # loss, grad = jax.value_and_grad(batch_jax_loss)(p_params, *batch)

    # # ## loss & grad 3 = 0.07825906753540039
    # rng, key = jax.random.split(rng, 2)  
    # batch = jax_rollout2(p_params, env, key)
    # loss, grad = jax_loss_jit(p_params, *batch)

    times.append(time.time() - start)

print(np.mean(times))
#%%
#%%
#%%