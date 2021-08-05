#%%
import jax
import jax.numpy as np 
import numpy as onp 
import distrax 
import optax
import gym 
from functools import partial

#%%
env_name = 'CartPole-v0'
env = gym.make(env_name)

n_actions = env.action_space.n 
obs_dim = env.observation_space.shape[0]

# n_actions = env.action_space.shape[0]
# obs_dim = env.observation_space.shape[0]

# a_high = env.action_space.high[0]
# a_low = env.action_space.low[0]

# print(f'[LOGGER] a_high: {a_high} a_low: {a_low} n_actions: {n_actions} obs_dim: {obs_dim}')
# assert -a_high == a_low

#%%
import haiku as hk 

def _policy_fcn(obs):
    probs = hk.Sequential([
        hk.Linear(32), jax.nn.relu,
        hk.Linear(32), jax.nn.relu,
        hk.Linear(n_actions), jax.nn.softmax,
    ])(obs)
    return probs 

policy_fcn = hk.transform(_policy_fcn)
policy_fcn = hk.without_apply_rng(policy_fcn)
p_frwd = jax.jit(policy_fcn.apply)

class Vector_ReplayBuffer:
    def __init__(self, buffer_capacity):
        self.buffer_capacity = buffer_capacity = int(buffer_capacity)
        self.i = 0
        # obs, obs2, a, r, done
        self.splits = [obs_dim, obs_dim+1, obs_dim+1+1, obs_dim*2+1+1, obs_dim*2+1+1+1]
        self.clear()

    def push(self, sample):
        assert self.i < self.buffer_capacity # dont let it get full
        (obs, a, r, obs2, done, log_prob) = sample
        self.buffer[self.i] = onp.array([*obs, onp.array(a), onp.array(r), *obs2, float(done), onp.array(log_prob)])
        self.i += 1 

    def contents(self):
        return onp.split(self.buffer[:self.i], self.splits, axis=-1)

    def clear(self):
        self.i = 0 
        self.buffer = onp.zeros((self.buffer_capacity, 2 * obs_dim + 1 + 2 + 1))

#%%
seed = onp.random.randint(1e5)
policy_lr = 1e-3

rng = jax.random.PRNGKey(seed)
onp.random.seed(seed)

obs = env.reset() # dummy input 
p_params = policy_fcn.init(rng, obs) 

## optimizers 
optimizer = lambda lr: optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.scale_by_adam(),
    optax.scale(-lr),
)
p_optim = optimizer(policy_lr)
p_opt_state = p_optim.init(p_params)

buffer = Vector_ReplayBuffer(500)

#%%
def reward2go(r, gamma=0.99):
    for i in range(len(r) - 1)[::-1]:
        r[i] = r[i] + gamma * r[i+1]
    r = (r - r.mean()) / r.std()
    return r 

def rollout(p_params, rng):
    buffer.clear()
    obs = env.reset()
    while True: 
        probs = p_frwd(p_params, obs)

        rng, subkey = jax.random.split(rng, 2)
        dist = distrax.Categorical(probs)
        a = dist.sample(seed=subkey).item()

        obs2, r, done, _ = env.step(a)        
        obs = obs2 

        buffer.push((obs, a, r, obs2, done, 0))
        if done: break 
    
    (obs, a, r, obs2, done, _) = buffer.contents()

    return (obs, a, r)

def policy_loss(p_params, batch):
    (obs, a, r) = batch
    probs = p_frwd(p_params, obs)
    dist = distrax.Categorical(probs)
    log_prob = dist.log_prob(a)

    loss = -(log_prob * r).sum()
    return loss 

# %%
rng, subkey = jax.random.split(rng, 2)
(obs, a, r) = rollout(p_params, subkey)
r = reward2go(r)

# %%

# %%
# %%
# %%
