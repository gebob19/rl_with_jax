#%%
import jax
import jax.numpy as np 
import numpy as onp 
import haiku as hk
import optax
import gym 
import copy 
import matplotlib.pyplot as plt 
from torch.utils.tensorboard import SummaryWriter

# from tqdm.notebook import tqdm 
from tqdm import tqdm 

import collections
import random 
from functools import partial

import pybullet as p 
import pybullet_envs
from numpngw import write_apng

from jax.config import config
config.update("jax_debug_nans", True) # break on nans

#%%
# env_name = 'AntBulletEnv-v0'
# env_name = 'CartPoleContinuousBulletEnv-v0'

# env_name = 'Pendulum-v0' ## works for this env with correct seed :o
env_name = 'HalfCheetahBulletEnv-v0'

env = gym.make(env_name)
n_actions = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]

a_high = env.action_space.high[0]
a_low = env.action_space.low[0]

print(f'[LOGGER] a_high: {a_high} a_low: {a_low} n_actions: {n_actions} obs_dim: {obs_dim}')

#%%
class FanIn_Uniform(hk.initializers.Initializer):
    def __call__(self, shape, dtype): 
        bound = 1/(shape[0] ** .5)
        return hk.initializers.RandomUniform(-bound, bound)(shape, dtype)

init_other = FanIn_Uniform()
init_final = hk.initializers.RandomUniform(-3e-3, 3e-3)

def middle_layer(d):
    return hk.Sequential([
        hk.Linear(d, w_init=init_other), jax.nn.relu,
    ])

def _policy_fcn(s):
    policy = hk.Sequential([
        middle_layer(400), 
        middle_layer(300),
        hk.Linear(n_actions, w_init=init_final), 
        jax.nn.tanh, 
    ])
    a = policy(s) * a_high # scale to action range 
    return a

def _q_fcn(s, a):
    z = np.concatenate([s, a])
    z = middle_layer(400)(z)
    z = np.concatenate([z, a])
    z = middle_layer(300)(z)
    q_sa = hk.Linear(1, w_init=init_final)(z)
    return q_sa

#%%
class ReplayBuffer(object): # clean code BUT extremely slow 
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=int(capacity))

    def push(self, sample):
        self.buffer.append(sample)

    def sample(self, batch_size):
        samples = zip(*random.sample(self.buffer, batch_size))
        samples = tuple(map(lambda x: np.stack(x).astype(np.float32), samples))
        return samples

    def is_ready(self, batch_size):
        return batch_size <= len(self.buffer)

#%%
class Vector_ReplayBuffer:
    def __init__(self, buffer_capacity):
        buffer_capacity = int(buffer_capacity)
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]
        self.state_buffer = onp.zeros((self.buffer_capacity, num_states))
        self.action_buffer = onp.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = onp.zeros((self.buffer_capacity, 1))
        self.dones = onp.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = onp.zeros((self.buffer_capacity, num_states))
        self.buffers = [self.state_buffer, self.action_buffer, self.next_state_buffer, self.reward_buffer, self.dones]

    def push(self, obs_tuple):
        # (obs, a, obs2, r, done)
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.next_state_buffer[index] = obs_tuple[2]
        self.reward_buffer[index] = obs_tuple[3]
        self.dones[index] = float(obs_tuple[4]) # dones T/F -> 1/0

        self.buffer_counter += 1

    def is_ready(self, batch_size): return self.buffer_counter >= batch_size

    def sample(self, batch_size):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = onp.random.choice(record_range, batch_size)
        batch = tuple(b[batch_indices] for b in self.buffers)
        return batch

#%%
def critic_loss(q_params, target_params, sample):
    p_params_t, q_params_t = target_params
    obs, a, obs2, r, done = sample
    # pred 
    q = q_frwd(q_params, obs, a)
    # target
    a_t = p_frwd(p_params_t, obs2) + action_noise.sample()
    q_t = q_frwd(q_params_t, obs2, a_t)
    y = r + (1 - done) * gamma * q_t
    y = jax.lax.stop_gradient(y)

    loss = (q - y) ** 2
    return loss 

def policy_loss(p_params, q_params, sample):
    obs, _, _, _, _ = sample
    a = p_frwd(p_params, obs)
    return -q_frwd(q_params, obs, a)

def batch_critic_loss(q_params, target_params, batch):
    return jax.vmap(partial(critic_loss, q_params, target_params))(batch).mean()

def batch_policy_loss(p_params, q_params, batch):
    return jax.vmap(partial(policy_loss, p_params, q_params))(batch).mean()

#%%
@jax.jit
def ddpg_step(params, opt_states, batch):
    p_params, q_params, p_params_t, q_params_t = params
    p_opt_state, q_opt_state = opt_states

    # update q/critic
    target_params = (p_params_t, q_params_t)
    q_loss, q_grad = jax.value_and_grad(batch_critic_loss)(q_params, target_params, batch)
    q_grad, q_opt_state = q_optim.update(q_grad, q_opt_state)
    q_params = optax.apply_updates(q_params, q_grad)

    # update policy 
    p_loss, p_grad = jax.value_and_grad(batch_policy_loss)(p_params, q_params, batch)
    p_grad, p_opt_state = p_optim.update(p_grad, p_opt_state)
    p_params = optax.apply_updates(p_params, p_grad)

    # slow update targets
    polask_avg = lambda target, w: (1 - tau) * target + tau * w
    p_params_t = jax.tree_multimap(polask_avg, p_params_t, p_params)
    q_params_t = jax.tree_multimap(polask_avg, q_params_t, q_params)

    params = (p_params, q_params, p_params_t, q_params_t) # re-pack with updated q
    opt_states = (p_opt_state, q_opt_state)
    losses = (p_loss, q_loss)
    grads = (p_grad, q_grad)
    return params, opt_states, losses, grads

class Gaussian_Noise:
    def __init__(self, shape):
        self.shape = shape 
        self.sigma = 0.2 
        self.mean = 0
        self.clip_value = 0.5
        self.reset() # set prev 
    
    def sample(self):
        x = onp.random.normal(loc=self.mean, scale=self.sigma, size=self.shape)
        x = onp.clip(x, -self.clip_value, self.clip_value)
        return x 

    def reset(self): pass 

#%%
# pendulum = 100epis
# n_episodes = 100 

total_n_steps = 0.1 * 1e6
batch_size = 100 
buffer_size = 1e6
gamma = 0.99 
tau = 5e-3 ## very important parameter -- make or break 
seed = onp.random.randint(1e5) # on pendulum: seed=78583 converges (-200r) and seed=8171 doesn't (-1000r)
print(f'[LOGGER] using seed={seed}')

policy_lr = 1e-3
q_lr = 1e-3

# metric writer 
writer = SummaryWriter(comment=f'td3_AHE_{env_name}')

rng = jax.random.PRNGKey(seed)
onp.random.seed(seed)
random.seed(seed)

## model defn
# actor
policy_fcn = hk.transform(_policy_fcn)
policy_fcn = hk.without_apply_rng(policy_fcn)
p_frwd = jax.jit(policy_fcn.apply)

# critic
q_fcn = hk.transform(_q_fcn)
q_fcn = hk.without_apply_rng(q_fcn)
q_frwd = jax.jit(q_fcn.apply)

## optimizers 
p_optim = optax.adam(policy_lr)
q_optim = optax.adam(q_lr)

#%%
vbuffer = Vector_ReplayBuffer(buffer_size)
action_noise = Gaussian_Noise((n_actions,))

# init models + optims 
obs = env.reset() # dummy input 
a = np.zeros(env.action_space.shape)
p_params = policy_fcn.init(rng, obs) 
q_params = q_fcn.init(rng, obs, a) 

# target networks
p_params_t = copy.deepcopy(p_params)
q_params_t = copy.deepcopy(q_params)

p_opt_state = p_optim.init(p_params)
q_opt_state = q_optim.init(q_params)

# bundle 
params = (p_params, q_params, p_params_t, q_params_t)
opt_states = (p_opt_state, q_opt_state)

step_i = 0 
epi_i = 0
pbar = tqdm(total=total_n_steps)

while step_i < total_n_steps: 
    action_noise.reset()
    obs = env.reset()
    rewards = []
    while True: 
        # rollout
        p_params = params[0]
        a = p_frwd(p_params, obs) + action_noise.sample()
        a = np.clip(a, a_low, a_high)

        obs2, r, done, _ = env.step(a)
        vbuffer.push((obs, a, obs2, r, done))
        rewards.append(onp.asanyarray(r))

        obs = obs2 ## ** 
        
        # update
        if not vbuffer.is_ready(batch_size): continue
        batch = vbuffer.sample(batch_size)
        params, opt_states, losses, grads = ddpg_step(params, opt_states, batch)

        p_loss, q_loss = losses
        writer.add_scalar('loss/policy', p_loss.item(), step_i)
        writer.add_scalar('loss/q_fcn', q_loss.item(), step_i)
            
        step_i += 1 
        pbar.update(1)
        if done: break 

    # evaluate without any noise 
    eval_rewards = []
    for _ in range(10):
        obs = env.reset()
        epi_reward = 0 
        while True: 
            p_params = params[0]
            a = p_frwd(p_params, obs)
            obs2, r, done, _ = env.step(a)
            epi_reward += r 
            obs = obs2
            if done: break 
        eval_rewards.append(epi_reward)
    eval_r = onp.mean(eval_rewards)
    
    writer.add_scalar('rollout/total_reward', sum(rewards), step_i)
    writer.add_scalar('rollout/mean_eval_reward', eval_r, step_i)
    writer.add_scalar('rollout/length', len(rewards), step_i)

pbar.close()

# %%
# %%
# %%
