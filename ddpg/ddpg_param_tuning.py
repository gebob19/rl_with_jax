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
env_name = 'CartPoleContinuousBulletEnv-v0'
# env_name = 'Pendulum-v0'

env = gym.make(env_name)
n_actions = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]

a_high = env.action_space.high[0]
a_low = env.action_space.low[0]

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
        middle_layer(256), 
        middle_layer(256),
        hk.Linear(n_actions, w_init=init_final), 
        jax.nn.tanh, 
    ])
    a = policy(s) * a_high # scale to action range 
    return a

def _q_fcn(s, a):
    z1 = middle_layer(16)(s)
    z1 = middle_layer(32)(s)

    z2 = middle_layer(32)(a)
    z = np.concatenate([z1, z2])
    z = middle_layer(256)(z)
    z = middle_layer(256)(z)
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
    a_t = p_frwd(p_params_t, obs2)
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

class OU_Noise:
    def __init__(self, shape):
        self.shape = shape 
        self.theta = 0.15
        self.dt = 1e-2
        self.sigma = 0.2 
        self.mean = onp.zeros(shape)
        self.reset() # set prev 
    
    def sample(self):
        noise = onp.random.normal(size=self.shape)
        x = (
            self.prev
            + self.theta * (self.mean - self.prev) * self.dt
            + self.sigma * onp.sqrt(self.dt) * noise
        )
        self.prev = x 
        return x 

    def reset(self): self.prev = onp.zeros(self.shape)

#%%
import optuna

def train(trial, study_name):
    n_episodes = 1000
    batch_size = trial.suggest_int("batch_size", 32, 256, step=32, log=False) 
    buffer_size = 1e6
    seed = 100 # 420 works
    
    global tau, gamma
    gamma = 0.99 
    # tau = 0.005 # 1e-4 ## very important parameter -- make or break 
    tau = trial.suggest_float("tau", 1e-4, 1e-1, log=True)

    # policy_lr = 1e-3
    # q_lr = 2e-3
    policy_lr = trial.suggest_float("policy_lr", 1e-4, 1e-1, log=True)
    q_lr = trial.suggest_float("q_lr", 1e-4, 1e-1, log=True)

    # metric writer 
    writer = SummaryWriter(comment=study_name)

    rng = jax.random.PRNGKey(seed)
    onp.random.seed(seed)
    random.seed(seed)

    ## model defn
    global p_frwd, q_frwd, p_optim, q_optim

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
    eps = 1
    eps_decay = 1/n_episodes

    vbuffer = Vector_ReplayBuffer(buffer_size)
    action_noise = OU_Noise((n_actions,))

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
    total_reward_sum = []
    for epi_i in tqdm(range(n_episodes)):

        action_noise.reset()
        obs = env.reset()
        rewards = []
        while True: 
            # rollout
            p_params = params[0]
            a = p_frwd(p_params, obs) + eps * action_noise.sample()
            a = np.clip(a, a_low, a_high)
            obs2, r, done, _ = env.step(a)
            vbuffer.push((obs, a, obs2, r, done))
            obs = obs2
            rewards.append(r)
            
            # update
            if not vbuffer.is_ready(batch_size): continue
            batch = vbuffer.sample(batch_size)
            params, opt_states, losses, _ = ddpg_step(params, opt_states, batch)

            p_loss, q_loss = losses
            writer.add_scalar('loss/policy', p_loss.item(), step_i)
            writer.add_scalar('loss/q_fcn', q_loss.item(), step_i)
                
            step_i += 1 
            if done: break 
        
        writer.add_scalar('rollout/total_reward', sum(rewards), epi_i)

        eps -= eps_decay # decay exploration 
        total_reward_sum.append(sum(rewards))

        reward_metric = sum(total_reward_sum[-10:])

        trial.report(reward_metric, epi_i)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return reward_metric # thing to maximize 

import logging
import sys

#%%
if __name__ == '__main__':
    study_name = 'ddpg_cartpole'

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage_name = "sqlite:///{}.db".format(study_name)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15)
    
    # note: will not be persistant without sqlDB storage 
    study = optuna.create_study( 
        direction="maximize",
        study_name=study_name,
        pruner=pruner, 
        storage=storage_name,
    )
    study.optimize(lambda trial: train(trial, study_name), n_trials=10)

    import pprint
    best_params = study.best_params
    pprint.pprint(best_params)
