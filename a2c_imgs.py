#%%
import gym
from matplotlib import pyplot as plt 
import ray 
import jax 
import jax.numpy as np 
import numpy as onp 
import haiku as hk 
import optax
from functools import partial

from baselines.common.atari_wrappers import FireResetEnv, WarpFrame, \
    ScaledFloatFrame, NoopResetEnv

class DiffFrame(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_frame = None 
    
    def reset(self):
        obs = self.env.reset()
        obs2, _, _, _ = self.env.step(0) # NOOP 
        obs = obs2 - obs 
        self.prev_frame = obs2 
        return obs 
    
    def step(self, a):
        obs2, r, done, info = self.env.step(a) 
        obs = obs2 - self.prev_frame
        self.prev_frame = obs2 
        return obs, r, done, info

class FlattenObs(gym.ObservationWrapper):
    def observation(self, obs):
        return obs.flatten()

#%%
env_name = 'Pong-v0'
env = gym.make(env_name)
env = FireResetEnv(env)
env = NoopResetEnv(env, noop_max=50)
env = WarpFrame(env)
# env = ScaledFloatFrame(env)
env = DiffFrame(env)
# env = FlattenObs(env)

#%%
obs = env.reset()
obs.shape

#%%
obs = env.reset()
for _ in range(20):
    obs, _, done, _ = env.step(env.action_space.sample())
    plt.title(f'{obs.max()}, {obs.min()}')
    plt.imshow(obs)
    plt.show()
    if done: break 

# %%
n_actions = env.action_space.n
obs = env.reset()
obs_dim = obs.shape[0]

def _policy_value(obs):
    backbone = hk.Sequential([
        hk.Conv2D(16, 8, 4), jax.nn.relu, 
        hk.Conv2D(32, 4, 2), jax.nn.relu, 
    ])
    z = backbone(obs)
    z = np.reshape(z, (-1,))

    pi = hk.Sequential([
        hk.Linear(128), jax.nn.relu, 
        hk.Linear(128), jax.nn.relu,
        hk.Linear(n_actions), jax.nn.softmax
    ])(z)
    v = hk.Sequential([
        hk.Linear(128), jax.nn.relu, 
        hk.Linear(128), jax.nn.relu,
        hk.Linear(1),
    ])(z)
    return pi, v

policy_value = hk.transform(_policy_value)
policy_value = hk.without_apply_rng(policy_value)
pv_frwd = jax.jit(policy_value.apply)

seed = 0 
rng = jax.random.PRNGKey(seed)

obs = env.reset() # dummy input 
a = np.zeros(env.action_space.shape)
params = policy_value.init(rng, obs) 

#%%
pi, v = pv_frwd(params, obs)
pi.shape

#%%
#%%
