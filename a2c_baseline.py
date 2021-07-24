#%%
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

from baselines.common.atari_wrappers import *

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

# env_name = 'Pong-v0'
env_name = 'PongNoFrameskip-v4'

import numpy as onp 
# def make_env():
#     env = gym.make(env_name)
#     env = FireResetEnv(env)
#     env = NoopResetEnv(env, noop_max=50)
#     env = WarpFrame(env)
#     # env = ScaledFloatFrame(env)
#     env = DiffFrame(env)
#     # env = FlattenObs(env)
#     return env 

def make_env():
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = BufferWrapper(env, 4)
    return 

# env = make_env()

# Parallel environments
env = make_vec_env(make_env, n_envs=16)
model = A2C("CnnPolicy", env, verbose=1, tensorboard_log="./baselines/")

model.learn(total_timesteps=int(1e6))

#%%
#%%