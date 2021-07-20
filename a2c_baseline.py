#%%
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("CartPole-v1", n_envs=1)
model = A2C("MlpPolicy", env, n_steps=200, verbose=1, tensorboard_log="./tmp/")

model.learn(total_timesteps=1000*200)

#%%
model.policy

#%%
#%%