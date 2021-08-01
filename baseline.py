#%%
import gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env('Pendulum-v0', n_envs=4)
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tmp/")
model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./tmp/")

model.learn(total_timesteps=1000*200)

#%%
#%%