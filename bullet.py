#%%
import pybullet as p 
import pybullet_envs
from numpngw import write_apng
import numpy as np 
import gym
import matplotlib.pyplot as plt 

#%%
env = gym.make('CartPoleContinuousBulletEnv-v0')
# env = gym.make('AntBulletEnv-v0')
obs = env.reset()
obs.shape

#%%
imgs = []
vs, rs = [], []

for _ in range(200):
    img = env.render(mode='rgb_array')
    imgs.append(img)

    # velocity = np.linalg.norm(env.unwrapped.robot_body.speed())
    # vs.append(velocity)

    _, r, _, _ = env.step(env.action_space.sample())
    rs.append(r)

env.close()

print(f'writing len {len(imgs)}...')
write_apng('tmp.png', imgs, delay=20)

#%%
plt.plot(vs)
plt.plot(rs)
plt.show()

#%%