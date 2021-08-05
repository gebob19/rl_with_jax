#%%
import gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env

from baselines.common.atari_wrappers import *

import collections
import numpy as np 
import cv2 

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip
    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info
    def reset(self):
          self._obs_buffer.clear()
          obs = self.env.reset()
          self._obs_buffer.append(obs)
          return obs

class ProcessFrame84(gym.ObservationWrapper):
    """
    Downsamples image to 84x84
    Greyscales image
    Returns numpy array
    """
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            rgb_img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            rgb_img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        
#        cv2.imwrite('./rgb_image2.png', rgb_img)
        ## Image Resizing to crop the not needed space
        rgb_img = cv2.resize(rgb_img, (84, 110), interpolation=cv2.INTER_AREA)
        rgb_img = rgb_img[18:102, :]
        
        ## Conversion from RGB to Gray Scale --> b as [0.2989, 0.5870, 0.1140]
        grayscale_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

#        Gray_Scale_Parameters = [0.2989, 0.5870, 0.1140]
#        grayscale_img = rgb_img[:, :, 0] * Gray_Scale_Parameters[0] + \
#              rgb_img[:, :, 1] * Gray_Scale_Parameters[1] + \
#              rgb_img[:, :, 2] * Gray_Scale_Parameters[2]
        
        ## Conversion from Gray Scale to Binary --> thershold as 127
        thresh = 127
        binary_img = cv2.threshold(grayscale_img, thresh, 255, cv2.THRESH_BINARY)[1]

        ## Image Resizing to crop the not needed space
#        binary_img = cv2.resize(binary_img, (84, 110), interpolation=cv2.INTER_AREA)
#        binary_img = binary_img[18:102, :]
        binary_img = np.reshape(binary_img, [84, 84, 1])
        
        return binary_img.astype(np.uint8)

class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
      ## moveaxis --> Move axes of an array to new positions
      ## as the Conv2d takes the image argument as (channels,height, width)
        return np.moveaxis(observation, 2, 0)

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.uint8):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=-1),
                                                old_space.high.repeat(n_steps, axis=-1), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:, :, :-1] = self.buffer[:, :, 1:]
        self.buffer[:, :, -1] = observation[:, :, 0]
        return self.buffer

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
def make_env():
    env = gym.make(env_name)
    # env = wrap_deepmind(env)

    # env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = NoopResetEnv(env, noop_max=50)
    env = WarpFrame(env)
    # env = BufferWrapper(env, 4)
    # env = ScaledFloatFrame(env)
    env = DiffFrame(env)
    # env = FlattenObs(env)
    return env 

# def make_env():
#     env = gym.make(env_name)
#     env = MaxAndSkipEnv(env)
#     env = FireResetEnv(env)
#     env = ProcessFrame84(env)
#     env = BufferWrapper(env, 4)
#     return env 

#%%
# Parallel environments
env = make_vec_env(make_env, n_envs=16)
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./baselines/")

model.learn(total_timesteps=int(1e7))

#%%
#%%