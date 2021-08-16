#%%
import numpy as np
import gym

from gym import spaces
from gym.utils import seeding

class Navigation2DEnv_Disc(gym.Env):
    def __init__(self, task={}, low=-0.5, high=0.5, max_n_steps=100):
        super().__init__()
        self.low = low
        self.high = high
        self.max_n_steps = max_n_steps
        self._step_count = 0 

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
            shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(4) # left, right, up, down

        self._task = task
        self._goal = task.get('goal', np.zeros(2, dtype=np.float32))
        self._state = np.zeros(2, dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_tasks(self, num_tasks):
        while True: 
            goals = self.np_random.uniform(self.low, self.high, size=(num_tasks, 2))
            if not (goals.sum(0) == 0).any(): break 

        goals = np.round_(goals, 1) # discrete them to 0.1 steps 
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal = task['goal']

    def reset(self):
        self._step_count = 0 
        self._state = np.zeros(2, dtype=np.float32)
        return self._state

    def step(self, action):
        assert self.action_space.contains(action)
        # up down left right
        step = np.array({
            0: [0, 0.1], 
            1: [0, -0.1],
            2: [-0.1, 0],
            3: [0.1, 0] 
        }[action])

        self._state = self._state + step

        diff = self._state - self._goal
        reward = -np.sqrt((diff**2).sum())
        done = (np.abs(diff) < 0.01).sum() == 2 

        done = done or self._step_count >= self.max_n_steps
        self._step_count += 1 

        return self._state, reward, done, {'task': self._task}

# %%
env = Navigation2DEnv_Disc()
task = env.sample_tasks(1)[0]
env.reset_task(task)
print(task)

# %%
x, y = env._goal
n_right = x / 0.1 
n_up = y / 0.1 
action_seq = []
for _ in range(int(abs(n_right)+.5)): action_seq.append(3 if n_right > 0 else 2)
for _ in range(int(abs(n_up)+.5)): action_seq.append(0 if n_up > 0 else 1)
action_seq

# %%
states = []
obs = env.reset()
states.append(obs)
for a in [2, 3] + action_seq:
    obs, r, done, _ = env.step(a)
    states.append(obs)
    print(r, done)
    if done: break 
states = np.stack(states)

# %%
import matplotlib.pyplot as plt 
plt.scatter(x, y, marker='*')
plt.scatter(states[:, 0], states[:, 1], color='b')
plt.plot(states[:, 0], states[:, 1], color='b')
plt.savefig('tmp.png')
plt.clf()
import cv2 
img = cv2.imread('tmp.png')
plt.imshow(img)

# %%
# %%
# %%


%%
