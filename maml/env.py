#%%
import numpy as np
import gym

from gym import spaces
from gym.utils import seeding

class Navigation2DEnv(gym.Env):
    """2D navigation problems, as described in [1]. The code is adapted from 
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/maml_examples/point_env_randgoal.py
    At each time step, the 2D agent takes an action (its velocity, clipped in
    [-0.1, 0.1]), and receives a penalty equal to its L2 distance to the goal 
    position (ie. the reward is `-distance`). The 2D navigation tasks are 
    generated by sampling goal positions from the uniform distribution 
    on [-0.5, 0.5]^2.
    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, task={}, low=-0.5, high=0.5, max_n_steps=100):
        super().__init__()
        self.low = low
        self.high = high
        self.max_n_steps = max_n_steps
        self._step_count = 0 

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
            shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1,
            shape=(2,), dtype=np.float32)

        self._task = task
        self._goal = task.get('goal', np.zeros(2, dtype=np.float32))
        self._state = np.zeros(2, dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_tasks(self, num_tasks):
        goals = self.np_random.uniform(self.low, self.high, size=(num_tasks, 2))
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
        action = np.clip(action, -0.1, 0.1)
        action = np.array(action)
        assert self.action_space.contains(action)
        self._state = self._state + action

        diff = self._state - self._goal
        reward = -np.sqrt((diff**2).sum())
        done = (np.abs(diff) < 0.01).sum() == 2 

        done = done or self._step_count >= self.max_n_steps
        self._step_count += 1 

        return self._state, reward, done, {'task': self._task}

class Navigation2DEnv_Disc(gym.Env):
    def __init__(self, task={}, low=-0.5, high=0.5, max_n_steps=100):
        super().__init__()
        self.low = low
        self.high = high
        self.max_n_steps = max_n_steps
        self._step_count = 0 

        self.observation_space = spaces.Box(low=self.low, high=self.high,
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
        self._state = np.clip(self._state, self.low, self.high)

        diff = self._state - self._goal
        reward = -np.sqrt((diff**2).sum())
        done = (np.abs(diff) < 0.01).sum() == 2 

        done = done or self._step_count >= self.max_n_steps
        self._step_count += 1 

        return self._state, reward, done, {'task': self._task}
