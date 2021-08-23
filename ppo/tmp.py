#%%
import jax 
import jax.numpy as jnp 
import numpy as onp 
import distrax 
import optax
import gym 
from functools import partial
import cloudpickle

from jax.config import config
config.update("jax_enable_x64", True) 
config.update("jax_debug_nans", True) # break on nans

import brax
from brax.envs import _envs, create_gym_env

for env_name, env_class in _envs.items():
    env_id = f"brax_{env_name}-v0"
    entry_point = partial(create_gym_env, env_name=env_name)
    if env_id not in gym.envs.registry.env_specs:
        print(f"Registring brax's '{env_name}' env under id '{env_id}'.")
        gym.register(env_id, entry_point=entry_point)

#%%
batch_size = 2
env = gym.make("brax_halfcheetah-v0", batch_size=batch_size, \
    episode_length=10) # this is very slow
1

#%%
buffer.clear()
obs = env.reset()  # this can be relatively slow (~10 secs)
for _ in range(100):
    a = onp.random.randn(*env.action_space.shape) 
    obs2, r, done, _ = env.step(a)
    sample = (obs, a, r, obs2, done, r)
    buffer.push(sample)
    if done.any(): break 

(obs, a, r, obs2, done, r) = buffer.contents()
done

#%%
#%%
obs_dim = env.observation_space.shape[-1]
n_actions = env.action_space.shape[-1]

class Vector_ReplayBuffer:
    def __init__(self, buffer_capacity):
        self.buffer_capacity = buffer_capacity = int(buffer_capacity)
        self.i = 0
        # obs, obs2, a, r, done
        self.splits = [obs_dim, obs_dim+n_actions, obs_dim+n_actions+1, obs_dim*2+1+n_actions, obs_dim*2+1+n_actions+1]
        self.clear()

    def push(self, sample):
        assert self.i < self.buffer_capacity # dont let it get full
        (obs, a, r, obs2, done, log_prob) = sample
        self.buffer[:, self.i] = onp.concatenate([obs, a, r[:, None], obs2, done.astype(float)[:, None], log_prob[:, None]], -1)
        self.i += 1 

    def contents(self):
        return onp.split(self.buffer[:, :self.i], self.splits, axis=-1)

    def clear(self):
        self.i = 0 
        self.buffer = onp.zeros((batch_size, self.buffer_capacity, 2 * obs_dim + n_actions + 2 + 1))

buffer = Vector_ReplayBuffer(1e6)

#%%
class Worker:
    def __init__(self, n_steps):
        self.n_steps = n_steps
        self.buffer = Vector_ReplayBuffer(n_step_rollout)
        self.env = make_env()
        self.obs = self.env.reset()

    def rollout(self, p_params, v_params, rng):
        self.buffer.clear()

        for _ in range(self.n_steps): # rollout 
            rng, subrng = jax.random.split(rng)
            a, log_prob = policy(p_params, self.obs, subrng)
            a = onp.array(a)
            
            obs2, r, done, _ = self.env.step(a)

            self.buffer.push((self.obs, a, r, obs2, done, log_prob))
            self.obs = obs2
            if done: 
                self.obs = self.env.reset()

        # update rollout contents 
        rollout = self.buffer.contents()
        advantages, v_target = compute_advantage_targets(v_params, rollout)
        (obs, a, r, _, _, log_prob) = rollout
        rollout = (obs, a, log_prob, v_target, advantages)

        return rollout