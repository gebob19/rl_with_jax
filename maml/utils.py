import numpy as onp 
import jax.numpy as np 
import jax 

class Cont_Vector_ReplayBuffer:
    def __init__(self, env, buffer_capacity):
        n_actions = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]

        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.buffer_capacity = buffer_capacity = int(buffer_capacity)
        self.i = 0
        # obs, a, r, obs2, done
        self.splits = [obs_dim, obs_dim+n_actions, obs_dim+n_actions+1, obs_dim*2+1+n_actions, obs_dim*2+1+n_actions+1]
        self.split_names = ['obs', 'a', 'r', 'obs2', 'done', 'log_prob']
        self.clear()

    def push(self, sample):
        assert self.i < self.buffer_capacity # dont let it get full
        (obs, a, r, obs2, done, log_prob) = sample
        self.buffer[self.i] = onp.array([*obs, *onp.array(a), onp.array(r), *obs2, float(done), onp.array(log_prob)])
        self.i += 1 

    def contents(self):
        contents = onp.split(self.buffer[:self.i], self.splits, axis=-1)
        d = {}
        for n, c in zip(self.split_names, contents): d[n] = c
        return d 

    def clear(self):
        self.i = 0 
        self.buffer = onp.zeros((self.buffer_capacity, 2 * self.obs_dim + self.n_actions + 2 + 1))

def discount_cumsum(l, discount):
    l = onp.array(l)
    for i in range(len(l) - 1)[::-1]:
        l[i] = l[i] + discount * l[i+1]
    return l 

def tree_shape(tree): 
    for l in jax.tree_leaves(tree): print(l.shape)