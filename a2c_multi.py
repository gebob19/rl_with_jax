#%%
import gym 
import ray 
import jax 
import jax.numpy as np 
import numpy as onp 
import haiku as hk 
import optax
from functools import partial

#%%
ray.init()

env_name = 'CartPole-v0'
env = gym.make(env_name)

n_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]

def _policy_value(obs):
    pi = hk.Sequential([
        hk.Linear(64), jax.nn.relu, 
        hk.Linear(64), jax.nn.relu,
        hk.Linear(n_actions), jax.nn.softmax
    ])(obs)
    v = hk.Sequential([
        hk.Linear(64), jax.nn.relu, 
        hk.Linear(64), jax.nn.relu,
        hk.Linear(1),
    ])(obs)
    return pi, v

policy_value = hk.transform(_policy_value)
policy_value = hk.without_apply_rng(policy_value)
pv_frwd = jax.jit(policy_value.apply)

class Categorical: # similar to pytorch categorical
    def __init__(self, probs):
        self.probs = probs
    def sample(self): 
        # https://stackoverflow.com/questions/46539431/np-random-choice-probabilities-do-not-sum-to-1
        p = onp.asanyarray(self.probs)
        p = p / p.sum()
        a = onp.random.choice(onp.arange(len(self.probs)), p=p)
        return a 
    def log_prob(self, i): return np.log(self.probs[i])
    def entropy(self): return -(self.probs * np.log(self.probs)).sum()

@ray.remote
class Worker: 
    def __init__(self, env_name, gamma=0.99, max_n_steps=200):
        self.env = gym.make(env_name)
        self.max_n = max_n_steps
        self.gamma = gamma
        # create jax policy fcn -- need to define in .remote due to pickling 
        policy_value = hk.transform(_policy_value)
        policy_value = hk.without_apply_rng(policy_value)
        self.pv_frwd = jax.jit(policy_value.apply) # forward fcn

    def rollout(self, params):
        # obs, obs2 + a, r, done, 
        v_buffer = onp.zeros((self.max_n, 2 * obs_dim + 3))

        obs = self.env.reset()
        for i in range(self.max_n):
            a_probs, _ = self.pv_frwd(params, obs)
            a = Categorical(a_probs).sample() # stochastic sample

            obs2, r, done, _ = self.env.step(a)        
            v_buffer[i] = onp.array([*obs, a, r, *obs2, float(done)])

            obs = obs2 
            if done: break 
        
        v_buffer = v_buffer[:i+1]
        obs, a, r, obs2, done = onp.split(v_buffer, [obs_dim, obs_dim+1, obs_dim+2, obs_dim*2+2], axis=-1)

        for i in range(len(r) - 1)[::-1]:
            r[i] = r[i] + self.gamma * r[i + 1]

        return obs, a, r, obs2, done

#%%
def eval(params, env):
    rewards = 0 
    obs = env.reset()
    while True: 
        a_probs, _ = pv_frwd(params, obs)
        a_dist = Categorical(a_probs)
        a = a_dist.sample()
        obs2, r, done, _ = env.step(a)        
        obs = obs2 
        rewards += r
        if done: break 
    return rewards

def policy_loss(params, obs, a, r):
    a_probs, v_s = pv_frwd(params, obs)
    a_dist = Categorical(a_probs)

    log_prob = a_dist.log_prob(a.astype(np.int32))
    advantage = jax.lax.stop_gradient(r - v_s)
    policy_loss = -(log_prob * advantage).sum()
    
    entropy_loss = -0.001 * a_dist.entropy()
    return policy_loss + entropy_loss

def critic_loss(params, obs, r):
    _, v_s = pv_frwd(params, obs)
    return ((v_s - r) ** 2).sum()

def a2c_loss(params, sample):
    obs, a, r, _, _ = sample
    ploss = policy_loss(params, obs, a, r)
    vloss = critic_loss(params, obs, r)
    loss = ploss + 0.25 * vloss
    return loss, ploss, vloss
    
def batch_a2c_loss(params, samples):
    loss, ploss, vloss = jax.vmap(partial(a2c_loss, params))(samples)
    return loss.mean(), (ploss.mean(), vloss.mean())

@jax.jit
def a2c_step(samples, params, opt_state):
    (loss, (ploss, vloss)), grad = jax.value_and_grad(batch_a2c_loss, has_aux=True)(params, samples)
    grad, opt_state = optim.update(grad, opt_state)
    params = optax.apply_updates(params, grad)
    return loss, ploss, vloss, opt_state, params, grad

#%%
seed = 0
rng = jax.random.PRNGKey(seed)

obs = env.reset() # dummy input 
a = np.zeros(env.action_space.shape)
params = policy_value.init(rng, obs) 

optim = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.scale_by_adam(),
    optax.scale(-1e-3),
)
opt_state = optim.init(params)

n_envs = 4
worker = Worker.remote(env_name)

from tqdm.notebook import tqdm 

for i in tqdm(range(1000)):
    rollouts = ray.get([worker.rollout.remote(params) for _ in range(n_envs)])
    samples = jax.tree_multimap(lambda *a: np.concatenate(a), *rollouts, is_leaf=lambda node: hasattr(node, 'shape'))

    loss, ploss, vloss, opt_state, params, grads = a2c_step(samples, params, opt_state)

    r = eval(params, env)
    print(f'episode {i}: {r}')

#%%
#%%