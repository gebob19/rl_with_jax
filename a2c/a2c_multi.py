#%%
import gym 
import ray 
import jax 
import jax.numpy as np 
import numpy as onp 
import haiku as hk 
import optax
from functools import partial

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 
import cloudpickle

#%%
ray.init()

env_name = 'CartPole-v0'
env = gym.make(env_name)

n_actions = env.action_space.n
obs = env.reset()
obs_dim = obs.shape[0]

print(f'[LOGGER] obs_dim: {obs_dim} n_actions: {n_actions}')

def _policy_value(obs):
    pi = hk.Sequential([
        hk.Linear(128), jax.nn.relu, 
        hk.Linear(128), jax.nn.relu,
        hk.Linear(n_actions), jax.nn.softmax
    ])(obs)
    v = hk.Sequential([
        hk.Linear(128), jax.nn.relu, 
        hk.Linear(128), jax.nn.relu,
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
    def __init__(self, gamma=0.99):
        self.env = gym.make(env_name)
        self.obs = self.env.reset()

        self.gamma = gamma
        # create jax policy fcn -- need to define in .remote due to pickling 
        policy_value = hk.transform(_policy_value)
        policy_value = hk.without_apply_rng(policy_value)
        self.pv_frwd = jax.jit(policy_value.apply) # forward fcn

    def rollout(self, params, n_steps):
        # obs, obs2 + a, r, done, 
        v_buffer = onp.zeros((n_steps, 2 * obs_dim + 3))

        for i in range(n_steps):
            a_probs, _ = self.pv_frwd(params, self.obs)
            a = Categorical(a_probs).sample() # stochastic sample

            obs2, r, done, _ = self.env.step(a)        
            v_buffer[i] = onp.array([*self.obs, a, r, *obs2, float(done)])

            self.obs = obs2 
            if done: 
                self.obs = self.env.reset()
                break 
        
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
seed = onp.random.randint(1e5)
rng = jax.random.PRNGKey(seed)

import random 
onp.random.seed(seed)
random.seed(seed)

obs = env.reset() # dummy input 
a = np.zeros(env.action_space.shape)
params = policy_value.init(rng, obs) 

optim = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.scale_by_adam(),
    optax.scale(-1e-3),
)
opt_state = optim.init(params)

n_envs = 16
n_steps = 20
print(f'[LOGGER] using batchsize = {n_envs * n_steps}')

workers = [Worker.remote() for _ in range(n_envs)]

writer = SummaryWriter(comment=f'{env_name}_n-envs{n_envs}_seed{seed}')
max_reward = -float('inf')

import pathlib 
model_path = pathlib.Path(f'./models/a2c/{env_name}')
model_path.mkdir(exist_ok=True, parents=True)

for step_i in tqdm(range(1000)):
    rollouts = ray.get([worker.rollout.remote(params, n_steps) for worker in workers])
    samples = jax.tree_multimap(lambda *a: np.concatenate(a), *rollouts, is_leaf=lambda node: hasattr(node, 'shape'))

    loss, ploss, vloss, opt_state, params, grads = a2c_step(samples, params, opt_state)
    writer.add_scalar('loss/policy', ploss.item(), step_i)
    writer.add_scalar('loss/critic', vloss.item(), step_i)
    writer.add_scalar('loss/total', loss.item(), step_i)
    writer.add_scalar('loss/batch_size', samples[0].shape[0], step_i)

    obs, a, r, done, _ = samples 
    a_probs, v_s = jax.vmap(lambda o: pv_frwd(params, o))(obs)
    mean_entropy = jax.vmap(lambda p: Categorical(p).entropy())(a_probs).mean()
    mean_value = v_s.mean()
    writer.add_scalar('policy/mean_entropy', mean_entropy.item(), step_i)
    writer.add_scalar('critic/mean_value', mean_value.item(), step_i)

    eval_r = eval(params, env)
    writer.add_scalar('rollout/eval_reward', eval_r, step_i)

    if eval_r > max_reward: 
        max_reward = eval_r
        model_save_path = str(model_path/f'params_{max_reward:.2f}')
        print(f'saving model to... {model_save_path}')
        with open(model_save_path, 'wb') as f: 
            cloudpickle.dump(params, f) 

#%%
#%%