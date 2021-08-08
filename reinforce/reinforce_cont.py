#%%
import jax
import jax.numpy as np 
import numpy as onp 
import distrax 
import optax
import gym 
from functools import partial

from jax.config import config
config.update("jax_enable_x64", True) 
config.update("jax_debug_nans", True) # break on nans

#%%
# env_name = 'CartPole-v0'
# env = gym.make(env_name)

from env import Navigation2DEnv
env_name = 'Navigation2D'
def make_env():
    env = Navigation2DEnv()
    env.seed(0)
    task = env.sample_tasks(1)[0]
    print(f'LOGGER: task = {task}')
    env.reset_task(task)
    return env 
env = make_env()

n_actions = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]

a_high = env.action_space.high[0]
a_low = env.action_space.low[0]

print(f'[LOGGER] a_high: {a_high} a_low: {a_low} n_actions: {n_actions} obs_dim: {obs_dim}')
assert -a_high == a_low

#%%
import haiku as hk 

def _policy_fcn(s):
    log_std = hk.get_parameter("log_std", shape=[n_actions,], init=np.ones)
    mu = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
        hk.Linear(n_actions), np.tanh 
    ])(s) * a_high
    sig = np.exp(log_std)
    return mu, sig

policy_fcn = hk.transform(_policy_fcn)
policy_fcn = hk.without_apply_rng(policy_fcn)
p_frwd = jax.jit(policy_fcn.apply)

@jax.jit
def update_step(params, grads, opt_state):
    grads, opt_state = p_optim.update(grads, opt_state)
    params = optax.apply_updates(params, grads)
    return params, opt_state

def reward2go(r, gamma=0.99):
    for i in range(len(r) - 1)[::-1]:
        r[i] = r[i] + gamma * r[i+1]
    r = (r - r.mean()) / (r.std() + 1e-8)
    return r 

# @jax.jit    
# def policy(p_params, obs, rng):
#     a_probs = p_frwd(p_params, obs)
#     a = distrax.Categorical(probs=a_probs).sample(seed=rng)        
#     return a

@jax.jit 
def policy(params, obs, rng):
    mu, sig = p_frwd(params, obs)
    rng, subrng = jax.random.split(rng)
    dist = distrax.MultivariateNormalDiag(mu, sig)
    a = dist.sample(seed=subrng)
    a = np.clip(a, a_low, a_high)
    return a

def rollout(p_params, rng):
    global step_count
    
    observ, action, rew = [], [], []
    obs = env.reset()
    while True: 
        rng, subkey = jax.random.split(rng, 2) 
        a = policy(p_params, obs, subkey)

        obs2, r, done, _ = env.step(a)

        observ.append(obs)
        action.append(a)
        rew.append(r)

        if done: break
        obs = obs2

    obs = np.stack(observ)
    a = np.stack(action)
    r = onp.stack(rew) # 

    return obs, a, r 

def reinforce_loss(p_params, obs, a, r):
    mu, sig = p_frwd(p_params, obs)
    log_prob = distrax.MultivariateNormalDiag(mu, sig).log_prob(a)
    loss = -(log_prob * r).sum()
    return loss 

from functools import partial
def batch_reinforce_loss(params, batch):
    return jax.vmap(partial(reinforce_loss, params))(*batch).sum()

# %%
seed = onp.random.randint(1e5)
policy_lr = 1e-3
batch_size = 32
max_n_steps = 1000

rng = jax.random.PRNGKey(seed)
onp.random.seed(seed)

obs = env.reset() # dummy input 
p_params = policy_fcn.init(rng, obs) 

## optimizers 
schedule_fn = optax.polynomial_schedule(
    init_value=-policy_lr, end_value=-1e-8, power=1, 
    transition_steps=max_n_steps)

p_optim = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.scale_by_adam(),
    optax.scale_by_schedule(schedule_fn),
)
# p_optim = optax.sgd(policy_lr)

p_opt_state = p_optim.init(p_params)

# %%
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment=f'reinforce_cont_{env_name}_seed={seed}')

# %%
from tqdm import tqdm 

step_count = 0
epi_i = 0 

pbar = tqdm(total=max_n_steps)
gradients = []
loss_grad_fcn = jax.jit(jax.value_and_grad(batch_reinforce_loss))

env_step_count = 0
while step_count < max_n_steps: 
    rng, subkey = jax.random.split(rng, 2) 
    obs, a, r = rollout(p_params, subkey)
    writer.add_scalar('rollout/reward', r.sum().item(), epi_i)
    r = reward2go(r)
    
    loss, grad = loss_grad_fcn(p_params, (obs, a, r))
    gradients.append(grad)
    writer.add_scalar('loss/loss', loss.item(), epi_i)

    epi_i += 1
    if epi_i % batch_size == 0:
        grad = jax.tree_multimap(lambda *x: sum(x), *gradients)
        p_params, p_opt_state = update_step(p_params, grad, p_opt_state)
        step_count += 1
        pbar.update(1)
        gradients = []

# %%
# %%
# %%
