# very slow (even on TPU :( )
import os 
from IPython.display import clear_output

!pip install distrax optax dm-haiku
clear_output()

try:
  import brax
except ImportError:
  !pip install git+https://github.com/google/brax.git@main
  clear_output()
  import brax

if 'COLAB_TPU_ADDR' in os.environ:
  from jax.tools import colab_tpu
  colab_tpu.setup_tpu()

#%%
import jax 
import jax.numpy as np 
import numpy as onp 
import distrax 
import optax
import gym 
from functools import partial
import cloudpickle

from jax.config import config
# config.update("jax_enable_x64", True) 
config.update("jax_debug_nans", True) # break on nans

import brax
from brax.envs import _envs, create_gym_env

for env_name, env_class in _envs.items():
    env_id = f"brax_{env_name}-v0"
    entry_point = partial(create_gym_env, env_name=env_name)
    if env_id not in gym.envs.registry.env_specs:
        print(f"Registring brax's '{env_name}' env under id '{env_id}'.")
        gym.register(env_id, entry_point=entry_point)

make_env = lambda n_envs: gym.make("brax_halfcheetah-v0", batch_size=n_envs, \
    episode_length=1000)
env = make_env(1) # tmp

obs_dim = env.observation_space.shape[-1]
n_actions = env.action_space.shape[-1]

a_low = -1 
a_high = 1 
assert (env.action_space.low == -1).all() and (env.action_space.high == 1).all()
print(f'[LOGGER] n_actions: {n_actions} obs_dim: {obs_dim}')

#%%
import haiku as hk
init_final = hk.initializers.RandomUniform(-3e-3, 3e-3)

def _policy_fcn(s):
    log_std = hk.get_parameter("log_std", shape=[n_actions,], init=np.ones)
    mu = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
        hk.Linear(n_actions, w_init=init_final), np.tanh 
    ])(s) 
    sig = np.exp(log_std)
    return mu, sig

def _critic_fcn(s):
    v = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
        hk.Linear(1), 
    ])(s)
    return v 

policy_fcn = hk.transform(_policy_fcn)
policy_fcn = hk.without_apply_rng(policy_fcn)
p_frwd = jax.jit(policy_fcn.apply)

critic_fcn = hk.transform(_critic_fcn)
critic_fcn = hk.without_apply_rng(critic_fcn)
v_frwd = jax.jit(critic_fcn.apply)

@jax.jit 
def policy(params, obs, rng):
    mu, sig = p_frwd(params, obs)
    dist = distrax.MultivariateNormalDiag(mu, sig)
    a = dist.sample(seed=rng)
    a = np.clip(a, a_low, a_high)
    log_prob = dist.log_prob(a)
    return a, log_prob

def pmap_policy(params, obs, rng):
    keys = jax.random.split(rng, n_envs)
    a, log_prob = jax.pmap(policy, in_axes=(None, 0, 0))(params, obs, keys)
    return a, log_prob

def eval(params, env, rng):
    rewards = 0 
    obs = env.reset()
    while True: 
        rng, subrng = jax.random.split(rng)
        a = pmap_policy(params, obs, subrng)[0]
        obs2, r, done, _ = env.step(a)        
        obs = obs2 
        rewards += r
        if done.any(): break 
    return rewards.sum()
    
class Batch_Vector_ReplayBuffer:
    def __init__(self, buffer_capacity, n_envs):
        self.buffer_capacity = buffer_capacity = int(buffer_capacity)
        self.i = 0
        # obs, obs2, a, r, done
        self.splits = [obs_dim, obs_dim+n_actions, obs_dim+n_actions+1, obs_dim*2+1+n_actions, obs_dim*2+1+n_actions+1]
        self.n_envs = n_envs
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
        self.buffer = onp.zeros((self.n_envs, self.buffer_capacity, 2 * obs_dim + n_actions + 2 + 1))

def shuffle_rollout(rollout):
    rollout_len = rollout[0].shape[0]
    idxs = onp.arange(rollout_len) 
    onp.random.shuffle(idxs)
    rollout = jax.tree_map(lambda x: x[idxs], rollout)
    return rollout

def rollout2batches(rollout, batch_size):
    rollout_len = rollout[0].shape[0]
    n_chunks = rollout_len // batch_size
    # shuffle / de-correlate
    rollout = shuffle_rollout(rollout)
    if n_chunks == 0: return rollout
    # batch 
    batched_rollout = jax.tree_map(lambda x: np.array_split(x, n_chunks), rollout)
    for i in range(n_chunks):
        batch = [d[i] for d in batched_rollout]
        yield batch 

from jax.ops import index, index_add
def discount_cumsum(l, discount):
    for i in range(len(l) - 1)[::-1]:
        l = index_add(l, index[i], discount * l[i+1])
    return l 

def compute_advantage_targets(v_params, rollout):
    (obs, _, r, obs2, done, _) = rollout
    
    batch_v_fcn = jax.vmap(partial(v_frwd, v_params))
    v_obs = batch_v_fcn(obs)
    v_obs2 = batch_v_fcn(obs2)

    # gae
    deltas = (r + (1 - done) * gamma * v_obs2) - v_obs
    deltas = jax.lax.stop_gradient(deltas)
    adv = discount_cumsum(deltas, discount=gamma * lmbda)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # reward2go
    v_target = discount_cumsum(r, discount=gamma)
    
    return adv, v_target

@jax.jit
def ppo_loss(p_params, v_params, sample):
    (obs, a, old_log_prob, v_target, advantages) = sample 

    ## critic loss
    v_obs = v_frwd(v_params, obs)
    critic_loss = (0.5 * ((v_obs - v_target) ** 2)).sum()

    ## policy losses 
    mu, sig = p_frwd(p_params, obs)
    dist = distrax.MultivariateNormalDiag(mu, sig)
    # entropy 
    entropy_loss = -dist.entropy()
    # policy gradient 
    log_prob = dist.log_prob(a)

    approx_kl = (old_log_prob - log_prob).sum()
    ratio = np.exp(log_prob - old_log_prob)
    p_loss1 = ratio * advantages
    p_loss2 = np.clip(ratio, 1-eps, 1+eps) * advantages
    policy_loss = -np.fmin(p_loss1, p_loss2).sum()

    clipped_mask = ((ratio > 1+eps) | (ratio < 1-eps)).astype(np.float32)
    clip_frac = clipped_mask.mean()

    loss = policy_loss + 0.001 * entropy_loss + critic_loss

    info = dict(ploss=policy_loss, entr=-entropy_loss, vloss=critic_loss, 
        approx_kl=approx_kl, cf=clip_frac)

    return loss, info

def ppo_loss_batch(p_params, v_params, batch):
    out = jax.pmap(partial(ppo_loss, p_params, v_params))(batch)
    loss, info = jax.tree_map(lambda x: x.mean(), out)
    return loss, info

ppo_loss_grad = jax.jit(jax.value_and_grad(ppo_loss_batch, argnums=[0,1], has_aux=True))

def optim_update_fcn(optim):
    @jax.jit
    def update_step(params, grads, opt_state):
        grads, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(params, grads)
        return params, opt_state
    return update_step

@jax.jit
def ppo_step(p_params, v_params, p_opt_state, v_opt_state, batch):
    (loss, info), (p_grads, v_grads) = ppo_loss_grad(p_params, v_params, batch)
    p_params, p_opt_state = p_update_step(p_params, p_grads, p_opt_state)
    v_params, v_opt_state = v_update_step(v_params, v_grads, v_opt_state)
    return loss, info, p_params, v_params, p_opt_state, v_opt_state

class Worker:
    def __init__(self, n_steps):
        self.n_steps = n_steps
        self.buffer = Batch_Vector_ReplayBuffer(n_step_rollout, n_envs)
        self.env = make_env(n_envs)
        # self.env_reset = jax.jit(self.env.reset)
        # self.env_step = jax.jit(self.env.step)

    def rollout(self, p_params, v_params, rng):
        self.buffer.clear()
        obs = self.env.reset()
        
        for _ in range(n_step_rollout):
            a, log_prob = pmap_policy(p_params, obs, rng)
            
            obs2, r, done, _ = self.env.step(a)
            self.buffer.push((obs, a, r, obs2, done, log_prob))

            obs = obs2 
            if done.all(): break # .any() == .all() in this case

        rollout = self.buffer.contents()

        advantages, v_target = jax.pmap(compute_advantage_targets, in_axes=(None, 0))(v_params, rollout)
        (obs, a, r, _, _, log_prob) = rollout
        rollout = (obs, a, log_prob, v_target, advantages)
        # (n_envs, length, _) -> (n_envs * length, _)
        rollout = jax.tree_map(lambda x: x.reshape(-1, x.shape[-1]), rollout)

        return rollout

#%%
seed = 90897 #onp.random.randint(1e5) # 90897 works very well 
gamma = 0.99 
lmbda = 0.95
eps = 0.2
batch_size = 128
policy_lr = 1e-3
v_lr = 1e-3
max_n_steps = 1e6
n_step_rollout = 1000 # env._max_episode_steps
save_models = False
n_envs = 8

rng = jax.random.PRNGKey(seed)
onp.random.seed(seed)

obs = np.array(onp.random.randn(obs_dim)) # dummy input 
p_params = policy_fcn.init(rng, obs) 
v_params = critic_fcn.init(rng, obs) 

worker = Worker(n_step_rollout)

## optimizers 
optimizer = lambda lr: optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.scale_by_adam(),
    optax.scale(-lr),
)
p_optim = optimizer(policy_lr)
v_optim = optimizer(v_lr)

p_opt_state = p_optim.init(p_params)
v_opt_state = v_optim.init(v_params)

p_update_step = optim_update_fcn(p_optim)
v_update_step = optim_update_fcn(v_optim)

eval_env = make_env(1)

import pathlib 
model_path = pathlib.Path(f'./models/ppo/{env_name}')
model_path.mkdir(exist_ok=True, parents=True)

#%%
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment=f'ppo_brax_{env_name}_seed={seed}')

epi_i = 0 
step_i = 0 
from tqdm.notebook import tqdm 
pbar = tqdm(total=max_n_steps)
while step_i < max_n_steps: 
    rng, subkey = jax.random.split(rng, 2) 
    rollout = worker.rollout(p_params, v_params, subkey) 

    for batch in rollout2batches(rollout, batch_size):
        loss, info, p_params, v_params, p_opt_state, v_opt_state = \
            ppo_step(p_params, v_params, p_opt_state, v_opt_state, batch)
        step_i += 1 
        pbar.update(1)
        writer.add_scalar('loss/loss', loss.item(), step_i)
        for k in info.keys(): 
            writer.add_scalar(f'info/{k}', info[k].item(), step_i)

    rng, subrng = jax.random.split(rng)
    reward = eval(p_params, eval_env, subrng) 
    writer.add_scalar('eval/total_reward', reward.item(), step_i)

    if save_models and (epi_i == 0 or reward > max_reward): 
        max_reward = reward
        save_path = str(model_path/f'params_{max_reward:.2f}')
        print(f'Saving model {save_path}...')
        with open(save_path, 'wb') as f: 
            cloudpickle.dump((p_params, v_params), f)

    epi_i += 1