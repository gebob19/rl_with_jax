#%%
import jax 
import jax.numpy as np 
import numpy as onp 
import distrax 
import optax
import gym 
from functools import partial
import cloudpickle

env_name = 'Pendulum-v0'
make_env = lambda: gym.make(env_name)

# from env import Navigation2DEnv
# env_name = 'Navigation2D'
# def make_env():
#     env = Navigation2DEnv()
#     env.seed(0)
#     task = env.sample_tasks(1)[0]
#     print(f'LOGGER: task = {task}')
#     env.reset_task(task)
#     return env 

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

# %%
@jax.jit 
def eval_policy(params, obs):
    a, _ = p_frwd(params, obs)
    a = np.clip(a, a_low, a_high)
    return a

@jax.jit 
def policy(params, obs, rng):
    mu, sig = p_frwd(params, obs)
    rng, subrng = jax.random.split(rng)
    dist = distrax.MultivariateNormalDiag(mu, sig)
    a = dist.sample(seed=subrng)
    a = np.clip(a, a_low, a_high)
    log_prob = dist.log_prob(a)
    return a, log_prob

def eval(params, env, rng):
    rewards = 0 
    obs = env.reset()
    while True: 
        # a = eval_policy(params, obs)
        rng, subkey = jax.random.split(rng)
        a = policy(params, obs, subkey)[0]

        obs2, r, done, _ = env.step(a)        
        obs = obs2 
        rewards += r
        if done: break 
    return rewards
    
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
        self.buffer[self.i] = onp.array([*obs, *onp.array(a), onp.array(r), *obs2, float(done), onp.array(log_prob)])
        self.i += 1 

    def contents(self):
        return onp.split(self.buffer[:self.i], self.splits, axis=-1)

    def clear(self):
        self.i = 0 
        self.buffer = onp.zeros((self.buffer_capacity, 2 * obs_dim + n_actions + 2 + 1))


def shuffle_rollout(rollout):
    rollout_len = rollout[0].shape[0]
    idxs = onp.arange(rollout_len) 
    onp.random.shuffle(idxs)
    rollout = jax.tree_map(lambda x: x[idxs], rollout, is_leaf=lambda x: hasattr(x, 'shape'))
    return rollout

def rollout2batches(rollout, batch_size):
    rollout_len = rollout[0].shape[0]
    n_chunks = rollout_len // batch_size
    # shuffle / de-correlate
    rollout = shuffle_rollout(rollout)
    if n_chunks == 0: return rollout
    # batch 
    batched_rollout = jax.tree_map(lambda x: np.array_split(x, n_chunks), rollout, is_leaf=lambda x: hasattr(x, 'shape'))
    for i in range(n_chunks):
        batch = [d[i] for d in batched_rollout]
        yield batch 

@jax.jit
def compute_advantages(v_params, rollout):
    (obs, _, r, obs2, done, _) = rollout
    r = (r - r.mean()) / (r.std() + 1e-10) # normalize
    
    batch_v_fcn = jax.vmap(partial(v_frwd, v_params))
    v_obs = batch_v_fcn(obs)
    v_obs2 = batch_v_fcn(obs2)

    v_target = r + (1 - done) * gamma * v_obs2
    advantages = v_target - v_obs
    advantages = jax.lax.stop_gradient(advantages)
    v_target = jax.lax.stop_gradient(v_target)
    
    # normalize 
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
    return advantages, v_target

@jax.jit
def ppo_loss(p_params, v_params, batch):
    (obs, a, old_log_prob, v_target, advantages) = batch 

    ## critic loss
    batch_v_fcn = jax.vmap(partial(v_frwd, v_params))
    v_obs = batch_v_fcn(obs)
    critic_loss = 0.5 * ((v_obs - v_target) ** 2)

    ## policy losses 
    batch_policy = jax.vmap(partial(p_frwd, p_params))
    mu, sig = batch_policy(obs)
    dist = distrax.MultivariateNormalDiag(mu, sig)
    # entropy 
    entropy_loss = -dist.entropy()[:, None]
    # policy gradient 
    log_probs = dist.log_prob(a)[:, None]
    ratio = np.exp(log_probs - old_log_prob)
    p_loss1 = ratio * advantages
    p_loss2 = np.clip(ratio, 1-eps, 1+eps) * advantages
    policy_loss = -np.fmin(p_loss1, p_loss2)

    # loss = policy_loss + 0.001 * entropy_loss + critic_loss
    loss = policy_loss + critic_loss
    loss = loss.mean()

    return loss 

def update_step(params, grads, optim, opt_state):
    grads, opt_state = optim.update(grads, opt_state)
    params = optax.apply_updates(params, grads)
    return params, opt_state

ppo_loss_grad = jax.jit(jax.value_and_grad(ppo_loss, argnums=[0,1]))

@jax.jit
def ppo_step(p_params, v_params, p_opt_state, v_opt_state, batch):
    loss, (p_grads, v_grads) = ppo_loss_grad(p_params, v_params, batch)
    p_params, p_opt_state = update_step(p_params, p_grads, p_optim, p_opt_state)
    v_params, v_opt_state = update_step(v_params, v_grads, v_optim, v_opt_state)
    return loss, p_params, v_params, p_opt_state, v_opt_state

class Worker:
    def __init__(self, n_steps):
        self.n_steps = n_steps
        self.buffer = Vector_ReplayBuffer(1e6)
        # import pybullet_envs
        self.env = make_env()
        self.obs = self.env.reset()

    def rollout(self, p_params, v_params, rng):
        self.buffer.clear()
        
        for _ in range(self.n_steps): # rollout 
            rng, subrng = jax.random.split(rng)
            a, log_prob = policy(p_params, self.obs, subrng)
            
            obs2, r, done, _ = self.env.step(a)

            self.buffer.push((self.obs, a, r, obs2, done, log_prob))
            self.obs = obs2
            if done: 
                self.obs = self.env.reset()

        # update rollout contents 
        rollout = self.buffer.contents()
        advantages, v_target = compute_advantages(v_params, rollout)
        (obs, a, r, _, _, log_prob) = rollout
        log_prob = jax.lax.stop_gradient(log_prob)
        rollout = (obs, a, log_prob, v_target, advantages)
        
        return rollout

#%%
seed = onp.random.randint(1e5)
gamma = 0.99 
eps = 0.2
batch_size = 32 
policy_lr = 1e-3
v_lr = 1e-3
max_n_steps = 1e6
n_step_rollout = 100 * 3 #env._max_episode_steps

rng = jax.random.PRNGKey(seed)
onp.random.seed(seed)

obs = env.reset() # dummy input 
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

import pathlib 
model_path = pathlib.Path(f'./models/ppo/{env_name}')
model_path.mkdir(exist_ok=True, parents=True)

#%%
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment=f'ppo_{env_name}_seed={seed}')

epi_i = 0 
step_i = 0 
from tqdm import tqdm 
pbar = tqdm(total=max_n_steps)
while step_i < max_n_steps: 
    rng, subkey = jax.random.split(rng) 
    rollout = worker.rollout(p_params, v_params, subkey)

    for batch in rollout2batches(rollout, batch_size):
        loss, p_params, v_params, p_opt_state, v_opt_state = \
            ppo_step(p_params, v_params, p_opt_state, v_opt_state, batch)
        step_i += 1 
        pbar.update(1)
        writer.add_scalar('loss/loss', loss.item(), step_i)

    writer.add_scalar('policy/log_std', p_params['~']['log_std'].mean().item(), step_i)
    
    rng, subkey = jax.random.split(rng) 
    reward = eval(p_params, env, rng)
    writer.add_scalar('eval/total_reward', reward.item(), step_i)

    if epi_i == 0 or reward > max_reward: 
        max_reward = reward
        save_path = str(model_path/f'params_{max_reward:.2f}')
        print(f'Saving model {save_path}...')
        with open(save_path, 'wb') as f: 
            cloudpickle.dump((p_params, v_params), f)
    
    epi_i += 1

# # %%
# with open('../models/ppo/Navigation2D/params_-40.54', 'rb') as f: 
#     (p_params, v_params) = cloudpickle.load(f)

# # %%
# import matplotlib.pyplot as plt 
# def render(p_params, env, rng, n_steps):
#     env.seed(0)
#     obs = env.reset()

#     plt.scatter(*env._task['goal'], marker='*')
#     plt.scatter(*env._state, color='r')
#     xp, yp = obs
#     rewards = []
#     actions = []
#     for _ in range(n_steps):
#         rng, subkey = jax.random.split(rng, 2)
#         a, _ = policy(p_params, obs, subkey)

#         obs2, r, done, _ = env.step(a)
#         x, y = obs2
#         rewards.append(r)
#         actions.append(a)

#         if done: break 
#         plt.plot([xp, x], [yp, y], color='red')
#         xp, yp = obs2
#         obs = obs2

#     plt.show()
#     return sum(rewards), actions

# # %%
# rng, subrng = jax.random.split(rng)
# r, actions = render(p_params, env, subrng, 100)
# print(r)

# # %%
# # %%
# # %%
