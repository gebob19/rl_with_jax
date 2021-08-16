#%%
import jax 
import jax.numpy as np 
import numpy as onp 
import distrax 
import optax
import gym 
from functools import partial
import cloudpickle
import pybullet_envs

from jax.config import config
config.update("jax_enable_x64", True) 
config.update("jax_debug_nans", True) # break on nans

# env_name = 'Pendulum-v0'
# make_env = lambda: gym.make(env_name)

from env import Navigation2DEnv, Navigation2DEnv_Disc
env_name = 'Navigation2D'
def make_env(init_task=False):
    env = Navigation2DEnv_Disc(max_n_steps=200)
    
    if init_task: 
        env.seed(0)
        task = env.sample_tasks(1)[0]
        print(f'[LOGGER]: task = {task}')
        env.reset_task(task)

        # log max reward 
        goal = env._task['goal']
        reward = 0 
        step_count = 0 
        obs = env.reset()
        while True: 
            a = goal - obs 
            obs2, r, done, _ = env.step(a)
            reward += r
            step_count += 1 
            if done: break 
            obs = obs2
        print(f'[LOGGER]: MAX_REWARD={reward} IN {step_count} STEPS')
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
init_final = hk.initializers.RandomUniform(-3e-3, 3e-3)

# def _policy_fcn(s):
#     log_std = hk.get_parameter("log_std", shape=[n_actions,], init=np.ones)
#     mu = hk.Sequential([
#         hk.Linear(64), jax.nn.relu,
#         hk.Linear(64), jax.nn.relu,
#         hk.Linear(n_actions, w_init=init_final), np.tanh 
#     ])(s) * a_high
#     sig = np.exp(log_std)
#     return mu, sig

def _policy_fcn(obs):
    a_probs = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
        hk.Linear(n_actions, w_init=init_final), jax.nn.softmax
    ])(obs)
    return a_probs 

policy_fcn = hk.transform(_policy_fcn)
policy_fcn = hk.without_apply_rng(policy_fcn)
p_frwd = jax.jit(policy_fcn.apply)

# %%
# @jax.jit 
# def policy(params, obs, rng):
#     mu, sig = p_frwd(params, obs)
#     dist = distrax.MultivariateNormalDiag(mu, sig)
#     a = dist.sample(seed=rng)
#     a = np.clip(a, a_low, a_high)
#     log_prob = dist.log_prob(a)
#     return a, log_prob

# @jax.jit 
# def eval_policy(params, obs, _):
#     a, _ = p_frwd(params, obs)
#     a = np.clip(a, a_low, a_high)
#     return a, None

@jax.jit
def policy(p_params, obs, rng):
    a_probs = p_frwd(p_params, obs)
    dist = distrax.Categorical(probs=a_probs)
    a = dist.sample(seed=rng)
    log_prob = dist.log_prob(a)
    return a, log_prob

def eval(params, env, rng):
    rewards = 0 
    obs = env.reset()
    while True: 
        rng, subrng = jax.random.split(rng)
        # a = eval_policy(params, obs, subrng)[0]
        a = policy(params, obs, subrng)[0]

        a = onp.array(a)
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
    rollout = jax.tree_map(lambda x: x[idxs], rollout)
    return rollout

def rollout2batches(rollout, batch_size, n_batches=None):
    rollout_len = rollout[0].shape[0]
    n_chunks = rollout_len // batch_size
    # shuffle / de-correlate
    rollout = shuffle_rollout(rollout)
    if n_chunks == 0: return rollout
    # batch 
    batched_rollout = jax.tree_map(lambda x: np.array_split(x, n_chunks), rollout)
    n_chunks = n_chunks if n_batches is not None else min(n_chunks, n_batches)
    for i in range(n_chunks):
        batch = [d[i] for d in batched_rollout]
        yield batch 

def discount_cumsum(l, discount):
    l = onp.array(l)
    for i in range(len(l) - 1)[::-1]:
        l[i] = l[i] + discount * l[i+1]
    return l 

def reinforce_loss(p_params, sample):
    (obs, a, _, advantages) = sample
    # mu, sig = p_frwd(p_params, obs)
    # log_prob = distrax.MultivariateNormalDiag(mu, sig).log_prob(a)
    a_probs = p_frwd(p_params, obs)
    log_prob = distrax.Categorical(probs=a_probs).log_prob(a.astype(int))
    loss = -(log_prob * advantages).sum()
    return loss 

from functools import partial
def batch_reinforce_loss(params, batch):
    return jax.vmap(partial(reinforce_loss, params))(batch).sum()

reinforce_loss_grad = jax.jit(jax.value_and_grad(batch_reinforce_loss))

def compute_advantage_targets(rollout):
    (_, _, r, _, _, _) = rollout
    
    # reward2go    
    adv = discount_cumsum(r, discount=gamma)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    
    return adv

@jax.jit
def ppo_loss(p_params, sample):
    (obs, a, old_log_prob, advantages) = sample 

    # ## policy losses 
    # mu, sig = p_frwd(p_params, obs)
    # dist = distrax.MultivariateNormalDiag(mu, sig)

    a_probs = p_frwd(p_params, obs)
    dist = distrax.Categorical(probs=a_probs)

    # entropy 
    entropy_loss = -dist.entropy()
    # policy gradient 
    log_prob = dist.log_prob(a.astype(int))

    approx_kl = (old_log_prob - log_prob).sum()
    ratio = np.exp(log_prob - old_log_prob)
    p_loss1 = ratio * advantages
    p_loss2 = np.clip(ratio, 1-eps, 1+eps) * advantages
    policy_loss = -np.fmin(p_loss1, p_loss2).sum()

    clipped_mask = ((ratio > 1+eps) | (ratio < 1-eps)).astype(np.float32)
    clip_frac = clipped_mask.mean()

    loss = policy_loss

    info = dict(ploss=policy_loss, entr=-entropy_loss,
        approx_kl=approx_kl, cf=clip_frac)

    return loss, info

@jax.jit
def ppo_loss_batch(p_params, batch):
    out = jax.vmap(partial(ppo_loss, p_params))(batch)
    loss, info = jax.tree_map(lambda x: x.mean(), out)
    return loss, info

def optim_update_fcn(optim):
    @jax.jit
    def update_step(params, grads, opt_state):
        grads, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(params, grads)
        return params, opt_state
    return update_step

#%%
seed = onp.random.randint(1e5) # 90897 works very well 
epochs = 500
eval_every = 1
n_step_rollout = 100 # env._max_episode_steps
## PPO
policy_lr = 1e-3
gamma = 0.99 
lmbda = 0.95
eps = 0.2
## MAML 
task_batch_size = 40
fast_batch_size = 20 # 20 traj
eval_fast_batch_size = 40
alpha = 0.1

save_models = False

rng = jax.random.PRNGKey(seed)
onp.random.seed(seed)

obs = env.reset() # dummy input 
p_params = policy_fcn.init(rng, obs) 

## optimizers 
optimizer = lambda lr: optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.scale_by_adam(),
    optax.scale(-lr),
)
p_optim = optimizer(policy_lr)
p_opt_state = p_optim.init(p_params)
p_update_step = optim_update_fcn(p_optim)

import pathlib 
model_path = pathlib.Path(f'./models/ppo/{env_name}')
model_path.mkdir(exist_ok=True, parents=True)

#%%
##### MAML FUNCTIONS 
buffer = Vector_ReplayBuffer(n_step_rollout)

def rollout(p_params, env, rng):
    buffer.clear()
    obs = env.reset()

    for _ in range(n_step_rollout): # rollout 
        rng, subrng = jax.random.split(rng)
        a, log_prob = policy(p_params, obs, subrng)
        
        a = jax.lax.stop_gradient(a)
        log_prob = jax.lax.stop_gradient(log_prob)
        a = onp.array(a)
        
        obs2, r, done, _ = env.step(a)

        buffer.push((obs, a, r, obs2, done, log_prob))
        obs = obs2
        if done: break 

    # update rollout contents 
    traj = buffer.contents()
    advantages = compute_advantage_targets(traj)
    (obs, a, _, _, _, log_prob) = traj
    traj = (obs, a, log_prob, advantages)
    return traj

@jax.jit
def sgd_step(params, grads, alpha):
    sgd_update = lambda param, grad: param - alpha * grad
    return jax.tree_multimap(sgd_update, params, grads)

def maml_inner(p_params, env, rng, fast_batch_size, alpha):
    gradients = []
    for _ in range(fast_batch_size):
        rng, subkey = jax.random.split(rng, 2) 
        traj = rollout(p_params, env, subkey) 
        _, grad = reinforce_loss_grad(p_params, traj)
        gradients.append(grad)

    grad = jax.tree_multimap(lambda *x: np.stack(x).sum(0), *gradients)
    inner_params_p = sgd_step(p_params, grad, alpha)
    return inner_params_p

def maml_loss(p_params, env, rng):
    # maml inner
    rng1, rng2 = jax.random.split(rng, 2) 
    inner_params_p = maml_inner(p_params, env, rng1, fast_batch_size, alpha)

    # maml outer 
    traj = rollout(inner_params_p, env, rng2) 
    loss, _ = ppo_loss_batch(inner_params_p, traj)
    return loss 

def maml_eval(p_params, env, rng, n_steps=1):
    rewards = []

    # 0-step 
    rng, subkey = jax.random.split(rng, 2)
    reward_0step = eval(p_params, env, subkey)
    rewards.append(reward_0step)

    eval_alpha = alpha
    for _ in range(n_steps):
        rng, *subkeys = jax.random.split(rng, 3)
        # update
        p_inner_params = maml_inner(p_params, env, subkeys[0], 
            eval_fast_batch_size, eval_alpha)
        # eval 
        reward_nstep = eval(p_inner_params, env, subkeys[1])
        
        rewards.append(reward_nstep)
        p_params = p_inner_params
        eval_alpha = alpha / 2 
    
    return rewards

#%%
env.seed(seed)
n_tasks = 2
task = env.sample_tasks(1)[0] 
assert n_tasks in [1, 2] 
if n_tasks == 1: 
    tasks = [task] * task_batch_size
elif n_tasks == 2: 
    task2 = {'goal': -task['goal'].copy()}
    tasks = [task, task2] * (task_batch_size//2)

for task in tasks[:n_tasks]: 
    env.reset_task(task)
    # log max reward 
    goal = env._task['goal']
    reward = 0 
    step_count = 0 
    obs = env.reset()
    while True: 
        a = goal - obs 
        obs2, r, done, _ = env.step(a)
        reward += r
        step_count += 1 
        if done: break 
        obs = obs2
    print(f'[LOGGER]: MAX_REWARD={reward} IN {step_count} STEPS')

print(f'[LOGGER]: n_tasks_per_step = {len(tasks)}')

#%%
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment=f'MAMLppo_noValue_{env_name}_seed={seed}')

from tqdm import tqdm 
for e in tqdm(range(1, epochs+1)):
    # training
    gradients = []
    mean_loss = 0
    for task_i, task in enumerate(tqdm(tasks)): 
        env.reset_task(task)
        rng, subkey = jax.random.split(rng, 2)
        loss, grads = jax.value_and_grad(maml_loss)(p_params, env, subkey)
        
        gradients.append(grads)
        mean_loss += loss 

    mean_loss /= len(tasks)
    writer.add_scalar(f'loss/mean_task_loss', mean_loss.item(), e)

    # update
    gradients = jax.tree_multimap(lambda *x: np.stack(x).mean(0), *gradients)
    p_params, p_opt_state = p_update_step(p_params, grads, p_opt_state)

    # eval 
    if e % eval_every == 0:
        eval_tasks = tasks[:n_tasks] # 1 eval per task 
        task_rewards = []
        for task_i, eval_task in enumerate(eval_tasks):
            env.reset_task(eval_task)

            rng, subkey = jax.random.split(rng, 2)
            rewards = maml_eval(p_params, env, subkey, n_steps=3)
            task_rewards.append(rewards)
            
            for step_i, r in enumerate(rewards):
                writer.add_scalar(f'task{task_i}/reward_{step_i}step', r, e)

        mean_rewards=[]
        for step_i in range(len(task_rewards[0])):
            mean_r = sum([task_rewards[j][step_i] for j in range(len(eval_tasks))]) / len(eval_tasks)
            writer.add_scalar(f'mean_task/reward_{step_i}step', mean_r, e)
            mean_rewards.append(mean_r)
    

# %%
# %%
# %%
