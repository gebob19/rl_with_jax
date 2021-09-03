# #%%
# %load_ext autoreload
# %autoreload 2

import jax
import jax.numpy as np 
import numpy as onp 
import distrax 
import optax
import gym 
from functools import partial
from env import Navigation2DEnv, Navigation2DEnv_Disc
import cloudpickle
import pathlib 
import haiku as hk
from tqdm import tqdm 

from jax.config import config
config.update("jax_enable_x64", True) 
config.update("jax_debug_nans", True) # break on nans

#%%
from utils import gaussian_log_prob, gaussian_sample
from utils import cont_policy as policy 
from utils import eval, init_policy_fcn, Cont_Vector_Buffer, discount_cumsum, \
    tree_mean, mean_vmap_jit, sum_vmap_jit, optim_update_fcn

env_name = 'Navigation2D'
env = Navigation2DEnv(max_n_steps=200) # maml debug env 

n_actions = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]

a_high = env.action_space.high[0]
a_low = env.action_space.low[0]
clip_range = [a_low, a_high]
print(f'[LOGGER] n_actions: {n_actions} obs_dim: {obs_dim} action_clip_range: {clip_range}')

# value function / baseline  
# https://github.com/rll/rllab/blob/master/rllab/baselines/linear_feature_baseline.py
def v_features(obs):
    o = np.clip(obs, -10, 10)
    l = len(o)
    al = np.arange(l).reshape(-1, 1) / 100.0
    return np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)

def v_fit(trajectories, feature_fcn=v_features, reg_coeff=1e-5):
    featmat = np.concatenate([feature_fcn(traj['obs']) for traj in trajectories])
    r = np.concatenate([traj['r'] for traj in trajectories])
    for _ in range(5):
        # solve argmin_x (F x = R) <-- unsolvable (F non-sqr) 
        # == argmin_x (F^T F x = F^T R) <-- solvable (sqr F^T F)
        # where F = Features, x = Weights, R = rewards
        _coeffs = np.linalg.lstsq(
            featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
            featmat.T.dot(r)
        )[0]
        if not np.any(np.isnan(_coeffs)):
            return _coeffs, 0 # succ
        reg_coeff *= 10
    return np.zeros_like(_coeffs), 1 # err 

def sample_trajectory(traj, p):
    traj_len = int(traj[0].shape[0] * p)
    idxs = onp.random.choice(traj_len, size=traj_len, replace=False)
    sampled_traj = jax.tree_map(lambda x: x[idxs], traj)
    return sampled_traj

def rollout(env, p_params, rng):
    buffer = Cont_Vector_Buffer(n_actions, obs_dim, max_n_steps)
    obs = env.reset()
    for _ in range(max_n_steps): 
        rng, subkey = jax.random.split(rng, 2)
        a, log_prob = policy(p_frwd, p_params, obs, subkey, clip_range, False)

        a = jax.lax.stop_gradient(a)
        log_prob = jax.lax.stop_gradient(log_prob)
        a = onp.array(a)

        obs2, r, done, _ = env.step(a)
        buffer.push((obs, a, r, obs2, done, log_prob))
        
        obs = obs2
        if done: break 

    trajectory = buffer.contents()
    return trajectory 

#%%
# inner optim 
@jax.jit
def _reinforce_loss(p_params, obs, a, adv):
    mu, std = p_frwd(p_params, obs)
    log_prob = gaussian_log_prob(a, mu, std)
    loss = -(log_prob * adv).sum()
    return loss

reinforce_loss = sum_vmap_jit(_reinforce_loss, (None, 0, 0, 0))
reinforce_loss_grad = jax.jit(jax.value_and_grad(reinforce_loss))

@jax.jit
def _ppo_loss(p_params, obs, a, adv, old_log_prob):
    ## policy losses 
    mu, std = p_frwd(p_params, obs)

    # policy gradient 
    log_prob = gaussian_log_prob(a, mu, std)

    approx_kl = (old_log_prob - log_prob).sum()
    ratio = np.exp(log_prob - old_log_prob)
    p_loss1 = ratio * adv
    p_loss2 = np.clip(ratio, 1-eps, 1+eps) * adv
    policy_loss = -np.fmin(p_loss1, p_loss2).sum()

    clipped_mask = ((ratio > 1+eps) | (ratio < 1-eps)).astype(np.float32)
    clip_frac = clipped_mask.mean()

    loss = policy_loss 
    info = dict(ploss=policy_loss, approx_kl=approx_kl, cf=clip_frac)

    return loss, info

ppo_loss = mean_vmap_jit(_ppo_loss, (None, 0, 0, 0, 0))

@jax.jit
def sgd_step_int(params, grads, alpha):
    sgd_update = lambda param, grad: param - alpha * grad
    return jax.tree_multimap(sgd_update, params, grads)

@jax.jit
def sgd_step_tree(params, grads, alphas):
    sgd_update = lambda param, grad, alpha: param - alpha * grad
    return jax.tree_multimap(sgd_update, params, grads, alphas)

def sgd_step(params, grads, alpha):
    step_fcn = sgd_step_int if type(alpha) in [int, float] else sgd_step_tree
    return step_fcn(params, grads, alpha)

#%%
seed = onp.random.randint(1e5)
epochs = 500
eval_every = 1
max_n_steps = 100 # env._max_episode_steps
## PPO
eps = 0.2
gamma = 0.99 
lmbda = 0.95
lr = 1e-3
## MAML 
task_batch_size = 40
train_n_traj = 20
eval_n_traj = 40
alpha = 0.1
damp_lambda = 0.01

rng = jax.random.PRNGKey(seed)
p_frwd, p_params = init_policy_fcn('continuous', env, rng)

p_update_fcn, p_opt_state = optim_update_fcn(optax.adam(lr), p_params)

#%%
task = env.sample_tasks(1)[0]
env.reset_task(task)

#%%
@jax.jit
def compute_advantage(W, traj):
    # linear fcn predict
    v_obs = v_features(traj['obs']) @ W
    # baseline 
    adv = traj['r'] - v_obs
    # normalize 
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv.squeeze()

def maml_inner(p_params, env, rng, n_traj, alpha):
    subkeys = jax.random.split(rng, n_traj) 
    trajectories = []
    for i in range(n_traj):
        traj = rollout(env, p_params, subkeys[i]) 
        traj['r'] = discount_cumsum(traj['r'], discount=gamma)
        trajectories.append(traj)

    W = v_fit(trajectories)[0]
    for i in range(len(trajectories)): 
        trajectories[i]['adv'] = compute_advantage(W, trajectories[i])
    
    gradients = []
    for traj in trajectories:
        _, grad = reinforce_loss_grad(p_params, traj['obs'], traj['a'], traj['adv'])
        gradients.append(grad)
    grads = jax.tree_multimap(lambda *x: np.stack(x).sum(0), *gradients)
    inner_params_p = sgd_step(p_params, grads, alpha)

    return inner_params_p, W

def maml_outer(p_params, env, rng):
    subkeys = jax.random.split(rng, 3)
    inner_p_params, W = maml_inner(p_params, env, subkeys[0], train_n_traj, alpha)

    traj = rollout(env, inner_p_params, subkeys[1]) 
    traj['adv'] = compute_advantage(W, traj)
    loss, info = ppo_loss(inner_p_params, traj['obs'], traj['a'], traj['adv'], traj['log_prob'])
    return loss, (info, traj)

def maml_eval(env, p_params, rng, n_steps=1):
    rewards = []

    rng, subkey = jax.random.split(rng, 2)
    reward_0step = eval(p_frwd, policy, p_params, env, subkey, clip_range, True)
    rewards.append(reward_0step)

    eval_alpha = alpha
    for _ in range(n_steps):
        rng, *subkeys = jax.random.split(rng, 3)
        
        inner_p_params, _ = maml_inner(p_params, env, subkeys[0], train_n_traj, eval_alpha)
        r = eval(p_frwd, policy, inner_p_params, env, subkeys[1], clip_range, True)

        rewards.append(r)
        eval_alpha = alpha / 2 
        p_params = inner_p_params

    return rewards

#%%
env.seed(0)
n_tasks = 1 
task = env.sample_tasks(1)[0] ## only two tasks
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

#%%
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment=f'maml_{n_tasks}task_test_seed={seed}')

#%%
step_count = 0 
for e in tqdm(range(1, epochs+1)):
    # training 
    # tasks = env.sample_tasks(task_batch_size)

    gradients = []
    mean_loss = 0
    for task_i, task in enumerate(tqdm(tasks)): 
        env.reset_task(task)
        rng, subkey = jax.random.split(rng, 2)
        (loss, (info, traj)), grads = jax.value_and_grad(maml_outer, has_aux=True)(p_params, env, rng)
        
        gradients.append(grads)
        mean_loss += loss 
        step_count += 1 
            
    mean_loss /= len(tasks)
    writer.add_scalar(f'loss/mean_task_loss', mean_loss, step_count)

    # update 
    gradients = jax.tree_multimap(lambda *x: np.stack(x).mean(0), *gradients)
    p_params, p_opt_state = p_update_fcn(p_params, p_opt_state, gradients)

    # eval 
    if e % eval_every == 0:
        eval_tasks = tasks
        task_rewards = []
        for task_i, eval_task in enumerate(eval_tasks):
            env.reset_task(eval_task)

            rng, subkey = jax.random.split(rng, 2)
            rewards, infos = maml_eval(env, p_params, subkey, n_steps=3)
            task_rewards.append(rewards)

            for step_i, r in enumerate(rewards):
                writer.add_scalar(f'task{task_i}/reward_{step_i}step', r, e)

        mean_rewards=[]
        for step_i in range(len(task_rewards[0])):
            mean_r = sum([task_rewards[j][step_i] for j in range(len(task_rewards))]) / 2
            writer.add_scalar(f'mean_task/reward_{step_i}step', mean_r, e)
            mean_rewards.append(mean_r)
        

#%%
#%%