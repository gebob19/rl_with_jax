#%%
%load_ext autoreload
%autoreload 2

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

from jax.config import config
config.update("jax_enable_x64", True) 
config.update("jax_debug_nans", True) # break on nans

#%%
from utils import disc_policy as policy 
from utils import eval, init_policy_fcn, Disc_Vector_Buffer, discount_cumsum, \
    tree_mean, mean_vmap_jit, sum_vmap_jit

env_name = 'Navigation2D'
env = Navigation2DEnv_Disc(max_n_steps=200) # maml debug env 

n_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]
print(f'[LOGGER] n_actions: {n_actions} obs_dim: {obs_dim}')

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
    buffer = Disc_Vector_Buffer(obs_dim, max_n_steps)
    obs = env.reset()
    for _ in range(max_n_steps): 
        rng, subkey = jax.random.split(rng, 2)
        a, log_prob = policy(p_frwd, p_params, obs, subkey, False)

        a = jax.lax.stop_gradient(a)
        log_prob = jax.lax.stop_gradient(log_prob)
        a = a.item()

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
    pi = p_frwd(p_params, obs)
    log_prob = distrax.Categorical(probs=pi).log_prob(a)
    loss = -(log_prob * adv).sum()
    return loss

# def reinforce_loss(p_params, obs, a, adv):
#     loss = jax.vmap(partial(_reinforce_loss, p_params))(obs, a, adv).sum()
#     return loss

reinforce_loss = sum_vmap_jit(_reinforce_loss, (None, 0, 0, 0))
reinforce_loss_grad = jax.jit(jax.value_and_grad(reinforce_loss))

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

# %%
seed = onp.random.randint(1e5)
epochs = 500
eval_every = 1
max_n_steps = 100 # env._max_episode_steps
## TRPO
delta = 0.01
n_search_iters = 10 
cg_iters = 10
gamma = 0.99 
lmbda = 0.95
## MAML 
task_batch_size = 40
train_n_traj = 20
eval_n_traj = 40
alpha = 0.1
damp_lambda = 0.01

rng = jax.random.PRNGKey(seed)
onp.random.seed(seed)

## model init
p_frwd, p_params = init_policy_fcn('discrete', env, rng)

## save path 
model_path = pathlib.Path(f'./models/maml/{env_name}')
model_path.mkdir(exist_ok=True, parents=True)

# %%
task = env.sample_tasks(1)[0]
env.reset_task(task)

# %%
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

def _trpo_policy_loss(p_params, obs, a, adv, old_log_prob):
    pi = p_frwd(p_params, obs)
    dist = distrax.Categorical(probs=pi)
    ratio = np.exp(dist.log_prob(a) - old_log_prob)
    loss = -(ratio * adv).sum()
    return loss

trpo_policy_loss = mean_vmap_jit(_trpo_policy_loss, (None, *([0]*4)))

def maml_outer(p_params, env, rng):
    subkeys = jax.random.split(rng, 3)
    newp, W = maml_inner(p_params, env, subkeys[0], train_n_traj, 0.1)

    traj = rollout(env, p_params, subkeys[1]) 
    adv = compute_advantage(W, traj)
    loss = trpo_policy_loss(newp, traj['obs'], traj['a'], adv, traj['log_prob'])
    return loss, traj

(loss, traj), grads = jax.value_and_grad(maml_outer, has_aux=True)(p_params, env, rng)
loss

# grads = grad(maml_outer)
# compute natural gradient 
# line search step 

# %%
def _natural_gradient(params, grads, obs):
    f = lambda w: p_frwd(w, obs)
    rho = D_KL_probs
    ngrad, _ = jax.scipy.sparse.linalg.cg(
            tree_mvp_dampen(lambda v: gnh_vp(f, rho, params, v), damp_lambda),
            grads, maxiter=cg_iters)

    vec = lambda x: x.flatten()[:, None]
    mat_mul = lambda x, y: np.sqrt(2 * delta / (vec(x).T @ vec(y)).flatten())
    alpha = jax.tree_multimap(mat_mul, grads, ngrad)
    return ngrad, alpha
natural_gradient = mean_vmap_jit(_natural_gradient, (None, 0))

ngrad, alpha = natural_gradient(p_params, grads, traj['obs'])
alpha

# %%
probs = p_frwd(p_params, traj['obs'][0])
probs

# %%
jax.hessian(D_KL_probs)(probs, np.array(onp.array([0.55, 0.1, 0.25, 0.1])))

# %%
# %%
# %%
#%%
### TRPO FCNS 
from utils import gnh_vp, tree_mvp_dampen

def D_KL_probs(p1, p2):
    d_kl = (p1 * (np.log(p1) - np.log(p2))).sum()
    return d_kl

def D_KL_probs_params(param1, param2, obs):
    p1, p2 = p_frwd(param1, obs), p_frwd(param2, obs)
    return D_KL_probs(p1, p2)

def sample(traj, p):
    traj_len = int(traj['obs'].shape[0] * p)
    idxs = onp.random.choice(traj_len, size=traj_len, replace=False)
    sampled_traj = jax.tree_map(lambda x: x[idxs], traj)
    return sampled_traj

import operator
tree_scalar_op = lambda op: lambda tree, arg2: jax.tree_map(lambda x: op(x, arg2), tree)
tree_scalar_divide = tree_scalar_op(operator.truediv)
tree_scalar_mult = tree_scalar_op(operator.mul)

# backtracking line-search 
def line_search(alpha_start, init_loss, p_params, p_ngrad, rollout, n_iters, delta):
    obs = rollout[0]
    for i in np.arange(n_iters):
        alpha = tree_scalar_divide(alpha_start, 2 ** i)

        new_p_params = sgd_step_tree(p_params, p_ngrad, alpha)
        new_loss = batch_policy_loss(new_p_params, rollout)

        d_kl = jax.vmap(partial(D_KL_probs_params, new_p_params, p_params))(obs).mean()

        if (new_loss < init_loss) and (d_kl <= delta): 
            writer.add_scalar('info/line_search_n_iters', i, e)
            return new_p_params # new weights 

    writer.add_scalar('info/line_search_n_iters', -1, e)
    return p_params # no new weights 

# %%


# %%
# %%
# %%
