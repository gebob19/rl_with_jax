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

from jax.config import config
config.update("jax_enable_x64", True) 
config.update("jax_debug_nans", True) # break on nans

#%%
env_name = 'Navigation2D'
env = Navigation2DEnv(max_n_steps=200) # maml debug env 

n_actions = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]

a_high = env.action_space.high[0]
a_low = env.action_space.low[0]

print(f'[LOGGER] a_high: {a_high} a_low: {a_low} n_actions: {n_actions} obs_dim: {obs_dim}')
assert -a_high == a_low

import haiku as hk
def _policy_fcn(s):
    log_std = hk.get_parameter("log_std", shape=[n_actions,], init=np.ones, dtype=np.float64)
    mu = hk.Sequential([
        hk.Linear(100), jax.nn.relu,
        hk.Linear(100), jax.nn.relu,
        hk.Linear(n_actions), np.tanh 
    ])(s) * a_high
    sig = np.exp(log_std)
    return mu, sig

policy_fcn = hk.transform(_policy_fcn)
policy_fcn = hk.without_apply_rng(policy_fcn)
p_frwd = jax.jit(policy_fcn.apply)

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
        # solve argmin_x (F x = R) == argmin_x (F^T F x = F^T R) -- solvable
        _coeffs = np.linalg.lstsq(
            featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
            featmat.T.dot(r)
        )[0]
        if not np.any(np.isnan(_coeffs)):
            return _coeffs, 0 # succ
        reg_coeff *= 10
    return np.zeros_like(_coeffs), 1 # err 

@jax.jit 
def policy(params, obs, rng):
    mu, sig = p_frwd(params, obs)
    rng, subrng = jax.random.split(rng)
    dist = distrax.MultivariateNormalDiag(mu, sig)
    a = dist.sample(seed=subrng)
    a = np.clip(a, a_low, a_high)
    log_prob = dist.log_prob(a)
    return a, log_prob

@jax.jit 
def eval_policy(params, obs, _):
    a, _ = p_frwd(params, obs)
    a = np.clip(a, a_low, a_high)
    return a, None

def eval(params, env, rng):
    rewards = 0 
    obs = env.reset()
    while True: 
        rng, subkey = jax.random.split(rng, 2)
        a = eval_policy(params, obs, subkey)[0]
        a = onp.array(a)
        obs2, r, done, _ = env.step(a)        
        obs = obs2 
        rewards += r
        if done: break 
    return rewards

def sample_trajectory(traj, p):
    traj_len = int(traj[0].shape[0] * p)
    idxs = onp.random.choice(traj_len, size=traj_len, replace=False)
    sampled_traj = jax.tree_map(lambda x: x[idxs], traj)
    return sampled_traj

from utils import Cont_Vector_ReplayBuffer, discount_cumsum
def rollout(env, p_params, rng):
    buffer = Cont_Vector_ReplayBuffer(env, max_n_steps)
    buffer.clear()
    obs = env.reset()
    for _ in range(max_n_steps): 
        rng, subkey = jax.random.split(rng, 2)
        a, log_prob = policy(p_params, obs, subkey)

        a = jax.lax.stop_gradient(a)
        log_prob = jax.lax.stop_gradient(log_prob)

        obs2, r, done, _ = env.step(a)
        buffer.push((obs, a, r, obs2, done, log_prob))
        
        obs = obs2
        if done: break 

    trajectory = buffer.contents()
    return trajectory 

# inner optim 
@jax.jit
def reinforce_loss(p_params, obs, a, adv):
    mu, sig = p_frwd(p_params, obs)
    log_prob = distrax.MultivariateNormalDiag(mu, sig).log_prob(a)
    loss = -(log_prob * adv).sum()
    return loss

def reinforce_loss_batch(p_params, obs, a, adv):
    loss = jax.vmap(partial(reinforce_loss, p_params))(obs, a, adv).sum()
    return loss

reinforce_loss_grad = jax.jit(jax.value_and_grad(reinforce_loss_batch))

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

@jax.jit
def compute_advantage(W, traj):
    # linear fcn predict
    v_obs = v_features(traj['obs']) @ W
    # baseline 
    adv = traj['r'] - v_obs
    # normalize 
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv.squeeze()

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

rng = jax.random.PRNGKey(seed)
onp.random.seed(seed)

## model init
obs = np.zeros(env.observation_space.shape)  # dummy input 
p_params = policy_fcn.init(rng, obs) 

## save path 
model_path = pathlib.Path(f'./models/maml/{env_name}')
model_path.mkdir(exist_ok=True, parents=True)

# %%
task = env.sample_tasks(1)[0]
env.reset_task(task)

# %%
def maml_inner(p_params, env, rng, n_traj, alpha):
    subkeys = jax.random.split(rng, n_traj) 
    trajectories = []
    for i in range(n_traj):
        traj = rollout(env, p_params, subkeys[i]) 
        traj['r'] = discount_cumsum(traj['r'], discount=gamma)
        trajectories.append(traj)

    W = v_fit(trajectories)[0]
    gradients = []
    for traj in trajectories:
        adv = compute_advantage(W, traj)
        _, grad = reinforce_loss_grad(p_params, traj['obs'], traj['a'], adv)
        gradients.append(grad)
    grads = jax.tree_multimap(lambda *x: np.stack(x).sum(0), *gradients)
    inner_params_p = sgd_step(p_params, grads, alpha)

    return inner_params_p, W

n_traj = 2
newp, W = maml_inner(p_params, env, rng, 2, 0.1)
1

# %%
rng, subkey = jax.random.split(rng, 2) 
traj = rollout(env, p_params, subkey) 

traj['adv'] = compute_advantage(W, traj)
s_traj = sample(traj, 0.1)
1

#%%
loss, p_ngrad, alpha = batch_natural_grad(p_params, 
    s_traj['obs'], s_traj['a'], s_traj['adv'])
loss

#%%
obs, a, adv = s_traj['obs'][0], s_traj['a'][0], s_traj['adv'][0]

#%%
natural_grad(p_params, obs, a, adv)

#%%
loss, p_grads = jax.value_and_grad(reinforce_loss)(p_params, obs, a, adv)
f = lambda w: p_frwd(w, obs)
rho = D_KL_Gauss
mvp = pullback_mvp(f, rho, p_params, p_grads)
1

#%%
for p in jax.tree_leaves(p_ngrad):
    print(p.sum(), p.shape)

#%%
p_ngrad, _ = jax.scipy.sparse.linalg.cg(
        lambda v: pullback_mvp(f, rho, p_params, v),
        p_grads, maxiter=cg_iters)

#%%
### TRPO FCNS 
def hvp(J, w, v):
    return jax.jvp(jax.grad(J), (w,), (v,))[1]

# https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
def D_KL_Gauss(θ1, θ2):
    𝜇1, 𝜎1 = θ1
    𝜇2, 𝜎2 = θ2
    d_kl = np.log((𝜎2 / 𝜎1)+1e-8) + (𝜎1**2 + (𝜇1 - 𝜇2)**2) / (2*𝜎2**2 +1e-8) - .5
    d_kl = d_kl.sum() # sum over n_actions 
    return d_kl

def D_KL_params(p1, p2, obs):
    θ1, θ2 = p_frwd(p1, obs), p_frwd(p2, obs)
    return D_KL_Gauss(θ1, θ2)

def pullback_mvp(f, rho, w, v):
    z, R_z = jax.jvp(f, (w,), (v,))
    # rho diff 
    R_gz = hvp(lambda z1: rho(z, z1), z, R_z)
    _, f_vjp = jax.vjp(f, w)
    return f_vjp(R_gz)[0]

@jax.jit
def sgd_step(params, grads, alpha):
    sgd_update = lambda param, grad: param - alpha * grad
    return jax.tree_multimap(sgd_update, params, grads)

@jax.jit
def sgd_step_tree(params, grads, alphas):
    sgd_update = lambda param, grad, alpha: param - alpha * grad
    return jax.tree_multimap(sgd_update, params, grads, alphas)

# backtracking line-search 
tree_divide = lambda tree, denom: jax.tree_map(lambda x: x / denom, tree)
def line_search(alpha_start, init_loss, p_params, p_ngrad, traj, n_iters, delta):
    for i in np.arange(n_iters):
        alpha = tree_divide(alpha_start, 2 ** i)
        new_p_params = sgd_step_tree(p_params, p_ngrad, alpha)

        new_loss = reinforce_loss_batch(new_p_params, traj['obs'], traj['a'], traj['adv'])

        d_kl = jax.vmap(partial(D_KL_params, new_p_params, p_params))(traj['obs']).mean()

        if (new_loss < init_loss) and (d_kl <= delta): 
            return new_p_params # new weights 
    return p_params # no new weights 

def natural_grad(p_params, obs, a, adv):
    loss, p_grads = jax.value_and_grad(reinforce_loss)(p_params, obs, a, adv)
    f = lambda w: p_frwd(w, obs)
    rho = D_KL_Gauss
    p_ngrad, _ = jax.scipy.sparse.linalg.cg(
            lambda v: pullback_mvp(f, rho, p_params, v),
            p_grads, maxiter=cg_iters)
    
    # compute optimal step 
    vec = lambda x: x.flatten()[:, None]
    mat_mul = lambda x, y: np.sqrt(2 * delta / (vec(x).T @ vec(y)).flatten())
    alpha = jax.tree_multimap(mat_mul, p_grads, p_ngrad)

    return loss, p_ngrad, alpha

@jax.jit
def batch_natural_grad(p_params, obs, a, adv):
    out = jax.vmap(partial(natural_grad, p_params))(obs, a, adv)
    out = jax.tree_map(lambda x: x.mean(0), out)
    return out

def sample(traj, p):
    traj_len = int(traj['obs'].shape[0] * p)
    idxs = onp.random.choice(traj_len, size=traj_len, replace=False)
    sampled_traj = jax.tree_map(lambda x: x[idxs], traj)
    return sampled_traj

# %%


# %%
# %%
# %%
