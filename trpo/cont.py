#%%
import jax 
import jax.numpy as np 
import numpy as onp 
import distrax 
import optax
import gym 
from functools import partial
import cloudpickle
import haiku as hk

from jax.config import config
config.update("jax_enable_x64", True) 
config.update("jax_debug_nans", True) # break on nans

env_name = 'Pendulum-v0'
env = gym.make(env_name)

n_actions = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]

a_high = env.action_space.high[0]
a_low = env.action_space.low[0]

print(f'[LOGGER] a_high: {a_high} a_low: {a_low} n_actions: {n_actions} obs_dim: {obs_dim}')
assert -a_high == a_low

#%%
init_final = hk.initializers.RandomUniform(-3e-3, 3e-3)

def mu_scale(mu):
    return np.tanh(mu) * a_high

def _policy_fcn(s):
    log_std = hk.get_parameter("log_std", shape=[n_actions,], init=np.ones, dtype=np.float64)
    mu = hk.Sequential([
        hk.Linear(100), jax.nn.relu,
        hk.Linear(100), jax.nn.relu,
        hk.Linear(n_actions)
    ])(s)
    mu = mu_scale(mu)
    # log_std = np.clip(log_std, -2, 2) # don't let it explode
    sig = np.exp(log_std)
    return mu, sig

def _critic_fcn(s):
    v = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
        hk.Linear(1, w_init=init_final), 
    ])(s)
    return v 

policy_fcn = hk.transform(_policy_fcn)
policy_fcn = hk.without_apply_rng(policy_fcn)
p_frwd = jax.jit(policy_fcn.apply)

critic_fcn = hk.transform(_critic_fcn)
critic_fcn = hk.without_apply_rng(critic_fcn)
v_frwd = jax.jit(critic_fcn.apply)

#%%
@jax.jit 
def policy(params, obs, rng):
    mu, sig = p_frwd(params, obs)
    dist = distrax.MultivariateNormalDiag(mu, sig)
    a = dist.sample(seed=rng)
    a = np.clip(a, a_low, a_high)
    log_prob = dist.log_prob(a)
    return a, log_prob

@jax.jit 
def eval_policy(params, obs, _):
    a, _ = p_frwd(params, obs)
    a = np.clip(a, a_low, a_high)
    return a, None

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

def eval(params, env, rng):
    rewards = 0 
    obs = env.reset()
    while True: 
        rng, subrng = jax.random.split(rng)
        a = policy(params, obs, subrng)[0]
        a = onp.array(a)
        obs2, r, done, _ = env.step(a)        
        obs = obs2 
        rewards += r
        if done: break 
    return rewards

def discount_cumsum(l, discount):
    l = onp.array(l)
    for i in range(len(l) - 1)[::-1]:
        l[i] = l[i] + discount * l[i+1]
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

class Worker:
    def __init__(self, n_steps):
        self.n_steps = n_steps
        self.buffer = Vector_ReplayBuffer(1e6)
        # import pybullet_envs
        # self.env = make_env()
        self.env = gym.make(env_name)
        self.obs = self.env.reset()

    def rollout(self, p_params, v_params, rng):
        self.buffer.clear()
        
        for _ in range(self.n_steps): # rollout 
            rng, subrng = jax.random.split(rng)
            a, log_prob = policy(p_params, self.obs, subrng)
            a = onp.array(a)

            obs2, r, done, _ = self.env.step(a)

            self.buffer.push((self.obs, a, r, obs2, done, log_prob))
            self.obs = obs2
            if done: 
                self.obs = self.env.reset()

        # update rollout contents 
        rollout = self.buffer.contents()
        advantages, v_target = compute_advantage_targets(v_params, rollout)
        (obs, a, r, _, _, log_prob) = rollout
        rollout = (obs, a, log_prob, v_target, advantages)
        
        return rollout

def optim_update_fcn(optim):
    @jax.jit
    def update_step(params, grads, opt_state):
        grads, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(params, grads)
        return params, opt_state
    return update_step

#%%
seed = onp.random.randint(1e5)

max_n_steps = 1e6
n_step_rollout = 4000 
batch_size = 128 
n_p_iters = 10 
# v training
v_lr = 1e-3
n_v_iters = 80
# gae 
gamma = 0.99 
lmbda = 0.95
# trpo 
delta = 0.01
n_search_iters = 10 
cg_iters = 10

rng = jax.random.PRNGKey(seed)
onp.random.seed(seed)

obs = env.reset() # dummy input 
p_params = policy_fcn.init(rng, obs) 
v_params = critic_fcn.init(rng, obs) 

worker = Worker(n_step_rollout)

optimizer = lambda lr: optax.chain(
    optax.scale_by_adam(),
    optax.scale(-lr),
)
v_optim = optimizer(v_lr)
v_opt_state = v_optim.init(v_params)
v_update_fcn = optim_update_fcn(v_optim)

# %%
def policy_loss(p_params, sample):
    (obs, a, _, _, advantages) = sample 

    mu, sig = p_frwd(p_params, obs)
    dist = distrax.MultivariateNormalDiag(mu, sig)
    # ratio = np.exp(dist.log_prob(a) - old_log_prob)
    loss = -(np.exp(dist.log_prob(a)) * advantages).sum()
    return loss

@jax.jit
def batch_policy_loss(p_params, batch):
    return jax.vmap(partial(policy_loss, p_params))(batch).mean()

def critic_loss(v_params, sample):
    (obs, _, _, v_target, _) = sample 

    v_obs = v_frwd(v_params, obs)
    loss = (0.5 * ((v_obs - v_target) ** 2)).sum()
    return loss

def batch_critic_loss(v_params, batch):
    return jax.vmap(partial(critic_loss, v_params))(batch).mean()

@jax.jit
def critic_step(v_params, opt_state, batch):
    loss, grads = jax.value_and_grad(batch_critic_loss)(v_params, batch)
    v_params, opt_state = v_update_fcn(v_params, grads, opt_state)
    return loss, v_params, opt_state

def hvp(J, w, v):
    return jax.jvp(jax.grad(J), (w,), (v,))[1]

# https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
def D_KL_Gauss(θ1, θ2):
    𝜇1, 𝜎1 = θ1
    𝜇2, 𝜎2 = θ2
    d_kl = np.log(𝜎2 / 𝜎1) + (𝜎1**2 + (𝜇1 - 𝜇2)**2) / (2*𝜎2**2) - .5
    return d_kl.sum() # sum over

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
def line_search(alpha_start, init_loss, p_params, p_ngrad, rollout, n_iters, delta):
    obs = rollout[0]
    for i in np.arange(n_iters):
        alpha = tree_divide(alpha_start, 2 ** i)
        new_p_params = sgd_step_tree(p_params, p_ngrad, alpha)

        new_loss = batch_policy_loss(new_p_params, rollout)

        d_kl = jax.vmap(partial(D_KL_params, new_p_params, p_params))(obs).mean()

        if (new_loss < init_loss) and (d_kl <= delta): 
            writer.add_scalar('info/line_search_n_iters', i, p_step)
            return new_p_params # new weights 

    writer.add_scalar('info/line_search_n_iters', -1, p_step)
    return p_params # no new weights 

def natural_grad(p_params, sample):
    obs = sample[0]
    loss, p_grads = jax.value_and_grad(policy_loss)(p_params, sample)
    f = lambda w: p_frwd(w, obs)
    rho = D_KL_Gauss
    p_ngrad, _ = jax.scipy.sparse.linalg.cg(
            lambda v: pullback_mvp(f, rho, p_params, v),
            p_grads, maxiter=cg_iters)
    
    # compute optimal step 
    vec = lambda x: x.flatten()[:, None]
    mat_mul = lambda x, y: np.sqrt(2 * delta / (vec(x).T @ vec(y) + 1e-8).flatten())
    alpha = jax.tree_multimap(mat_mul, p_grads, p_ngrad)

    return loss, p_ngrad, alpha, p_grads

@jax.jit
def batch_natural_grad(p_params, batch):
    out = jax.vmap(partial(natural_grad, p_params))(batch)
    out = jax.tree_map(lambda x: x.mean(0), out)
    return out

def sample_rollout(rollout, p):
    if p < 1.: rollout_len = int(rollout[0].shape[0] * p)
    else: rollout_len = p 
    idxs = onp.random.choice(rollout_len, size=rollout_len, replace=False)
    sampled_rollout = jax.tree_map(lambda x: x[idxs], rollout)
    return sampled_rollout

# #%%
# # rollout
# rng, subkey = jax.random.split(rng, 2) 
# rollout = worker.rollout(p_params, v_params, subkey)

# #%%
# sample = [r[0] for r in rollout]
# policy_loss(p_params, sample)

# #%%
# jax.vmap(partial(policy_loss, p_params))(rollout)[0]

#%%
#%%
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment=f'trpo_{env_name}_seed={seed}_nrollout={n_step_rollout}')

#%%
from tqdm import tqdm 

p_step = 0 
v_step = 0 
pbar = tqdm(total=max_n_steps)
while p_step < max_n_steps: 
    # rollout
    rng, subkey = jax.random.split(rng, 2) 
    rollout = worker.rollout(p_params, v_params, subkey)

    # train
    sampled_rollout = sample_rollout(rollout, 0.1) # natural grad on 10% of data
    loss, p_ngrad, alpha, p_grads = batch_natural_grad(p_params, sampled_rollout)
    p_params = line_search(alpha, loss, p_params, p_ngrad, rollout, n_search_iters, delta)
    writer.add_scalar('info/ploss', loss.item(), p_step)
    p_step += 1
    pbar.update(1)

    # v_func
    for _ in range(n_v_iters):
        loss, v_params, v_opt_state = critic_step(v_params, v_opt_state, rollout)
        writer.add_scalar('info/vloss', loss.item(), v_step)
        v_step += 1

    # print('===============')
    # print('-----PARAMS')
    # for i, g in enumerate(jax.tree_leaves(p_params)): 
    #     name = 'b' if len(g.shape) == 1 else 'w'
    #     print(onp.array(g))
    #     writer.add_histogram(f'{name}_{i}_params', onp.array(g), p_step)
    # print('-----GRAD')
    # for i, g in enumerate(jax.tree_leaves(p_grads)): 
    #     name = 'b' if len(g.shape) == 1 else 'w'
    #     print(onp.array(g))
    #     writer.add_histogram(f'{name}_{i}_grad', onp.array(g), p_step)
    # print('-----NGRAD----')
    # for i, g in enumerate(jax.tree_leaves(p_ngrad)): 
    #     name = 'b' if len(g.shape) == 1 else 'w'
    #     print(onp.array(g))
    #     writer.add_histogram(f'{name}_{i}_ngrad', onp.array(g), p_step)
    # print('-----ALPHA')
    # for i, g in enumerate(jax.tree_leaves(alpha)): 
    #     name = 'b' if len(g.shape) == 1 else 'w'
    #     print(onp.array(g))
    #     writer.add_histogram(f'{name}_{i}_alpha', onp.array(g), p_step)

    rng, subkey = jax.random.split(rng, 2)
    r = eval(p_params, env, subkey)
    writer.add_scalar('eval/total_reward', r, p_step)

# %%

# %%
# %%
