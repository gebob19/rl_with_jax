# works :) 

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

env_name = 'CartPole-v0'
env = gym.make(env_name)

n_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]

print(f'[LOGGER] n_actions: {n_actions} obs_dim: {obs_dim}')

#%%
init_final = hk.initializers.RandomUniform(-3e-3, 3e-3)

def _policy_fcn(s):
    pi = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
        hk.Linear(n_actions, w_init=init_final), jax.nn.softmax
    ])(s)
    return pi

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

#%%
@jax.jit 
def policy(params, obs, rng):
    pi = p_frwd(params, obs)
    dist = distrax.Categorical(probs=pi)
    a = dist.sample(seed=rng)
    log_prob = dist.log_prob(a)
    return a, log_prob

class Vector_ReplayBuffer:
    def __init__(self, buffer_capacity):
        self.buffer_capacity = buffer_capacity = int(buffer_capacity)
        self.i = 0
        # obs, obs2, a, r, done
        self.splits = [obs_dim, obs_dim+1, obs_dim+1+1, obs_dim*2+1+1, obs_dim*2+1+1+1]
        self.clear()

    def push(self, sample):
        assert self.i < self.buffer_capacity # dont let it get full
        (obs, a, r, obs2, done, log_prob) = sample
        self.buffer[self.i] = onp.array([*obs, onp.array(a), onp.array(r), *obs2, float(done), onp.array(log_prob)])
        self.i += 1 
    
    def contents(self):
        return onp.split(self.buffer[:self.i], self.splits, axis=-1)

    def clear(self):
        self.i = 0 
        self.buffer = onp.zeros((self.buffer_capacity, 2 * obs_dim + 1 + 2 + 1))

def eval(params, env, rng):
    rewards = 0 
    obs = env.reset()
    while True: 
        rng, subrng = jax.random.split(rng)
        a = policy(params, obs, subrng)[0].item()
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
            a = a.item()

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
seed = 24910 #onp.random.randint(1e5)

epochs = 1000
n_step_rollout = 4000 
# v training
v_lr = 1e-3
n_v_iters = 80
# gae 
gamma = 0.99 
lmbda = 0.95
# trpo 
delta = 0.01
damp_lambda = 1e-5
n_search_iters = 10 
cg_iters = 10

rng = jax.random.PRNGKey(seed)
onp.random.seed(seed)
env.seed(seed)

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
    (obs, a, old_log_prob, _, advantages) = sample 

    pi = p_frwd(p_params, obs)
    dist = distrax.Categorical(probs=pi)
    ratio = np.exp(dist.log_prob(a) - old_log_prob)
    loss = -(ratio * advantages).sum()
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

def D_KL(z1, z2):
    # p1, p2 = jax.nn.softmax(z1), jax.nn.softmax(z2)
    p1, p2 = z1, z2
    d_kl = (p1 * (np.log(p1) - np.log(p2))).sum()
    return d_kl

def D_KL_params(p1, p2, obs):
    z1, z2 = p_frwd(p1, obs), p_frwd(p2, obs)
    return D_KL(z1, z2)

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

        d_kl = jax.vmap(partial(D_KL_params, new_p_params, p_params))(obs).mean()

        if (new_loss < init_loss) and (d_kl <= delta): 
            writer.add_scalar('info/line_search_n_iters', i, e)
            return new_p_params # new weights 

    writer.add_scalar('info/line_search_n_iters', -1, e)
    return p_params # no new weights 

def tree_mvp_dampen(mvp, lmbda=0.1):
    dampen_fcn = lambda mvp_, v_: mvp_ + lmbda * v_
    damp_mvp = lambda v: jax.tree_multimap(dampen_fcn, mvp(v), v)
    return damp_mvp

def natural_grad(p_params, sample):
    obs = sample[0]
    loss, p_grads = jax.value_and_grad(policy_loss)(p_params, sample)
    f = lambda w: p_frwd(w, obs)
    rho = D_KL
    p_ngrad, _ = jax.scipy.sparse.linalg.cg(
            tree_mvp_dampen(lambda v: pullback_mvp(f, rho, p_params, v), damp_lambda),
            p_grads, maxiter=cg_iters)
    
    # compute optimal step 
    vec = lambda x: x.flatten()[:, None]
    mat_mul = lambda x, y: np.sqrt(2 * delta / (vec(x).T @ vec(y)).flatten())
    alpha = jax.tree_multimap(mat_mul, p_grads, p_ngrad)

    return loss, p_ngrad, alpha

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

#%%
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment=f'trpo_{env_name}_seed={seed}_nrollout={n_step_rollout}')

#%%
from tqdm import tqdm 

for e in tqdm(range(epochs)):
    # rollout
    rng, subkey = jax.random.split(rng, 2) 
    rollout = worker.rollout(p_params, v_params, subkey)

    # train
    sampled_rollout = sample_rollout(rollout, 0.1) # natural grad on 10% of data
    loss, p_ngrad, alpha = batch_natural_grad(p_params, sampled_rollout)
    for i, g in enumerate(jax.tree_leaves(alpha)): 
        writer.add_scalar(f'alpha/{i}', g.item(), e)
    
    # update 
    p_params = line_search(alpha, loss, p_params, p_ngrad, rollout, n_search_iters, delta)
    writer.add_scalar('info/ploss', loss.item(), e)

    v_loss = 0
    for _ in range(n_v_iters):
        loss, v_params, v_opt_state = critic_step(v_params, v_opt_state, rollout)
        v_loss += loss

    v_loss /= n_v_iters
    writer.add_scalar('info/vloss', v_loss.item(), e)

    for i, g in enumerate(jax.tree_leaves(p_ngrad)): 
        name = 'b' if len(g.shape) == 1 else 'w'
        writer.add_histogram(f'{name}_{i}_grad', onp.array(g), e)

    rng, subkey = jax.random.split(rng, 2)
    r = eval(p_params, env, subkey)
    writer.add_scalar('eval/total_reward', r, e)

# %%

# %%
# %%
