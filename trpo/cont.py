# doesnt work :(

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
import scipy 

from jax.config import config
config.update("jax_enable_x64", True) 
config.update("jax_debug_nans", True) # break on nans

# env_name = 'Pendulum-v0' ## this env doesn't converge with TRPO (tried other impls too)
import pybullet_envs
env_name = 'HalfCheetahBulletEnv-v0' ## this works :)
env = gym.make(env_name)

n_actions = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]

a_high = env.action_space.high[0]
a_low = env.action_space.low[0]

print(f'[LOGGER] a_high: {a_high} a_low: {a_low} n_actions: {n_actions} obs_dim: {obs_dim}')
assert -a_high == a_low

#%%
init_final = hk.initializers.RandomUniform(-3e-3, 3e-3)

# def mu_scale(mu):
#     return np.tanh(mu) * a_high

def _policy_fcn(s):
    # log_std_init = lambda shape, dtype: -0.5*np.ones(shape, dtype)
    log_std = hk.get_parameter("log_std", shape=[n_actions,], init=np.zeros, dtype=np.float64)
    mu = hk.Sequential([
        hk.Linear(64), np.tanh,
        hk.Linear(64), np.tanh,
        hk.Linear(n_actions, b_init=np.zeros)
    ])(s)
    # mu = mu_scale(mu)
    std = np.exp(log_std)
    return mu, std

def _critic_fcn(s):
    v = hk.Sequential([
        hk.Linear(64), np.tanh,
        hk.Linear(64), np.tanh,
        hk.Linear(1), 
    ])(s)
    return v 

def normal_log_density(x, mean, std):
    log_std = np.log(std)
    var = np.power(std, 2)
    log_density = -np.power(x - mean, 2) / (
        2 * var) - 0.5 * np.log(2 * np.pi) - log_std
    return np.sum(log_density, 1, keepdims=True)

policy_fcn = hk.transform(_policy_fcn)
policy_fcn = hk.without_apply_rng(policy_fcn)
p_frwd = jax.jit(policy_fcn.apply)

critic_fcn = hk.transform(_critic_fcn)
critic_fcn = hk.without_apply_rng(critic_fcn)
v_frwd = jax.jit(critic_fcn.apply)

#%%
@jax.jit 
def policy(params, obs, rng):
    mu, std = p_frwd(params, obs)
    dist = distrax.MultivariateNormalDiag(mu, std)
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
        return onp.split(self.buffer[:self.i].astype(np.float64), self.splits, axis=-1)

    def clear(self):
        self.i = 0 
        self.buffer = onp.zeros((self.buffer_capacity, 2 * obs_dim + n_actions + 2 + 1))

def eval(params, env, rng):
    rewards = 0 
    running_state = ZFilter((obs_dim,), clip=5)
    obs = env.reset()
    obs = running_state(obs)
    while True: 
        rng, subrng = jax.random.split(rng)
        a = policy(params, obs, subrng)[0]
        a = onp.array(a)
        obs2, r, done, _ = env.step(a)        
        obs2 = running_state(obs2)
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
    # record un-normalized advs 
    writer.add_scalar('info/adv_mean', adv.mean().item(), p_step)
    writer.add_scalar('info/adv_std', adv.std().item(), p_step)

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # reward2go
    # finish rollout if not done 
    r[-1] = r[-1] if done[-1] else v_obs[-1].item()
    v_target = discount_cumsum(r, discount=gamma)

    # record targets
    writer.add_scalar('info/v_target_mean', v_target.mean().item(), p_step)
    writer.add_scalar('info/v_target_std', v_target.std().item(), p_step)
    
    return adv, v_target

# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = onp.zeros(shape)
        self._S = onp.zeros(shape)

    def push(self, x):
        x = onp.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else onp.square(self._M)

    @property
    def std(self):
        return onp.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape

class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = onp.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self, input_space):
        return input_space.shape

class Worker:
    def __init__(self, n_steps):
        self.n_steps = n_steps
        self.buffer = Vector_ReplayBuffer(n_steps)
        # import pybullet_envs
        # self.env = make_env()
        self.env = gym.make(env_name)
        self.obs = self.env.reset()
        
        self.running_state = ZFilter((obs_dim,), clip=5)
        self.epi_reward = 0 
        self.epi_rewards = []

    def rollout(self, p_params, v_params, rng):
        self.buffer.clear()
        
        for _ in range(self.n_steps): # rollout 
            rng, subrng = jax.random.split(rng)
            a, log_prob = policy(p_params, self.obs, subrng)
            a = onp.array(a)

            obs2, r, done, _ = self.env.step(a)
            obs2 = self.running_state(onp.array(obs2))
            self.epi_reward += r 

            self.buffer.push((self.obs, a, r, obs2, done, log_prob))
            self.obs = obs2
            if done: 
                self.obs = self.env.reset()
                self.obs = self.running_state(onp.array(self.obs))
                
                self.epi_rewards.append(self.epi_reward)
                self.epi_reward = 0 

        print(f'mean_reward = {onp.mean(self.epi_rewards)}')
        self.epi_rewards = []
        
        # update rollout contents 
        rollout = self.buffer.contents()
        advantages, v_target = compute_advantage_targets(v_params, rollout)
        (obs, a, _, _, _, log_prob) = rollout
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
seed = 85014 # onp.random.randint(1e5)

max_n_steps = 1e6
n_step_rollout = 25000
# gae 
gamma = 0.995
lmbda = 0.97
# trpo 
delta = 0.01
damp_lambda = 1e-1
n_search_iters = 10 
cg_iters = 10

rng = jax.random.PRNGKey(seed)
onp.random.seed(seed)
env.seed(seed)

obs = env.reset() # dummy input 
p_params = policy_fcn.init(rng, obs) 
v_params = critic_fcn.init(rng, obs) 

make_f64 = lambda params: jax.tree_map(lambda x: x.astype(np.float64), params)
p_params = make_f64(p_params)

worker = Worker(n_step_rollout)

# %%
def policy_loss(p_params, sample):
    (obs, a, old_log_prob, _, advantages) = sample 

    mu, std = p_frwd(p_params, obs)
    dist = distrax.MultivariateNormalDiag(mu, std)
    ratio = np.exp(dist.log_prob(a) - old_log_prob.squeeze())
    loss = -(ratio * advantages.squeeze()).sum()

    info = dict(entr=dist.entropy())
    return loss, info

tree_mean = lambda tree: jax.tree_map(lambda x: x.mean(0), tree)

@jax.jit
def batch_policy_loss(p_params, batch):
    return tree_mean(jax.vmap(partial(policy_loss, p_params))(batch))

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
@jax.jit
def D_KL_Gauss(θ1, θ2):
    mu0, std0 = θ1
    mu1, std1 = θ2
    
    log_std0 = np.log(std0)
    log_std1 = np.log(std1)
    
    d_kl = log_std1 - log_std0 + (np.power(std0, 2) + np.power(mu0 - mu1, 2)) / (2.0 * np.power(std1, 2)) - 0.5
    return d_kl.sum() # sum over actions

@jax.jit
def D_KL_params(p1, p2, obs):
    θ1, θ2 = p_frwd(p1, obs), p_frwd(p2, obs)
    return D_KL_Gauss(θ1, θ2)

def pullback_mvp(f, rho, w, v):
    # J 
    z, R_z = jax.jvp(f, (w,), (v,))
    # H J
    R_gz = hvp(lambda z1: rho(z, z1), z, R_z)
    _, f_vjp = jax.vjp(f, w)
    # (HJ)^T J = J^T H J 
    return f_vjp(R_gz)[0]

@jax.jit
def sgd_step(params, grads, alpha):
    sgd_update = lambda param, grad: param - alpha * grad
    return jax.tree_multimap(sgd_update, params, grads)

@jax.jit
def sgd_step_tree(params, grads, alphas):
    sgd_update = lambda param, grad, alpha: param - alpha * grad
    return jax.tree_multimap(sgd_update, params, grads, alphas)

tree_op = lambda op: lambda tree, arg2: jax.tree_map(lambda x: op(x, arg2), tree)
import operator
tree_divide = tree_op(operator.truediv)
tree_mult = tree_op(operator.mul)

# backtracking line-search 
def line_search(alpha_start, init_loss, p_params, p_ngrad, rollout, n_iters, delta):
    obs = rollout[0]
    for i in np.arange(n_iters):
        alpha = tree_mult(alpha_start, 0.9 ** i)

        new_p_params = sgd_step_tree(p_params, p_ngrad, alpha)
        new_loss, _ = batch_policy_loss(new_p_params, rollout)

        d_kl = jax.vmap(partial(D_KL_params, new_p_params, p_params))(obs).mean()

        if (new_loss < init_loss) and (d_kl <= delta): 
            writer.add_scalar('info/line_search_n_iters', i, p_step)
            writer.add_scalar('info/d_kl', d_kl.item(), p_step)
            writer.add_scalar('info/init_loss', (init_loss).item(), p_step)
            writer.add_scalar('info/new_loss', (new_loss).item(), p_step)
            writer.add_scalar('info/loss_diff_p', ((new_loss - init_loss) / init_loss).item(), p_step)
            return new_p_params # new weights 

    writer.add_scalar('info/line_search_n_iters', -1, p_step)
    return p_params # no new weights 

def line_search2(full_step, expected_improve_rate, p_params, rollout, n_iters=10, accept_ratio=0.1):
    L = lambda p: batch_policy_loss(p, rollout)[0]
    init_loss = L(p_params)
    print("loss before", init_loss.item())
    for i in np.arange(n_iters):
        step_frac = .5 ** i
        step = tree_mult(full_step, step_frac)
        new_p_params = jax.tree_map(lambda p, s: p+s, p_params, step)
        new_loss = L(new_p_params)

        actual_improve = init_loss - new_loss
        expected_improve = expected_improve_rate * step_frac
        ratio = actual_improve / expected_improve

        print(f"{i} a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

        if (actual_improve.item() > 0) and (ratio.item() > accept_ratio): 
            print("loss after", new_loss.item())
            writer.add_scalar('info/line_search_n_iters', i, p_step)
            return new_p_params # new weights 

    writer.add_scalar('info/line_search_n_iters', -1, p_step)
    return p_params # no new weights 

policy_loss_grad = jax.jit(jax.value_and_grad(policy_loss, has_aux=True))

def tree_mvp_dampen(mvp, lmbda=0.1):
    dampen_fcn = lambda mvp_, v_: mvp_ + lmbda * v_
    damp_mvp = lambda v: jax.tree_multimap(dampen_fcn, mvp(v), v)
    return damp_mvp

def jax_conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = np.zeros_like(b)
    r = np.zeros_like(b) + b 
    p = np.zeros_like(b) + b 
    rdotr = np.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / np.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = np.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        
        if rdotr < residual_tol:
            break
    return x

def lax_jax_conjugate_gradients(Avp, b, nsteps):
    x = np.zeros_like(b)
    r = np.zeros_like(b) + b 
    p = np.zeros_like(b) + b 
    residual_tol=1e-10

    cond = lambda v: v['step'] < nsteps
    def body(v):
        p, r, x = v['p'], v['r'], v['x']
        rdotr = np.dot(r, r)
        _Avp = Avp(p)
        alpha = rdotr / np.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = np.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        new_step = jax.lax.cond(rdotr < residual_tol, lambda s: s + nsteps, lambda s: s +1, v['step'])
        return {'step': new_step, 'p': p, 'r': r, 'x': x}

    init = {'step': 0, 'p': p, 'r': r, 'x': x}
    x = jax.lax.while_loop(cond, body, init)['x']
    return x

def natural_grad(p_params, p_grads, sample):
    obs = sample[0]
    f = lambda w: p_frwd(w, obs)
    rho = D_KL_Gauss
    mvp = tree_mvp_dampen(lambda v: pullback_mvp(f, rho, p_params, v), damp_lambda) 
    flat_grads, unflatten_fcn = jax.flatten_util.ravel_pytree(p_grads)
    flatten = lambda x: jax.flatten_util.ravel_pytree(x)[0]
    flat_mvp = lambda v: flatten(mvp(unflatten_fcn(v)))

    # step_dir, _ = jax.scipy.sparse.linalg.cg(flat_mvp, flat_neg_grads, maxiter=cg_iters, tol=1e-10)
    step_dir = lax_jax_conjugate_gradients(flat_mvp, -flat_grads, 10)
    
    # # compute optimal step (my way -- correct per-layer alpha?) 
    # vec = lambda x: x.flatten()[:, None]
    # # sometimes matMul is negative which when + np.sqrt results in NaN
    # safe_mat_mul_denom = lambda x, Hx: np.maximum(vec(x).T @ vec(Hx), 1e-8).flatten()
    # mat_mul = lambda x, Hx: np.sqrt(2 * delta / safe_mat_mul_denom(x, Hx))
    # alpha = jax.tree_multimap(mat_mul, p_grads, p_ngrad)

    # flat_step_dir, unflatten_fcn = jax.flatten_util.ravel_pytree(step_dir)
    # flat_step_dir_mvp, _ = jax.flatten_util.ravel_pytree(mvp(step_dir))
    
    shs = .5 * (step_dir * flat_mvp(step_dir)).sum()
    lm = np.sqrt(shs / 1e-2)
    fullstep = step_dir / lm

    neggdotstepdir = (-flat_grads * step_dir).sum()

    fullstep = unflatten_fcn(fullstep)
    expected_improve_rate = neggdotstepdir / lm 

    return None, None, None, None, None, (fullstep, expected_improve_rate, lm, flat_grads)

def batch_natural_grad(p_params, batch):
    (loss, info), p_grads = tree_mean(jax.vmap(policy_loss_grad, (None, 0))(p_params, batch))
    out = jax.vmap(partial(natural_grad, p_params, p_grads))(batch)
    out = tree_mean(out)
    return out

def sample_rollout(rollout, p):
    rollout_len = int(rollout[0].shape[0] * p)
    idxs = onp.random.choice(rollout_len, size=rollout_len, replace=False)
    sampled_rollout = jax.tree_map(lambda x: x[idxs], rollout)
    return sampled_rollout

#%%
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment=f'trpo_{env_name}_seed={seed}_nrollout={n_step_rollout}_2')
p_step = 0 
v_step = 0 

# #%%
# rng, subkey = jax.random.split(rng, 2) 
# rollout = worker.rollout(p_params, v_params, subkey)

# #%%
# # train
# sampled_rollout = sample_rollout(rollout, 0.1) # natural grad on 10% of data
# loss, info, p_ngrad, alpha, p_grads, (fullstep, expected_improve_rate) = batch_natural_grad(p_params, sampled_rollout)
# loss, expected_improve_rate

# #%%
# new_p = line_search2(fullstep, expected_improve_rate, p_params, rollout)

#%%
# import scipy 
# flat_v_params, unflatten_fcn = jax.flatten_util.ravel_pytree(v_params)
# loss = lambda p: batch_critic_loss(unflatten_fcn(p), rollout).astype(np.double)
# loss_grad_fcn = lambda p: jax.tree_map(lambda x: onp.array(x).astype(onp.double), jax.value_and_grad(loss)(p))
# flat_vp_params, _, _ = scipy.optimize.fmin_l_bfgs_b(loss_grad_fcn, onp.array(flat_v_params).astype(onp.double), maxiter=25)
# v_params = unflatten_fcn(flat_vp_params)

#%%
from tqdm import tqdm 
# from tqdm.notebook import tqdm 
pbar = tqdm(total=max_n_steps)
while p_step < max_n_steps: 
    # rollout
    rng, subkey = jax.random.split(rng, 2) 
    rollout = worker.rollout(p_params, v_params, subkey)

    # train
    sampled_rollout = sample_rollout(rollout, 0.1) # natural grad on 10% of data
    # loss, info, p_ngrad, alpha, p_grads = batch_natural_grad(p_params, sampled_rollout)
    # p_params = line_search(alpha, loss, p_params, p_ngrad, rollout, n_search_iters, delta)

    _, _, _, _, _, (fullstep, expected_improve_rate, lm, flat_grads) = batch_natural_grad(p_params, sampled_rollout)
    print("lagrange multiplier:", lm.item(), "grad_norm:", np.linalg.norm(flat_grads).item())

    p_params = line_search2(fullstep, expected_improve_rate, p_params, rollout)

    # writer.add_scalar('info/ploss', loss.item(), p_step)
    # for k in info.keys(): 
    #     writer.add_scalar(f'info/{k}', info[k].item(), p_step)

    p_step += 1
    pbar.update(1)

    # v update 
    flat_v_params, unflatten_fcn = jax.flatten_util.ravel_pytree(v_params)
    v_loss_fcn = lambda p: batch_critic_loss(unflatten_fcn(p), rollout).astype(np.double)
    v_loss_grad_fcn = lambda p: jax.tree_map(lambda x: onp.array(x).astype(onp.double), jax.value_and_grad(v_loss_fcn)(p))
    flat_vp_params, _, _ = scipy.optimize.fmin_l_bfgs_b(v_loss_grad_fcn, onp.array(flat_v_params).astype(onp.double), maxiter=25)
    v_params = unflatten_fcn(flat_vp_params)

    # # v_func
    # for _ in range(n_v_iters):
    #     loss, v_params, v_opt_state = critic_step(v_params, v_opt_state, rollout)
    #     writer.add_scalar('info/vloss', loss.item(), v_step)
    #     v_step += 1

    # # metrics 
    # for i, g in enumerate(jax.tree_leaves(p_params)): 
    #     name = 'b' if len(g.shape) == 1 else 'w'
    #     writer.add_histogram(f'{name}_{i}_params', onp.array(g), p_step)
    
    # for i, g in enumerate(jax.tree_leaves(p_grads)): 
    #     name = 'b' if len(g.shape) == 1 else 'w'
    #     writer.add_histogram(f'{name}_{i}_grad', onp.array(g), p_step)
    
    # for i, g in enumerate(jax.tree_leaves(p_ngrad)): 
    #     name = 'b' if len(g.shape) == 1 else 'w'
    #     writer.add_histogram(f'{name}_{i}_ngrad', onp.array(g), p_step)
    
    # for i, g in enumerate(jax.tree_leaves(alpha)): 
    #     writer.add_scalar(f'alpha/{i}', g.item(), p_step)

    rng, subkey = jax.random.split(rng, 2)
    r = eval(p_params, env, subkey)
    writer.add_scalar('eval/total_reward', r, p_step)

# except: 
#     print('err!')
