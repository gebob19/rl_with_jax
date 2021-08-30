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

#%%
# https://github.com/ikostrikov/pytorch-trpo
import torch
import torch.autograd as autograd
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

policy_net = Policy(obs_dim, n_actions)

#%%
params = []
for linear in [policy_net.affine1, policy_net.affine2, policy_net.action_mean]:
    w = onp.array(linear.weight.data)
    b = onp.array(linear.bias.data)
    params.append((w, b))

#%%
class CustomWInit(hk.initializers.Initializer):
    def __init__(self, i) -> None:
        super().__init__()
        self.i = i 
    def __call__(self, shape, dtype):
        return params[self.i][0].T
class CustomBInit(hk.initializers.Initializer):
    def __init__(self, i) -> None:
        super().__init__()
        self.i = i 
    def __call__(self, shape, dtype):
        return params[self.i][1].T

def _policy_fcn(s):
    log_std = hk.get_parameter("log_std", shape=[n_actions,], init=np.zeros)
    mu = hk.Sequential([
        hk.Linear(64, w_init=CustomWInit(0), b_init=CustomBInit(0)), np.tanh,
        hk.Linear(64, w_init=CustomWInit(1), b_init=CustomBInit(1)), np.tanh,
        hk.Linear(n_actions, w_init=CustomWInit(2), b_init=CustomBInit(2))
    ])(s)
    std = np.exp(log_std)
    return mu, log_std, std

policy_fcn = hk.transform(_policy_fcn)
policy_fcn = hk.without_apply_rng(policy_fcn)
p_frwd = jax.jit(policy_fcn.apply)

seed = 0 
rng = jax.random.PRNGKey(seed)
onp.random.seed(seed)
env.seed(seed)

obs = env.reset() # dummy input 
p_params = policy_fcn.init(rng, obs) 

# %%
t_obs = torch.from_numpy(obs).float()[None]
n_obs = t_obs.numpy()

p_frwd(p_params, n_obs), policy_net(t_obs)

# %%
a = env.action_space.sample()
obs2, r, done, _ = env.step(a)

# %%
ta = torch.from_numpy(a).float()
na = ta.numpy()

import math 
def normal_log_density_jax(x, mean, log_std, std):
    var = np.power(std, 2)
    log_density = -np.power(x - mean, 2) / (
        2 * var) - 0.5 * np.log(2 * np.pi) - log_std
    return np.sum(log_density, 1, keepdims=True)

def get_loss_jax(p_params):
    action_means, action_log_stds, action_stds = p_frwd(p_params, n_obs)
            
    log_prob = normal_log_density_jax(na, action_means, action_log_stds, action_stds)
    fixed_log_prob = jax.lax.stop_gradient(log_prob)
    action_loss = -r * np.exp(log_prob - fixed_log_prob)
    return action_loss.mean()

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)

def get_loss_torch():
    action_means, action_log_stds, action_stds = policy_net(t_obs)
            
    log_prob = normal_log_density(ta, action_means, action_log_stds, action_stds)
    fixed_log_prob = log_prob.detach()
    action_loss = -r * torch.exp(log_prob - fixed_log_prob)
    return action_loss.mean()

get_loss_torch(), get_loss_jax(p_params)

# %%
from torch.autograd import Variable
def get_kl():
    mean1, log_std1, std1 = policy_net(t_obs)

    mean0 = Variable(mean1.data)
    log_std0 = Variable(log_std1.data)
    std0 = Variable(std1.data)
    kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
    return kl.sum(1, keepdim=True)

def D_KL_Gauss(θ1, θ2):
    mu0, _, std0 = θ1
    mu1, _, std1 = θ2

    log_std0 = np.log(std0)
    log_std1 = np.log(std1)
    
    d_kl = log_std1 - log_std0 + (np.power(std0, 2) + np.power(mu0 - mu1, 2)) / (2.0 * np.power(std1, 2)) - 0.5
    return d_kl.sum() # sum over actions

theta = p_frwd(p_params, n_obs)
get_kl(), D_KL_Gauss(theta, theta)

# %%
loss = get_loss_torch()
grads = torch.autograd.grad(loss, policy_net.parameters())
for g in grads: print(g.shape)

# %%
jax_grads = jax.grad(get_loss_jax)(p_params)
for g in jax.tree_leaves(jax_grads): print(g.shape)

# %%
def Fvp(v):
    kl = get_kl()
    kl = kl.mean()

    grads = torch.autograd.grad(kl, policy_net.parameters(), create_graph=True)
    flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

    kl_v = (flat_grad_kl * Variable(v)).sum()
    grads = torch.autograd.grad(kl_v, policy_net.parameters())
    flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

    return flat_grad_grad_kl

def unflat(model, flat_params):
    prev_ind = 0
    params = []
    for param in model.parameters():
        flat_size = int(onp.prod(list(param.size())))
        params.append(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size
    return params

loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
vp = Fvp(-loss_grad)
vp = unflat(policy_net, vp)

# %%
def hvp(J, w, v):
    return jax.jvp(jax.grad(J), (w,), (v,))[1]

def pullback_mvp(f, rho, w, v):
    # J 
    z, R_z = jax.jvp(f, (w,), (v,))
    # H J
    R_gz = hvp(lambda z1: rho(z, z1), z, R_z)
    _, f_vjp = jax.vjp(f, w)
    # (HJ)^T J = J^T H J 
    return f_vjp(R_gz)[0]

f = lambda w: p_frwd(w, n_obs)
rho = D_KL_Gauss
neg_jax_grads = jax.tree_map(lambda x: -1 * x, jax_grads)
vp_jax = pullback_mvp(f, rho, p_params, neg_jax_grads)

# %%
vp_jax['~']['log_std'], vp[0]

# %%
vp_v = []
for i in range(1, len(vp), 2):
    vp_v.append((vp[i], vp[i+1]))

jax_v = []
for k in list(vp_jax.keys())[:-1]:
    jax_v.append((vp_jax[k]['w'], vp_jax[k]['b']))

# %%
for (tw, tb), (jw, jb) in zip(vp_v, jax_v):
    print(np.mean(np.abs(tw.numpy() - jw.T)))
    print(np.mean(np.abs(tb.numpy() - jb.T)))

# %%
def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

stepdir = conjugate_gradients(Fvp, -loss_grad, 10)
stepdir.shape

# %%
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
    residual_tol = 1e-10

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

mvp = lambda v: pullback_mvp(f, rho, p_params, v)
neg_grads = jax.tree_map(lambda x: -1 * x, jax_grads)
flat_grads, unflatten_fcn = jax.flatten_util.ravel_pytree(jax_grads)
flat_mvp = lambda v: jax.flatten_util.ravel_pytree(mvp(unflatten_fcn(v)))[0]
stepdir_jax = lax_jax_conjugate_gradients(flat_mvp, -flat_grads, 10)
stepdir_jax.shape

# %%
np.abs(stepdir_jax - stepdir.numpy()).mean() # 0.00201443

# %%
shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
lm = torch.sqrt(shs / 1e-2)
fullstep = stepdir / lm[0]
neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
expected_improve_rate = neggdotstepdir / lm[0]

lm, expected_improve_rate

# %%
shs_j = .5 * (stepdir_jax * flat_mvp(stepdir_jax)).sum()
lm = np.sqrt(shs_j / 1e-2)
fullstep = stepdir_jax / lm

neggdotstepdir = (-flat_grads * stepdir_jax).sum()

fullstep = unflatten_fcn(fullstep)
expected_improve_rate = neggdotstepdir / lm 

lm, expected_improve_rate

# %%
f = lambda w: p_frwd(w, n_obs)
rho = D_KL_Gauss
neg_jax_grads = jax.tree_map(lambda x: -1 * x, jax_grads)
mvp = lambda v: pullback_mvp(f, rho, p_params, v)
p_ngrad, _ = jax.scipy.sparse.linalg.cg(mvp,
            neg_jax_grads, maxiter=10, tol=1e-10)
flat_ngrad, _ = jax.flatten_util.ravel_pytree(p_ngrad)

np.abs(flat_ngrad - stepdir.numpy()).mean() # 475711.62

# %%





