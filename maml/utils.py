import numpy as onp 
import jax.numpy as np 
import jax 
from functools import partial
import haiku as hk
import optax

def init_policy_fcn(type, env, rng, nhidden=64, jit=True):

    if type == 'continuous':
        n_actions = env.action_space.shape[0]
        a_high = env.action_space.high[0]
        a_low = env.action_space.low[0]
        assert -a_high == a_low

        def _cont_policy_fcn(s):
            log_std = hk.get_parameter("log_std", shape=[n_actions,], init=np.ones, dtype=np.float64)
            mu = hk.Sequential([
                hk.Linear(nhidden), jax.nn.relu,
                hk.Linear(nhidden), jax.nn.relu,
                hk.Linear(n_actions), np.tanh 
            ])(s) * a_high
            sig = np.exp(log_std)
            return mu, sig
        _policy_fcn = _cont_policy_fcn

    elif type == 'discrete':
        n_actions = env.action_space.n

        init_final = hk.initializers.RandomUniform(-3e-3, 3e-3)
        def _disc_policy_fcn(s):
            pi = hk.Sequential([
                hk.Linear(nhidden), jax.nn.relu,
                hk.Linear(nhidden), jax.nn.relu,
                hk.Linear(n_actions, w_init=init_final), jax.nn.softmax
            ])(s)
            return pi
        _policy_fcn = _disc_policy_fcn

    policy_fcn = hk.transform(_policy_fcn)
    policy_fcn = hk.without_apply_rng(policy_fcn)
    p_frwd = policy_fcn.apply
    if jit: 
        p_frwd = jax.jit(p_frwd)

    obs = np.zeros(env.observation_space.shape)  # dummy input 
    p_params = policy_fcn.init(rng, obs) 

    return p_frwd, p_params

def optim_update_fcn(optim, params):
    opt_state = optim.init(params)
    @jax.jit
    def update_step(params, opt_state, grads):
        grads, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(params, grads)
        return params, opt_state
    return update_step, opt_state

def gaussian_log_prob(x, mean, std):
    log_std = np.log(std)
    var = np.power(std, 2)
    log_density = -np.power(x - mean, 2) / (
        2 * var) - 0.5 * np.log(2 * np.pi) - log_std
    return np.sum(log_density)

def gaussian_sample(mean, std, rng):
    return jax.random.normal(rng) * std + mean 

@partial(jax.jit, static_argnums=(0,))
def cont_policy(p_frwd, params, obs, rng, clip_range, greedy):
    mu, std = p_frwd(params, obs)
    a = jax.lax.cond(greedy, lambda _: mu, lambda _: gaussian_sample(mu, std, rng), None)
    a = np.clip(a, *clip_range) # [low, high]
    log_prob = gaussian_log_prob(a, mu, std)
    return a, log_prob

@partial(jax.jit, static_argnums=(0,))
def disc_policy(p_frwd, params, obs, rng, greedy):
    pi = p_frwd(params, obs)
    dist = distrax.Categorical(probs=pi)
    a = jax.lax.cond(greedy, lambda _: pi.argmax(), lambda _: dist.sample(seed=rng), None)
    a = dist.sample(seed=rng)
    log_prob = dist.log_prob(a)
    return a, log_prob

def eval(p_frwd, policy, params, env, rng, clip_range, greedy):
    rewards = 0 
    obs = env.reset()
    while True: 
        rng, subkey = jax.random.split(rng, 2)
        a = policy(p_frwd, params, obs, subkey, clip_range, greedy)[0]
        a = onp.array(a)
        obs2, r, done, _ = env.step(a)        
        obs = obs2 
        rewards += r
        if done: break 
    return rewards

class Cont_Vector_Buffer:
    def __init__(self, n_actions, obs_dim, buffer_capacity):
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.buffer_capacity = buffer_capacity = int(buffer_capacity)
        self.i = 0
        # obs, a, r, obs2, done
        self.splits = [obs_dim, obs_dim+n_actions, obs_dim+n_actions+1, obs_dim*2+1+n_actions, obs_dim*2+1+n_actions+1]
        self.split_names = ['obs', 'a', 'r', 'obs2', 'done', 'log_prob']
        self.clear()

    def push(self, sample):
        assert self.i < self.buffer_capacity # dont let it get full
        (obs, a, r, obs2, done, log_prob) = sample
        self.buffer[self.i] = onp.array([*obs, *onp.array(a), onp.array(r), *obs2, float(done), onp.array(log_prob)])
        self.i += 1 

    def contents(self):
        contents = onp.split(self.buffer[:self.i], self.splits, axis=-1)
        return contents
        # d = {}
        # for n, c in zip(self.split_names, contents): d[n] = c
        # return d 

    def clear(self):
        self.i = 0 
        self.buffer = onp.zeros((self.buffer_capacity, 2 * self.obs_dim + self.n_actions + 2 + 1))

class Disc_Vector_Buffer(Cont_Vector_Buffer):
    def __init__(self, obs_dim, buffer_capacity):
        super().__init__(1, obs_dim, buffer_capacity)

    def push(self, sample):
        assert self.i < self.buffer_capacity # dont let it get full
        (obs, a, r, obs2, done, log_prob) = sample
        self.buffer[self.i] = onp.array([*obs, onp.array(a), onp.array(r), *obs2, float(done), onp.array(log_prob)])
        self.i += 1 

## second order stuff (trpo)
def hvp(J, w, v):
    return jax.jvp(jax.grad(J), (w,), (v,))[1]

def gnh_vp(f, rho, w, v):
    z, R_z = jax.jvp(f, (w,), (v,))
    R_gz = hvp(lambda z1: rho(z, z1), z, R_z)
    _, f_vjp = jax.vjp(f, w)
    return f_vjp(R_gz)[0]

def tree_mvp_dampen(mvp, lmbda=0.1):
    dampen_fcn = lambda mvp_, v_: mvp_ + lmbda * v_
    damp_mvp = lambda v: jax.tree_multimap(dampen_fcn, mvp(v), v)
    return damp_mvp

def discount_cumsum(l, discount):
    l = onp.array(l)
    for i in range(len(l) - 1)[::-1]:
        l[i] = l[i] + discount * l[i+1]
    return l 

def tree_shape(tree): 
    for l in jax.tree_leaves(tree): print(l.shape)

tree_mean = jax.jit(lambda tree: jax.tree_map(lambda x: x.mean(0), tree))
tree_sum = jax.jit(lambda tree: jax.tree_map(lambda x: x.sum(0), tree))

def jit_vmap_tree_op(jit_tree_op, f, *vmap_args):
    return lambda *args: jit_tree_op(jax.vmap(f, *vmap_args)(*args))

mean_vmap_jit = partial(jit_vmap_tree_op, tree_mean)
sum_vmap_jit = partial(jit_vmap_tree_op, tree_sum)