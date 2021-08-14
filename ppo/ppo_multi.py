#%%
import jax
import jax.numpy as np 
import numpy as onp 
import distrax 
import optax
import gym 
from functools import partial
import ray 
import pybullet_envs

ray.init()

#%%
env_name = 'Pendulum-v0'
# env_name = 'HalfCheetahBulletEnv-v0'
env = gym.make(env_name)

# from env import Navigation2DEnv
# env_name = 'Navigation2D'
# def make_env():
#     env = Navigation2DEnv()
#     env.seed(0)
#     task = env.sample_tasks(1)[0]
#     env.reset_task(task)
#     return env 
# env = make_env()

n_actions = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]

a_high = env.action_space.high[0]
a_low = env.action_space.low[0]

print(f'[LOGGER] a_high: {a_high} a_low: {a_low} n_actions: {n_actions} obs_dim: {obs_dim}')
assert -a_high == a_low

#%%
import haiku as hk
init_final = hk.initializers.RandomUniform(-3e-3, 3e-3)

def _policy_fcn(s):
    log_std = hk.get_parameter("log_std", shape=[n_actions,], init=np.ones)
    mu = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
        hk.Linear(n_actions, w_init=init_final), np.tanh 
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

# %%
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

def discount_cumsum(l, discount):
    l = onp.array(l)
    for i in range(len(l) - 1)[::-1]:
        l[i] = l[i] + discount * l[i+1]
    return l 

@jax.jit
def ppo_loss(p_params, v_params, sample):
    (obs, a, old_log_prob, v_target, advantages) = sample 

    ## critic loss
    v_obs = v_frwd(v_params, obs)
    critic_loss = (0.5 * ((v_obs - v_target) ** 2)).sum()

    ## policy losses 
    mu, sig = p_frwd(p_params, obs)
    dist = distrax.MultivariateNormalDiag(mu, sig)
    # entropy 
    entropy_loss = -dist.entropy()
    # policy gradient 
    log_prob = dist.log_prob(a)

    approx_kl = (old_log_prob - log_prob).sum()
    ratio = np.exp(log_prob - old_log_prob)
    p_loss1 = ratio * advantages
    p_loss2 = np.clip(ratio, 1-eps, 1+eps) * advantages
    policy_loss = -np.fmin(p_loss1, p_loss2).sum()

    clipped_mask = ((ratio > 1+eps) | (ratio < 1-eps)).astype(np.float32)
    clip_frac = clipped_mask.mean()

    loss = policy_loss + 0.001 * entropy_loss + critic_loss

    info = dict(ploss=policy_loss, entr=-entropy_loss, vloss=critic_loss, 
        approx_kl=approx_kl, cf=clip_frac)

    return loss, info

def ppo_loss_batch(p_params, v_params, batch):
    out = jax.vmap(partial(ppo_loss, p_params, v_params))(batch)
    loss, info = jax.tree_map(lambda x: x.mean(), out)
    return loss, info

ppo_loss_grad = jax.jit(jax.value_and_grad(ppo_loss_batch, argnums=[0,1], has_aux=True))

def optim_update_fcn(optim):
    @jax.jit
    def update_step(params, grads, opt_state):
        grads, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(params, grads)
        return params, opt_state
    return update_step

@jax.jit
def ppo_step(p_params, v_params, p_opt_state, v_opt_state, batch):
    (loss, info), (p_grads, v_grads) = ppo_loss_grad(p_params, v_params, batch)
    p_params, p_opt_state = p_update_step(p_params, p_grads, p_opt_state)
    v_params, v_opt_state = v_update_step(v_params, v_grads, v_opt_state)
    return loss, info, p_params, v_params, p_opt_state, v_opt_state

@ray.remote
class Worker:
    def __init__(self, n_steps):
        self.n_steps = n_steps
        self.p_frwd = jax.jit(policy_fcn.apply)
        self.v_frwd = jax.jit(critic_fcn.apply)

        self.buffer = Vector_ReplayBuffer(1e6)
        import pybullet_envs
        self.env = gym.make(env_name)
        # self.env = make_env()
        self.obs = self.env.reset()

    def compute_advantage_targets(self, v_params, rollout):
        (obs, _, r, obs2, done, _) = rollout
    
        batch_v_fcn = jax.vmap(partial(self.v_frwd, v_params)) # need in class bc of this line i.e v_frwd
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

    def rollout(self, p_params, v_params, rng):
        self.buffer.clear()
        
        for _ in range(self.n_steps): # rollout 
            rng, subrng = jax.random.split(rng)
            
            mu, sig = self.p_frwd(p_params, self.obs)
            dist = distrax.MultivariateNormalDiag(mu, sig)
            a = dist.sample(seed=subrng)
            a = np.clip(a, a_low, a_high)
            log_prob = dist.log_prob(a)
            a = onp.array(a)
            
            obs2, r, done, _ = self.env.step(a)

            self.buffer.push((self.obs, a, r, obs2, done, log_prob))
            self.obs = obs2
            if done: 
                self.obs = self.env.reset()

        # update rollout contents 
        rollout = self.buffer.contents()
        advantages, v_target = self.compute_advantage_targets(v_params, rollout)
        (obs, a, r, _, _, log_prob) = rollout
        rollout = (obs, a, log_prob, v_target, advantages)
        
        return rollout

#%%
n_envs = 4

seed = onp.random.randint(1e5) # 90897 works well for pendulum 
gamma = 0.99 
lmbda = 0.95
eps = 0.2
batch_size = 128 
policy_lr = 1e-3
v_lr = 1e-3
max_n_steps = 1e6
n_step_rollout = 2048 #env._max_episode_steps

rng = jax.random.PRNGKey(seed)
onp.random.seed(seed)

obs = env.reset() # dummy input 
p_params = policy_fcn.init(rng, obs) 
v_params = critic_fcn.init(rng, obs) 

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

p_update_step = optim_update_fcn(p_optim)
v_update_step = optim_update_fcn(v_optim)

#%%
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment=f'ppo_multi_{n_envs}_{env_name}_seed={seed}_nrollout={n_step_rollout}')

#%%
workers = [Worker.remote(n_step_rollout) for _ in range(n_envs)]

step_i = 0 
from tqdm import tqdm 
pbar = tqdm(total=max_n_steps)
while step_i < max_n_steps:
    ## rollout
    rng, *subkeys = jax.random.split(rng, 1+n_envs+1) # +1 for eval rollout
    rollouts = ray.get([workers[i].rollout.remote(p_params, v_params, subkeys[i]) for i in range(n_envs)])
    rollout = jax.tree_multimap(lambda *a: np.concatenate(a), *rollouts, is_leaf=lambda node: hasattr(node, 'shape'))

    ## update
    for batch in rollout2batches(rollout, batch_size):
        loss, info, p_params, v_params, p_opt_state, v_opt_state = \
            ppo_step(p_params, v_params, p_opt_state, v_opt_state, batch)
        step_i += 1 
        pbar.update(1)
        writer.add_scalar('loss/loss', loss.item(), step_i)
        for k in info.keys(): 
            writer.add_scalar(f'info/{k}', info[k].item(), step_i)

    reward = eval(p_params, env, subkeys[-1])
    writer.add_scalar('eval/total_reward', reward.item(), step_i)

#%%
#%%
#%%