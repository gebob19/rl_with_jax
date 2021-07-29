#%%
import jax 
import jax.numpy as np 
import numpy as onp 
import distrax 
import optax
import gym 
from functools import partial

#%%
env_name = 'Pendulum-v0'
env = gym.make(env_name)

n_actions = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]

a_high = env.action_space.high[0]
a_low = env.action_space.low[0]

print(f'[LOGGER] a_high: {a_high} a_low: {a_low} n_actions: {n_actions} obs_dim: {obs_dim}')
assert -a_high == a_low

#%%
import haiku as hk

def _policy_fcn(s):
    policy = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
        hk.Linear(2*n_actions), 
    ])
    x = policy(s)
    mu, sig = x[:n_actions], x[n_actions:]
    mu = np.tanh(mu) * a_high
    sig = jax.nn.softplus(sig)
    return mu, sig

# def _policy_fcn(s):
#     log_std = hk.get_parameter("log_std", shape=[n_actions,], init=np.ones)
#     mu = hk.Sequential([
#         hk.Linear(64), jax.nn.relu,
#         hk.Linear(64), jax.nn.relu,
#         hk.Linear(n_actions), np.tanh 
#     ])(s) * a_high
#     sig = np.exp(log_std)
#     return mu, sig

def _critic_fcn(s):
    v = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
        hk.Linear(1), 
    ])(s)
    return v 

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

policy_fcn = hk.transform(_policy_fcn)
policy_fcn = hk.without_apply_rng(policy_fcn)
p_frwd = jax.jit(policy_fcn.apply)

critic_fcn = hk.transform(_critic_fcn)
critic_fcn = hk.without_apply_rng(critic_fcn)
v_frwd = jax.jit(critic_fcn.apply)

# %%
def rollout2batches(rollout, batch_size):
    rollout_len = rollout[0].shape[0]
    n_chunks = rollout_len // batch_size
    # shuffle / de-correlate
    idxs = onp.arange(rollout_len) 
    onp.random.shuffle(idxs)
    rollout = jax.tree_map(lambda x: x[idxs], rollout, is_leaf=lambda x: hasattr(x, 'shape'))
    # batch 
    batched_rollout = jax.tree_map(lambda x: np.array_split(x, n_chunks), rollout, is_leaf=lambda x: hasattr(x, 'shape'))
    for i in range(n_chunks):
        batch = [d[i] for d in batched_rollout]
        yield batch 

@jax.jit
def compute_advantages(v_params, rollout):
    (obs, _, r, obs2, done, _) = rollout
    r = (r - r.mean()) / (r.std() + 1e-10) # normalize
    
    batch_v_fcn = jax.vmap(partial(v_frwd, v_params))
    v_obs = batch_v_fcn(obs)
    v_obs2 = batch_v_fcn(obs2)

    v_target = r + (1 - done) * gamma * v_obs2
    advantages = v_target - v_obs
    advantages = jax.lax.stop_gradient(advantages)
    v_target = jax.lax.stop_gradient(v_target)
    
    # normalize 
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
    return advantages, v_target

@jax.jit
def ppo_loss(p_params, v_params, batch):
    (obs, a, old_log_prob, v_target, advantages) = batch 

    ## critic loss
    batch_v_fcn = jax.vmap(partial(v_frwd, v_params))
    v_obs = batch_v_fcn(obs)
    critic_loss = 0.5 * ((v_obs - v_target) ** 2)

    ## policy losses 
    batch_policy = jax.vmap(partial(p_frwd, p_params))
    mu, sig = batch_policy(obs)
    dist = distrax.MultivariateNormalDiag(mu, sig)
    # entropy 
    entropy_loss = -dist.entropy()[:, None]
    # policy gradient 
    log_probs = dist.log_prob(a)[:, None]
    ratio = np.exp(log_probs - old_log_prob)
    p_loss1 = ratio * advantages
    p_loss2 = np.clip(ratio, 1-eps, 1+eps) * advantages
    policy_loss = -np.fmin(p_loss1, p_loss2)

    loss = policy_loss + 0.001 * entropy_loss + critic_loss
    loss = loss.mean()

    return loss 

def update_step(params, grads, optim, opt_state):
    grads, opt_state = optim.update(grads, opt_state)
    params = optax.apply_updates(params, grads)
    return params, opt_state

@jax.jit
def ppo_step(p_params, v_params, p_opt_state, v_opt_state, batch):
    loss, (p_grads, v_grads) = jax.value_and_grad(ppo_loss, argnums=[0,1])(p_params, v_params, batch)
    p_params, p_opt_state = update_step(p_params, p_grads, p_optim, p_opt_state)
    v_params, v_opt_state = update_step(v_params, v_grads, v_optim, v_opt_state)
    return loss, p_params, v_params, p_opt_state, v_opt_state

#%%
seed = onp.random.randint(1e5)
gamma = 0.99 
eps = 0.2
batch_size = 32 
policy_lr = 1e-3
v_lr = 1e-3

rng = jax.random.PRNGKey(seed)
onp.random.seed(seed)

obs = env.reset() # dummy input 
p_params = policy_fcn.init(rng, obs) 
v_params = critic_fcn.init(rng, obs) 
buffer = Vector_ReplayBuffer(1e6)

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

#%%
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment=f'ppo_{env_name}_seed={seed}')

env_step = 0 
from tqdm import tqdm 
for epi_i in tqdm(range(500)):
    total_reward = 0 
    obs = env.reset()

    buffer.clear()
    while True: # rollout 
        mu, sig = p_frwd(p_params, obs)

        rng, subrng = jax.random.split(rng)
        dist = distrax.MultivariateNormalDiag(mu, sig)
        a = dist.sample(seed=subrng)
        a = np.clip(a, a_low, a_high)
        log_prob = dist.log_prob(a)
        
        writer.add_scalar('policy/entropy', dist.entropy().item(), env_step)
        env_step += 1 

        obs2, r, done, _ = env.step(a)
        total_reward += r

        buffer.push((obs, a, r, obs2, done, log_prob))
        obs = obs2
        if done: break

    writer.add_scalar('rollout/reward', total_reward.item(), epi_i)
    rollout = buffer.contents()
    advantages, v_target = compute_advantages(v_params, rollout)
    # update rollout contents 
    (obs, a, _, _, _, log_prob) = rollout
    log_prob = jax.lax.stop_gradient(log_prob)
    rollout = (obs, a, log_prob, v_target, advantages)

    rollout_length = obs.shape[0]
    writer.add_scalar('rollout/length', rollout_length, epi_i)

    total_loss = 0 
    for batch in rollout2batches(rollout, batch_size):
        loss, p_params, v_params, p_opt_state, v_opt_state = \
            ppo_step(p_params, v_params, p_opt_state, v_opt_state, batch)
        total_loss += loss
    writer.add_scalar('loss/loss', total_loss.item(), epi_i)
    # writer.add_scalar('policy/log_std', p_params['~']['log_std'].item(), epi_i)

# %%