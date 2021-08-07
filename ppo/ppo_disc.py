#%%
import jax 
import jax.numpy as np 
import numpy as onp 
import distrax 
import optax
import gym 
from functools import partial
import cloudpickle

env_name = 'CartPole-v0'
env = gym.make(env_name)

n_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]

print(f'[LOGGER] n_actions: {n_actions} obs_dim: {obs_dim}')

#%%
import haiku as hk
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
        hk.Linear(1, w_init=init_final), 
    ])(s)
    return v 

policy_fcn = hk.transform(_policy_fcn)
policy_fcn = hk.without_apply_rng(policy_fcn)
p_frwd = jax.jit(policy_fcn.apply)

critic_fcn = hk.transform(_critic_fcn)
critic_fcn = hk.without_apply_rng(critic_fcn)
v_frwd = jax.jit(critic_fcn.apply)

# %%
@jax.jit 
def policy(params, obs, rng):
    pi = p_frwd(params, obs)
    dist = distrax.Categorical(probs=pi)
    a = dist.sample(seed=rng)
    log_prob = dist.log_prob(a)
    return a, log_prob

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

@jax.jit
def ppo_loss(p_params, v_params, sample):
    (obs, a, old_log_prob, v_target, advantages) = sample 

    ## critic loss
    v_obs = v_frwd(v_params, obs)
    critic_loss = 0.5 * ((v_obs - v_target) ** 2)

    ## policy losses 
    pi = p_frwd(p_params, obs)
    dist = distrax.Categorical(probs=pi)
    # entropy 
    entropy_loss = -dist.entropy()
    # policy gradient 
    log_prob = dist.log_prob(a)

    ratio = np.exp(log_prob - old_log_prob)
    p_loss1 = ratio * advantages
    p_loss2 = np.clip(ratio, 1-eps, 1+eps) * advantages
    policy_loss = -np.fmin(p_loss1, p_loss2)

    loss = policy_loss + 0.001 * entropy_loss + critic_loss

    return loss.sum()

def ppo_loss_batch(p_params, v_params, batch):
    return jax.vmap(partial(ppo_loss, p_params, v_params))(batch).mean()

ppo_loss_grad = jax.jit(jax.value_and_grad(ppo_loss_batch, argnums=[0,1]))

def update_step(params, grads, optim, opt_state):
    grads, opt_state = optim.update(grads, opt_state)
    params = optax.apply_updates(params, grads)
    return params, opt_state

@jax.jit
def ppo_step(p_params, v_params, p_opt_state, v_opt_state, batch):
    loss, (p_grads, v_grads) = ppo_loss_grad(p_params, v_params, batch)
    p_params, p_opt_state = update_step(p_params, p_grads, p_optim, p_opt_state)
    v_params, v_opt_state = update_step(v_params, v_grads, v_optim, v_opt_state)
    return loss, p_params, v_params, p_opt_state, v_opt_state

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
        
        import time 
        s = time.time()
        for _ in range(self.n_steps): # rollout 
            rng, subrng = jax.random.split(rng)
            a, log_prob = policy(p_params, self.obs, subrng)
            a = a.item()

            obs2, r, done, _ = self.env.step(a)

            self.buffer.push((self.obs, a, r, obs2, done, log_prob))
            self.obs = obs2
            if done: 
                self.obs = self.env.reset()

        print(f'rollout: {time.time() - s}')

        # update rollout contents 
        rollout = self.buffer.contents()
        advantages, v_target = compute_advantage_targets(v_params, rollout)
        (obs, a, r, _, _, log_prob) = rollout
        log_prob = jax.lax.stop_gradient(log_prob)
        rollout = (obs, a, log_prob, v_target, advantages)
        
        return rollout

#%%
seed = onp.random.randint(1e5)

batch_size = 32 
policy_lr = 1e-3
v_lr = 1e-3
gamma = 0.99 
lmbda = 0.95
eps = 0.2
max_n_steps = 1e6
n_step_rollout = 200 #env._max_episode_steps

rng = jax.random.PRNGKey(seed)
onp.random.seed(seed)

obs = env.reset() # dummy input 
p_params = policy_fcn.init(rng, obs) 
v_params = critic_fcn.init(rng, obs) 

worker = Worker(n_step_rollout)

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

import pathlib 
model_path = pathlib.Path(f'./models/ppo/{env_name}')
model_path.mkdir(exist_ok=True, parents=True)

#%%
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment=f'ppo_{env_name}_seed={seed}')

#%%
epi_i = 0 
step_i = 0 
from tqdm import tqdm 
pbar = tqdm(total=max_n_steps)
while step_i < max_n_steps: 
    rng, subkey = jax.random.split(rng, 2) 
    rollout = worker.rollout(p_params, v_params, subkey)

    for batch in rollout2batches(rollout, batch_size):
        loss, p_params, v_params, p_opt_state, v_opt_state = \
            ppo_step(p_params, v_params, p_opt_state, v_opt_state, batch)
        step_i += 1 
        pbar.update(1)
        writer.add_scalar('loss/loss', loss.item(), step_i)

    rng, subrng = jax.random.split(rng)
    reward = eval(p_params, env, subrng)
    writer.add_scalar('eval/total_reward', reward, step_i)

    if epi_i == 0 or reward > max_reward: 
        max_reward = reward
        with open(str(model_path/f'params_{max_reward:.2f}'), 'wb') as f: 
            cloudpickle.dump((p_params, v_params), f)
    
    epi_i += 1

# %%
# %%
# %%