#%%
import jax
import jax.numpy as np 
import numpy as onp 
import distrax 
import optax
import gym 
from functools import partial

from jax.config import config
config.update("jax_enable_x64", True) 
config.update("jax_debug_nans", True) # break on nans

#%%
env_name = 'CartPole-v0'
env = gym.make(env_name)

# from tmp import Navigation2DEnv_Disc
# env_name = 'Navigation2D'
# def make_env(init_task=False):
#     env = Navigation2DEnv_Disc(max_n_steps=200)
    
#     if init_task: 
#         env.seed(0)
#         task = env.sample_tasks(1)[0]
#         print(f'[LOGGER]: task = {task}')
#         env.reset_task(task)

#         # log max reward 
#         goal = env._task['goal']
#         x, y = goal
#         n_right = x / 0.1
#         n_up = y / 0.1
#         action_seq = []
#         for _ in range(int(abs(n_right))): action_seq.append(3 if n_right > 0 else 2)
#         for _ in range(int(abs(n_up))): action_seq.append(0 if n_up > 0 else 1)

#         reward = 0 
#         step_count = 0 
#         env.reset()
#         for a in action_seq: 
#             _, r, done, _ = env.step(a)
#             reward += r
#             step_count += 1 
#             if done: break 
#         assert done 
#         print(f'[LOGGER]: MAX_REWARD={reward} IN {step_count} STEPS')
#     return env 

# env = make_env(init_task=True)

n_actions = env.action_space.n 
obs_dim = env.observation_space.shape[0]
print(f'[LOGGER] n_actions: {n_actions} obs_dim: {obs_dim}')

#%%
import haiku as hk 
init_final = hk.initializers.RandomUniform(-3e-3, 3e-3)

def _policy_fcn(obs):
    a_probs = hk.Sequential([
        hk.Linear(32), jax.nn.relu,
        hk.Linear(32), jax.nn.relu,
        hk.Linear(n_actions, w_init=init_final), jax.nn.softmax
    ])(obs)
    return a_probs 

policy_fcn = hk.transform(_policy_fcn)
policy_fcn = hk.without_apply_rng(policy_fcn)
p_frwd = jax.jit(policy_fcn.apply)

@jax.jit
def update_step(params, grads, opt_state):
    grads, opt_state = p_optim.update(grads, opt_state)
    params = optax.apply_updates(params, grads)
    return params, opt_state

def reward2go(r, gamma=0.99):
    for i in range(len(r) - 1)[::-1]:
        r[i] = r[i] + gamma * r[i+1]
    r = (r - r.mean()) / (r.std() + 1e-8)
    return r 

@jax.jit
def policy(p_params, obs, rng):
    a_probs = p_frwd(p_params, obs)
    dist = distrax.Categorical(probs=a_probs)
    a = dist.sample(seed=rng)
    entropy = dist.entropy()
    return a, entropy

def rollout(p_params, rng):
    global step_count
    
    observ, action, rew = [], [], []
    obs = env.reset()
    while True: 
        rng, subkey = jax.random.split(rng, 2) 
        a, entropy = policy(p_params, obs, subkey)
        a = a.item()

        writer.add_scalar('policy/entropy', entropy.item(), step_count)

        obs2, r, done, _ = env.step(a)
        step_count += 1
        pbar.update(1)

        observ.append(obs)
        action.append(a)
        rew.append(r)

        if done: break
        obs = obs2

    obs = np.stack(observ)
    a = np.stack(action)
    r = onp.stack(rew) # 

    return obs, a, r 

def reinforce_loss(p_params, obs, a, r):
    a_probs = p_frwd(p_params, obs)
    log_prob = distrax.Categorical(probs=a_probs).log_prob(a.astype(int))
    loss = -(log_prob * r).sum()
    return loss 

from functools import partial
def batch_reinforce_loss(params, batch):
    return jax.vmap(partial(reinforce_loss, params))(*batch).sum()

# https://github.com/rll/rllab/blob/master/rllab/baselines/linear_feature_baseline.py
def v_features(obs):
    o = np.clip(obs, -10, 10)
    l = len(o)
    al = np.arange(l).reshape(-1, 1) / 100.0
    return np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)

def v_fit(obs_l, rew_l, feature_fcn=v_features, reg_coeff=1e-5):
    featmat = np.concatenate([feature_fcn(obs) for obs in obs_l])
    r = np.concatenate(rew_l)
    for _ in range(5):
        # solve argmin_x (F x = R)
        _coeffs = np.linalg.lstsq(
            featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
            featmat.T.dot(r)
        )[0]
        if not np.any(np.isnan(_coeffs)):
            return _coeffs, 0 # succ
        reg_coeff *= 10
    return np.zeros_like(_coeffs), 1 # err 

# %%
seed = onp.random.randint(1e5)
policy_lr = 1e-3
batch_size = 32
max_n_steps = 100000

rng = jax.random.PRNGKey(seed)
onp.random.seed(seed)

obs = env.reset() # dummy input 
p_params = policy_fcn.init(rng, obs) 

## optimizers 
p_optim = optax.sgd(policy_lr)
p_opt_state = p_optim.init(p_params)

# %%
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment=f'reinforce_LinearBaseline_{env_name}_seed={seed}')

# %%
from tqdm import tqdm 

step_count = 0
epi_i = 0 

pbar = tqdm(total=max_n_steps)
loss_grad_fcn = jax.jit(jax.value_and_grad(batch_reinforce_loss))

while step_count < max_n_steps: 
    obs_l, act_l, rew_l = [], [], []
    for _ in range(batch_size):
        rng, subkey = jax.random.split(rng, 2) 
        obs, a, r = rollout(p_params, subkey)
        writer.add_scalar('rollout/reward', r.sum().item(), epi_i)

        obs_l.append(obs)
        act_l.append(a)
        rew_l.append(r)
        epi_i += 1

    W = v_fit(obs_l, [reward2go(r) for r in rew_l])[0]

    gradients = []
    for i in range(batch_size):
        obs, a, r = obs_l[i], act_l[i], rew_l[i]
        r = r - (v_features(obs) @ W) # subtract baseline 

        loss, grad = loss_grad_fcn(p_params, (obs, a, r))
        writer.add_scalar('loss/loss', loss.item(), step_count)
        gradients.append(grad)

    grads = jax.tree_multimap(lambda *x: np.stack(x).sum(0), *gradients)
    p_params, p_opt_state = update_step(p_params, grads, p_opt_state)

    for i, g in enumerate(jax.tree_leaves(grads)): 
        name = 'b' if len(g.shape) == 1 else 'w'
        writer.add_histogram(f'{name}_{i}_grad', onp.array(g), epi_i)
