#%%
import jax
import jax.numpy as np 
import numpy as onp 
import distrax 
import optax
import gym 
from functools import partial
from env import Navigation2DEnv
# from jax_env import Navigation2DEnvJAX
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

#%%
import haiku as hk
init_final = hk.initializers.RandomUniform(-3e-3, 3e-3)

def _policy_fcn(s):
    log_std = hk.get_parameter("log_std", shape=[n_actions,], init=np.ones)
    mu = hk.Sequential([
        hk.Linear(100), jax.nn.relu,
        hk.Linear(100), jax.nn.relu,
        hk.Linear(n_actions, w_init=init_final), np.tanh 
    ])(s) * a_high
    sig = np.exp(log_std)
    return mu, sig

def _critic_fcn(s):
    v = hk.Sequential([
        hk.Linear(100), jax.nn.relu,
        hk.Linear(100), jax.nn.relu,
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

#%%
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
def eval_policy(params, obs, rng):
    mu, _ = p_frwd(params, obs)
    a = mu
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

def shuffle_rollout(rollout):
    rollout_len = rollout[0].shape[0]
    idxs = onp.arange(rollout_len) 
    onp.random.shuffle(idxs)
    rollout = jax.tree_map(lambda x: x[idxs], rollout, is_leaf=lambda x: hasattr(x, 'shape'))
    return rollout

def rollout2batches(rollout, batch_size, n_batches=1):
    assert n_batches == 1
    rollout_len = rollout[0].shape[0]
    n_chunks = rollout_len // batch_size
    # shuffle / de-correlate
    rollout = shuffle_rollout(rollout)
    if n_chunks == 0: return rollout

    batched_rollout = jax.tree_map(lambda x: np.array_split(x, n_chunks), rollout, is_leaf=lambda x: hasattr(x, 'shape'))
    batch = [d[0] for d in batched_rollout] ## only 1 batch
    return batch

def get_optim_fcn(optim):
    @jax.jit
    def update_step(params, grads, opt_state):
        grads, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(params, grads)
        return params, opt_state
    return update_step

def discount_cumsum(l, discount):
    l = onp.array(l)
    for i in range(len(l) - 1)[::-1]:
        l[i] = l[i] + discount * l[i+1]
    return l 

#%%
### optim, lr and alpha update 
seed = onp.random.randint(1e5) # 0 
policy_lr = 1e-3
v_lr = 1e-3
# ppo 
gamma = 0.99 
lmbda = 0.95
eps = 0.2 
# maml 
epochs = 500
max_n_steps = 100 
task_batch_size = 40
fast_batch_size = 20
eval_fast_batch_size = 40
alpha = 0.1
# eval 
eval_every = 1

rng = jax.random.PRNGKey(seed)
onp.random.seed(seed)

## model init
obs = np.zeros(env.observation_space.shape)  # dummy input 
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

p_optim_func = get_optim_fcn(p_optim)
v_optim_func = get_optim_fcn(v_optim)

model_path = pathlib.Path(f'./models/maml/{env_name}')
model_path.mkdir(exist_ok=True, parents=True)

#%%
buffer = Vector_ReplayBuffer(max_n_steps)

def rollout(env, p_params, rng):
    buffer.clear()
    obs = env.reset()
    for _ in range(max_n_steps): 
        rng, subkey = jax.random.split(rng, 2)
        a, log_prob = policy(p_params, obs, subkey)
        
        # stop the trace! 
        a = jax.lax.stop_gradient(a)
        log_prob = jax.lax.stop_gradient(log_prob)

        obs2, r, done, _ = env.step(a)
        buffer.push((obs, a, r, obs2, done, log_prob))
        
        if done: break 
        obs = obs2

    trajectory = buffer.contents()
    return trajectory 

def advantage_vtarget(v_params, rollout):
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
def post_process_trajectory(v_params, trajectory):
    (obs, a, _, _, _, log_prob) = trajectory
    log_prob = jax.lax.stop_gradient(log_prob)
    advantages, v_target = advantage_vtarget(v_params, trajectory)
    # repackage for ppo_loss
    trajectory = (obs, a, log_prob, v_target, advantages)
    return trajectory

def get_ppo_trajectory(env, params, rng):
    p_params, v_params = params
    traj = rollout(env, p_params, rng)
    traj = post_process_trajectory(v_params, traj)
    return traj 

@jax.jit
def ppo_loss(p_params, v_params, sample):
    (obs, a, old_log_prob, v_target, advantages) = sample 

    ## critic loss
    v_obs = v_frwd(v_params, obs)
    critic_loss = 0.5 * ((v_obs - v_target) ** 2)

    ## policy losses 
    mu, sig = p_frwd(p_params, obs)
    dist = distrax.MultivariateNormalDiag(mu, sig)
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

ppo_loss_grad = jax.jit(jax.grad(ppo_loss_batch, argnums=[0,1]))

@jax.jit
def sgd_step(params, grads, alpha):
    sgd_update = lambda lr: lambda param, grad: param - lr * grad
    return jax.tree_multimap(sgd_update(alpha), params, grads)

def maml_inner(params, trajectory, alpha, batch_size):
    p_params, v_params = params

    batch = rollout2batches(trajectory, batch_size, n_batches=1)
    p_g, v_g = ppo_loss_grad(*params, batch)

    inner_params_p = sgd_step(p_params, p_g, alpha)
    inner_params_v = sgd_step(v_params, v_g, alpha)

    return inner_params_p, inner_params_v

def maml_loss(params, env, rng):
    subkeys = jax.random.split(rng, 2)
    # meta train
    meta_train_traj = get_ppo_trajectory(env, params, subkeys[0])
    inner_params = maml_inner(params, meta_train_traj, alpha, fast_batch_size)
    # meta test
    meta_test_traj = get_ppo_trajectory(env, inner_params, subkeys[1])
    meta_test_loss = ppo_loss_batch(*inner_params, meta_test_traj)
    return meta_test_loss

def maml_eval(env, params, rng, n_steps=1):
    rewards = []
    rng, subkey = jax.random.split(rng, 2)
    reward_0step = rollout(env, params[0], subkey)[2].sum()
    rewards.append(reward_0step)

    eval_alpha = alpha
    for _ in range(n_steps):
        rng, *subkeys = jax.random.split(rng, 3)
        meta_train_traj = get_ppo_trajectory(env, params, subkeys[0])
        inner_params = maml_inner(params, meta_train_traj, eval_alpha, eval_fast_batch_size)

        r = rollout(env, inner_params[0], subkeys[1])[2].sum()
        rewards.append(r)
        params = inner_params
        eval_alpha = alpha / 2 

    return rewards

#%%
task = env.sample_tasks(1)[0]
tasks = [task] * task_batch_size

#%%
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment=f'maml_1task_test_seed={seed}')

#%%
from tqdm import tqdm 

step_count = 0 
for e in tqdm(range(1, epochs+1)):
    # training 
    params = (p_params, v_params)
    # tasks = env.sample_tasks(task_batch_size)

    gradients = []
    for task in tqdm(tasks): 
        env.reset_task(task)
        rng, subkey = jax.random.split(rng, 2)
        loss, grads = jax.value_and_grad(maml_loss)(params, env, subkey)
        
        writer.add_scalar('loss/loss', loss.item(), step_count)
        gradients.append(grads)
        step_count += 1 
        
    grad = jax.tree_multimap(lambda *x: np.stack(x).mean(0), *gradients)

    p_grad, v_grad = grad
    p_params, p_opt_state = p_optim_func(p_params, p_grad, p_opt_state)
    v_params, v_opt_state = v_optim_func(v_params, v_grad, v_opt_state)

    # eval 
    if e % eval_every == 0:
        # eval_task = env.sample_tasks(1)[0]
        eval_task = task 
        env.reset_task(eval_task)

        rng, subkey = jax.random.split(rng, 2)
        rewards = maml_eval(env, (p_params, v_params), subkey, n_steps=3)

        for i, r in enumerate(rewards):
            writer.add_scalar(f'reward/{i}step', r, e)

        if e == eval_every or rewards[1] > max_reward: 
            max_reward = rewards[1] # eval on single grad step 
            save_path = str(model_path/f'params_e{e}_{max_reward:.2f}')
            print(f'saving model to {save_path}...')
            with open(save_path, 'wb') as f: 
                cloudpickle.dump((p_params, v_params), f)


#%%
#%%
# #%%
# save_path = '/home/brennan/rl_with_jax/models/maml/Navigation2D/params_e80_-40.68'
# with open(save_path, 'rb') as f: 
#     (p_params, v_params) = cloudpickle.load(f)

# #%%
# task = env.sample_tasks(1)[0]
# env.reset_task(task)

# #%%
# rng, subkey = jax.random.split(rng, 2)
# render(params[0], env, subkey, 100)

# #%%
# n_steps = 3 
# eval_alpha = alpha
# params = (p_params, v_params)
# for i in range(1, n_steps+1):
#     rng, *subkeys = jax.random.split(rng, 3)
#     meta_train_traj = get_ppo_trajectory(env, params, subkeys[0])
#     inner_params = maml_inner(params, meta_train_traj, eval_alpha)

#     rng, subkey = jax.random.split(rng, 2)
#     plt.title(f'step{i}')
#     r = render(inner_params[0], env, subkey, 100)
#     print(f'step{i} : {r}')

#     params = inner_params
#     eval_alpha = alpha / 2 

# #%%

# #%%
# env.seed(0)
# obs = env.reset()

# plt.scatter(*task['goal'], marker='*')
# plt.scatter(*env._state, color='r')
# xp, yp = obs
# rewards = []
# for _ in range(100):
#     a = np.array([-0.1, 0.1])
#     obs2, r, done, _ = env.step(a)
#     if done: break 
#     x, y = obs2
#     rewards.append(r)

#     plt.plot([xp, x], [yp, y], color='red')
#     xp, yp = obs2
#     obs = obs2

# plt.show()

# #%%
# import matplotlib.pyplot as plt 

# def render(p_params, env, rng, n_steps):
#     env.seed(0)
#     obs = env.reset()

#     plt.scatter(*task['goal'], marker='*')
#     plt.scatter(*env._state, color='r')
#     xp, yp = obs
#     rewards = []
#     for _ in range(n_steps):
#         rng, subkey = jax.random.split(rng, 2)
#         a, _ = policy(p_params, obs, subkey)

#         obs2, r, done, _ = env.step(a)
#         if done: break 
#         x, y = obs2
#         rewards.append(r)

#         plt.plot([xp, x], [yp, y], color='red')
#         xp, yp = obs2
#         obs = obs2

#     plt.show()
#     return sum(rewards)

#     # plt.plot(rewards)
#     # plt.show()

# #%%