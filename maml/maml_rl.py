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

from utils import normal_log_density, sample_gaussian

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
def eval_policy(params, obs, _):
    a, _ = p_frwd(params, obs)
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
policy_lr = 1e-2
# ppo 
gamma = 0.99 
eps = 0.2 
# maml 
epochs = 500
max_n_steps = 100 
task_batch_size = 40
fast_batch_size = 20
eval_fast_batch_size = 40
alpha = 0.5
# eval 
eval_every = 1

rng = jax.random.PRNGKey(seed)
onp.random.seed(seed)

## model init
obs = np.zeros(env.observation_space.shape)  # dummy input 
p_params = policy_fcn.init(rng, obs) 

## optimizers 
optimizer = lambda lr: optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.scale_by_adam(),
    optax.scale(-lr),
)

p_optim = optimizer(policy_lr)
p_opt_state = p_optim.init(p_params)
p_optim_func = get_optim_fcn(p_optim)

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

        a = jax.lax.stop_gradient(a)
        log_prob = jax.lax.stop_gradient(log_prob)

        obs2, r, done, _ = env.step(a)
        buffer.push((obs, a, r, obs2, done, log_prob))
        
        obs = obs2
        if done: break 

    trajectory = buffer.contents()
    return trajectory 

def post_process_trajectory(trajectory):
    (obs, a, r, _, _, _) = trajectory
    r = discount_cumsum(r, discount=gamma)
    trajectory = (obs, a, r)
    return trajectory

def get_trajectory(env, p_params, rng):
    traj = rollout(env, p_params, rng)
    traj = post_process_trajectory(traj)
    return traj 

@jax.jit
def reinforce_loss(p_params, sample):
    obs, a, r = sample
    mu, sig = p_frwd(p_params, obs)
    log_prob = distrax.MultivariateNormalDiag(mu, sig).log_prob(a)
    loss = -(log_prob * r).sum()
    info = dict()
    return loss, info

def loss_batch(p_params, batch):
    out = jax.vmap(partial(reinforce_loss, p_params))(batch)
    loss, info = jax.tree_map(lambda x: x.mean(), out)
    return loss, info

loss_grad = jax.jit(jax.value_and_grad(loss_batch, has_aux=True))

@jax.jit
def sgd_step(params, grads, alpha):
    sgd_update = lambda param, grad: param - alpha * grad
    return jax.tree_multimap(sgd_update, params, grads)

def maml_inner(p_params, trajectory, alpha, batch_size):
    batch = rollout2batches(trajectory, batch_size, n_batches=1)
    (_, info), p_g = loss_grad(p_params, batch)

    inner_params_p = sgd_step(p_params, p_g, alpha)
    return inner_params_p, info

def maml_loss(params, env, rng):
    subkeys = jax.random.split(rng, 2)
    # meta train
    meta_train_traj = get_trajectory(env, params, subkeys[0])
    inner_params, _ = maml_inner(params, meta_train_traj, alpha, fast_batch_size)
    # meta test
    meta_test_traj = get_trajectory(env, inner_params, subkeys[1])
    meta_test_loss, _ = loss_batch(inner_params, meta_test_traj)
    return meta_test_loss

def maml_eval(env, params, rng, n_steps=1):
    rewards = []
    infos = []
    rng, subkey = jax.random.split(rng, 2)
    reward_0step = eval(params, env, subkey)
    
    rewards.append(reward_0step)

    eval_alpha = alpha
    for _ in range(n_steps):
        rng, *subkeys = jax.random.split(rng, 3)
        meta_train_traj = get_trajectory(env, params, subkeys[0])
        inner_params, info = maml_inner(params, meta_train_traj, eval_alpha, eval_fast_batch_size)

        r = eval(inner_params, env, subkeys[1])

        rewards.append(r)
        infos.append(info)
        params = inner_params
        eval_alpha = alpha / 2 

    return rewards, infos

#%%
env.seed(0)
n_tasks = 1 
task = env.sample_tasks(1)[0] ## only two tasks
assert n_tasks in [1, 2] 
if n_tasks == 1: 
    tasks = [task] * task_batch_size
elif n_tasks == 2: 
    task2 = {'goal': -task['goal'].copy()}
    tasks = [task, task2] * (task_batch_size//2)

for task in tasks[:n_tasks]: 
    env.reset_task(task)
    # log max reward 
    goal = env._task['goal']
    reward = 0 
    step_count = 0 
    obs = env.reset()
    while True: 
        a = goal - obs 
        obs2, r, done, _ = env.step(a)
        reward += r
        step_count += 1 
        if done: break 
        obs = obs2
    print(f'[LOGGER]: MAX_REWARD={reward} IN {step_count} STEPS')
    

# tasks = tasks * (task_batch_size//2)
# tasks = env.sample_tasks(1) * task_batch_size ## only ONE tasks 

#%%
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment=f'maml_{n_tasks}task_test_seed={seed}')

#%%
from tqdm import tqdm 

step_count = 0 
for e in tqdm(range(1, epochs+1)):
    # training 
    # tasks = env.sample_tasks(task_batch_size)

    gradients = []
    mean_loss = 0
    for task_i, task in enumerate(tqdm(tasks)): 
        env.reset_task(task)
        rng, subkey = jax.random.split(rng, 2)
        loss, grads = jax.value_and_grad(maml_loss)(p_params, env, subkey)
        
        # writer.add_scalar(f'loss/task_{task_i}loss', loss.item(), step_count)
        
        # for i, g in enumerate(jax.tree_leaves(grads)): 
        #     name = 'b' if len(g.shape) == 1 else 'w'
        #     writer.add_histogram(f'task_{task_i}_{name}_{i}_grad', onp.array(g), step_count)

        gradients.append(grads)
        mean_loss += loss 
        step_count += 1 
            
    mean_loss /= len(tasks)
    writer.add_scalar(f'loss/mean_task_loss', mean_loss, step_count)

    grads = jax.tree_multimap(lambda *x: np.stack(x).mean(0), *gradients)
    for i, g in enumerate(jax.tree_leaves(grads)): 
        name = 'b' if len(g.shape) == 1 else 'w'
        writer.add_histogram(f'task_{task_i}_{name}_{i}_grad', onp.array(g), step_count)

    p_params, p_opt_state = p_optim_func(p_params, grads, p_opt_state)

    # eval 
    if e % eval_every == 0:
        # eval_task = env.sample_tasks(1)[0]
        eval_tasks = tasks
        task_rewards = []
        for task_i, eval_task in enumerate(eval_tasks):
            env.reset_task(eval_task)

            rng, subkey = jax.random.split(rng, 2)
            rewards, infos = maml_eval(env, p_params, subkey, n_steps=3)
            task_rewards.append(rewards)

            for step_i, r in enumerate(rewards):
                writer.add_scalar(f'task{task_i}/reward_{step_i}step', r, e)

            # for step_i, info in enumerate(infos):
            #     for k in info.keys(): 
            #         writer.add_scalar(f'task{task_i}/{k}_{step_i}step', info[k].item(), e)

        mean_rewards=[]
        for step_i in range(len(task_rewards[0])):
            mean_r = sum([task_rewards[j][step_i] for j in range(len(task_rewards))]) / 2
            writer.add_scalar(f'mean_task/reward_{step_i}step', mean_r, e)
            mean_rewards.append(mean_r)

        # if e == eval_every or rewards[1] > max_reward: 

        # max_reward = mean_rewards[1] # eval on single grad step 
        # save_path = str(model_path/f'params_e{e}_{max_reward:.2f}')
        # print(f'saving model to {save_path}...')
        # with open(save_path, 'wb') as f: 
        #     cloudpickle.dump((p_params, v_params), f)

#%%
#%%
# #%%
# save_path = '/home/brennan/rl_with_jax/models/maml/Navigation2D/params_e16_-7.17'
# with open(save_path, 'rb') as f: 
#     (p_params, v_params) = cloudpickle.load(f)

# #%%
# task = env.sample_tasks(1)[0]
# env.reset_task(task)

# #%%
# rng, subkey = jax.random.split(rng, 2)
# render(p_params, env, subkey, 100)

# #%%
# n_steps = 3 
# eval_alpha = alpha
# params = (p_params, v_params)
# for i in range(1, n_steps+1):
#     rng, *subkeys = jax.random.split(rng, 3)
#     meta_train_traj = get_trajectory(env, params, subkeys[0])
#     inner_params = maml_inner(params, meta_train_traj, eval_alpha, eval_fast_batch_size)

#     rng, subkey = jax.random.split(rng, 2)
#     plt.title(f'step{i}')
#     r = render(inner_params[0], env, subkey, 100)
#     print(f'step{i} : {r}')

#     params = inner_params
#     eval_alpha = alpha / 2
#     break  

# #%%
# for p1, p2 in zip(jax.tree_leaves((p_params, v_params)), jax.tree_leaves(inner_params)):
#     print(np.abs(p1 - p2).sum())

# #%%
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
#         a = eval_policy(p_params, obs)

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