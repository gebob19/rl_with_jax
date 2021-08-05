#%%
import jax
import jax.numpy as np 
import numpy as onp 
import distrax 
import optax
import gym 
from functools import partial
import pybullet_envs

# import ray 
# ray.init()

#%%
# env_name = 'Pendulum-v0'
env_name = 'HalfCheetahBulletEnv-v0'
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
    log_std = hk.get_parameter("log_std", shape=[n_actions,], init=np.ones)
    mu = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
        hk.Linear(n_actions), np.tanh 
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
    return a 

def eval(params, env, rng):
    rewards = 0 
    obs = env.reset()
    while True: 
        rng, subkey = jax.random.split(rng, 2)
        a = policy(params, obs, subkey)

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
    # batch 
    batched_rollout = jax.tree_map(lambda x: np.array_split(x, n_chunks), rollout, is_leaf=lambda x: hasattr(x, 'shape'))
    for i in range(n_chunks):
        batch = [d[i] for d in batched_rollout]
        yield batch 

def update_step(params, grads, optim, opt_state):
    grads, opt_state = optim.update(grads, opt_state)
    params = optax.apply_updates(params, grads)
    return params, opt_state

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

@jax.jit
def ppo_step(p_params, v_params, p_opt_state, v_opt_state, p_grads, v_grads):
    p_params, p_opt_state = update_step(p_params, p_grads, p_optim, p_opt_state)
    v_params, v_opt_state = update_step(v_params, v_grads, v_optim, v_opt_state)
    return (p_params, v_params), (p_opt_state, v_opt_state)

class Worker:
    def __init__(self, make_env, n_steps):
        self.n_steps = n_steps

        policy_fcn = hk.transform(_policy_fcn)
        policy_fcn = hk.without_apply_rng(policy_fcn)
        self.p_frwd = jax.jit(policy_fcn.apply)

        critic_fcn = hk.transform(_critic_fcn)
        critic_fcn = hk.without_apply_rng(critic_fcn)
        self.v_frwd = jax.jit(critic_fcn.apply)

        self.buffer = Vector_ReplayBuffer(1e6)
        import pybullet_envs
        # self.env = gym.make(env_name)
        self.env = make_env()
        self.obs = self.env.reset()
        self.eval_env = make_env()

    def eval(self, p_params, rng):
        env = self.eval_env
        obs = env.reset()
        total_reward = 0 
        while True: 
            mu, sig = self.p_frwd(p_params, obs)

            rng, subrng = jax.random.split(rng)
            dist = distrax.MultivariateNormalDiag(mu, sig)
            a = dist.sample(seed=subrng)
            a = np.clip(a, a_low, a_high)
            a = jax.lax.stop_gradient(a) # stop the trace!
            
            obs2, r, done, _ = env.step(a)
            obs = obs2
            total_reward += r 
            if done: break 
        
        return total_reward

    def compute_advantages(self, v_params, rollout):
        (obs, _, r, obs2, done, _) = rollout
        r = (r - r.mean()) / (r.std() + 1e-10) # normalize
        
        batch_v_fcn = jax.vmap(partial(self.v_frwd, v_params))
        v_obs = batch_v_fcn(obs)
        v_obs2 = batch_v_fcn(obs2)

        v_target = r + (1 - done) * 0.99 * v_obs2
        advantages = v_target - v_obs
        advantages = jax.lax.stop_gradient(advantages)
        v_target = jax.lax.stop_gradient(v_target)
        
        # normalize 
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        return advantages, v_target

    def rollout(self, p_params, v_params, rng):
        self.buffer.clear()
        
        for _ in range(self.n_steps): # rollout 
            mu, sig = self.p_frwd(p_params, self.obs)

            rng, subrng = jax.random.split(rng)
            dist = distrax.MultivariateNormalDiag(mu, sig)
            a = dist.sample(seed=subrng)
            a = np.clip(a, a_low, a_high)
            log_prob = dist.log_prob(a)
            
            # stop the trace! 
            a = jax.lax.stop_gradient(a)
            log_prob = jax.lax.stop_gradient(log_prob)
            
            obs2, r, done, _ = self.env.step(a)

            self.buffer.push((self.obs, a, r, obs2, done, log_prob))
            self.obs = obs2
            if done: 
                self.obs = self.env.reset()

        # update rollout contents 
        rollout = self.buffer.contents()
        advantages, v_target = self.compute_advantages(v_params, rollout)
        (obs, a, r, _, _, log_prob) = rollout
        rollout = (obs, a, log_prob, v_target, advantages)
        
        return rollout

@jax.jit
def maml_inner(params, rollout):
    p_params, v_params = params
    p_g, v_g = jax.grad(ppo_loss, argnums=[0,1])(*params, rollout)

    sgd_update = lambda lr: lambda param, grad: param - lr * grad
    inner_params_p = jax.tree_multimap(sgd_update(policy_lr), p_params, p_g)
    inner_params_v = jax.tree_multimap(sgd_update(v_lr), v_params, v_g)

    return inner_params_p, inner_params_v

def maml_outer(params, task_idx, rng):    
    subkeys = jax.random.split(rng, 2) 
    # inner update / meta train
    rollout = workers[task_idx].rollout(*params, subkeys[0])
    inner_params = maml_inner(params, rollout)

    # meta test
    eval_rollout = workers[task_idx].rollout(*inner_params, subkeys[-1])
    loss = ppo_loss(*inner_params, eval_rollout)
    return loss 

def batch_maml_loss(params, rng):
    task_idxs = onp.random.choice(onp.arange(len(tasks)), size=(task_batch_size,))
    subkeys = jax.random.split(rng, task_batch_size) 
    loss = 0 
    for task_idx, subkey in zip(task_idxs, subkeys):
        loss += maml_outer(params, task_idx, subkey)
    loss /= task_batch_size
    return loss 

## task defn
env_name = 'HalfCheetahBulletEnv-v0'
def task_forwards():
    env = gym.make(env_name)
    return env 

def task_backwards():
    env = gym.make(env_name)
    env.unwrapped.robot.walk_target_x = -1000.
    return env 

#%%
seed = onp.random.randint(1e5)
gamma = 0.99 
eps = 0.2
task_batch_size = 2
policy_lr = 1e-3
v_lr = 1e-3
max_n_steps = 1e6
n_step_rollout = 200 #env._max_episode_steps
n_steps_eval = 10000

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

#%%
tasks = [task_forwards, task_backwards]
workers = [Worker(task_fcn, n_step_rollout) for task_fcn in tasks]

#%%
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment=f'maml_test')

#%%
step_i = 0 

# pack 
params = (p_params, v_params)
opt_states = (p_opt_state, v_opt_state)

from tqdm import tqdm 
pbar = tqdm(total=max_n_steps)
while step_i < max_n_steps:
    # update 
    rng, subkey = jax.random.split(rng, 2) 
    loss, g_params = jax.value_and_grad(batch_maml_loss)(params, subkey)
    params, opt_states = ppo_step(*params, *opt_states, *g_params)

    writer.add_scalar('loss/loss', loss.item(), step_i)
    step_i += 1 

    # eval 
    if step_i % n_steps_eval == 0: 
        print('evaluating...')
        task_ids = np.arange(len(tasks))
        for task_idx in task_ids: 
            rng, sk1, sk2, sk3 = jax.random.split(rng, 4) 

            # eval without any step
            total_reward_step0 = workers[task_idx].eval(params[0], sk3)

            # step 
            rollout = workers[task_idx].rollout(*params, sk1)
            inner_params = maml_inner(params, rollout)
            # eval with one step 
            total_reward_step1 = workers[task_idx].eval(inner_params[0], sk2)

            writer.add_scalar(f'tasks/task{task_idx}_rstep0', total_reward_step0, step_i)
            writer.add_scalar(f'tasks/task{task_idx}_rstep1', total_reward_step1, step_i)

#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%