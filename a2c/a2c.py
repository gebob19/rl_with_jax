#%%
import jax 
import jax.numpy as np 
import numpy as onp 
import matplotlib.pyplot as plt 
import gym
import haiku as hk 
import random 
import optax

#%%
env = gym.make('CartPole-v0')

n_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]

#%%
init_final = hk.initializers.RandomUniform(-3e-3, 3e-3)

def _policy_value(obs):
    pi = hk.Sequential([
        hk.Linear(64), jax.nn.relu, 
        hk.Linear(64), jax.nn.relu,
        hk.Linear(n_actions, w_init=init_final), jax.nn.softmax
    ])(obs)

    v = hk.Sequential([
        hk.Linear(64), jax.nn.relu, 
        hk.Linear(64), jax.nn.relu,
        hk.Linear(1, w_init=init_final),
    ])(obs)
    return pi, v

policy_value = hk.transform(_policy_value)
policy_value = hk.without_apply_rng(policy_value)
pv_frwd = jax.jit(policy_value.apply) # forward fcn

#%%
seed = onp.random.randint(1e5) # seed=81705 works 
print(f'[LOGGING] seed={seed}')
rng = jax.random.PRNGKey(seed)
onp.random.seed(seed)
random.seed(seed)

#%%
class Categorical: # similar to pytorch categorical
    def __init__(self, probs):
        self.probs = probs
    def sample(self): 
        # https://stackoverflow.com/questions/46539431/np-random-choice-probabilities-do-not-sum-to-1
        p = onp.asanyarray(self.probs)
        p = p / p.sum()
        a = onp.random.choice(onp.arange(len(self.probs)), p=p)
        return a 
    def log_prob(self, i): return np.log(self.probs[i])
    def entropy(self): return -(self.probs * np.log(self.probs)).sum()

def discount_cumsum(l, discount):
    l = onp.array(l)
    for i in range(len(l) - 1)[::-1]:
        l[i] = l[i] + discount * l[i+1]
    return l 

global_env_count = 0
def rollout_v(step_i, params, env, max_n_steps=200):
    global global_env_count

    obs = env.reset()
    # obs, obs2 + a, r, done, 
    v_buffer = onp.zeros((max_n_steps, 2 * obs_dim + 3))

    for i in range(max_n_steps):
        a_probs, v_s = pv_frwd(params, obs)
        a_dist = Categorical(a_probs)
        a = a_dist.sample()

        entropy = a_dist.entropy().item()
        writer.add_scalar('policy/entropy', entropy, global_env_count)
        writer.add_scalar('policy/value', v_s.item(), global_env_count)

        obs2, r, done, _ = env.step(a)        
        v_buffer[i] = onp.array([*obs, a, r, *obs2, float(done)])

        global_env_count += 1 
        obs = obs2 
        if done: break 

    v_buffer = v_buffer[:i+1]
    obs, a, r, obs2, done = onp.split(v_buffer, [obs_dim, obs_dim+1, obs_dim+2, obs_dim*2+2], axis=-1)
    writer.add_scalar('rollout/total_reward', r.sum(), step_i)

    r = discount_cumsum(r, discount=0.99)

    return obs, a, r, obs2, done

from functools import partial

# obs, a, r, obs2, done
def policy_loss(params, obs, a, r):
    a_probs, v_s = pv_frwd(params, obs)
    a_dist = Categorical(a_probs)

    log_prob = a_dist.log_prob(a.astype(np.int32))
    advantage = jax.lax.stop_gradient(r - v_s)
    policy_loss = -(log_prob * advantage).sum()
    
    entropy_loss = -0.001 * a_dist.entropy()
    return policy_loss + entropy_loss

def critic_loss(params, obs, r):
    _, v_s = pv_frwd(params, obs)
    return ((v_s - r) ** 2).sum()

def a2c_loss(params, sample):
    obs, a, r, _, _ = sample
    ploss = policy_loss(params, obs, a, r)
    vloss = critic_loss(params, obs, r)
    loss = ploss + 0.25 * vloss
    return loss, ploss, vloss
    
def batch_a2c_loss(params, samples):
    loss, ploss, vloss = jax.vmap(partial(a2c_loss, params))(samples)
    return loss.mean(), (ploss.mean(), vloss.mean())

@jax.jit
def a2c_step(samples, params, opt_state):
    (loss, (ploss, vloss)), grad = jax.value_and_grad(batch_a2c_loss, has_aux=True)(params, samples)
    grad, opt_state = optim.update(grad, opt_state)
    params = optax.apply_updates(params, grad)
    return loss, ploss, vloss, opt_state, params, grad

#%%
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment='a2c_test')

#%%
obs = env.reset() # dummy input 
a = np.zeros(env.action_space.shape)
params = policy_value.init(rng, obs) 

optim = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.scale_by_adam(),
    optax.scale(-1e-3),
)

opt_state = optim.init(params)

# from tqdm.notebook import tqdm 
from tqdm import tqdm 

n_episodes = 1000
for step_i in tqdm(range(n_episodes)):

    samples = rollout_v(step_i, params, env)
    loss, ploss, vloss, opt_state, params, grads = a2c_step(samples, params, opt_state)

    writer.add_scalar('loss/loss', loss.item(), step_i)
    writer.add_scalar('loss/policy', ploss.item(), step_i)
    writer.add_scalar('loss/critic', vloss.item(), step_i)

    # for i, g in enumerate(jax.tree_leaves(grads)): 
    #     name = 'b' if len(g.shape) == 1 else 'w'
    #     writer.add_histogram(f'{name}_{i}_grad', onp.array(g), step_i)

# #%%
# #%%