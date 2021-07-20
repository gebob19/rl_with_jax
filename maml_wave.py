#%%
from functools import partial
import jax 
import jax.numpy as np 
from jax import random, vmap, jit, grad
from jax.experimental import stax, optimizers
from jax.experimental.stax import Dense, Relu
import matplotlib.pyplot as plt 
from tqdm.notebook import tqdm 

#%%
# Use stax to set up network initialization and evaluation functions
net_init, net_apply = stax.serial(
    Dense(40), Relu,
    Dense(40), Relu,
    Dense(1)
)
in_shape = (-1, 1,)
rng = random.PRNGKey(0)
out_shape, params = net_init(rng, in_shape)

#%%
import numpy as onp 
def get_wave(wave_gen, n_samples=100, wave_params=False):
    x = wave_gen(n_samples)
    amp = onp.random.uniform(low=0.1, high=5.0)
    phase = onp.random.uniform(low=0., high=onp.pi)
    wave_data = x, onp.sin(x + phase) * amp

    if wave_params: wave_data = (wave_data, (phase, amp))
    return wave_data
    
def vis_wave_gen(N): # better for visualization 
    x = onp.linspace(-5, 5, N).reshape((N, 1))
    return x

def train_wave_gen(N): # for model training 
    x = onp.random.uniform(low=-5., high=5., size=(N, 1))
    return x 

def mse(params, batch):
    x, y = batch 
    ypred = net_apply(params, x)
    return np.mean((y - ypred)**2)

#%%
batch = get_wave(vis_wave_gen, 100)
predictions = net_apply(params, batch[0]) 
losses = mse(params, batch)

plt.plot(batch[0], predictions, label='prediction')
plt.plot(*batch, label='target')
plt.legend()

#%%
opt_init, opt_update, get_params = optimizers.adam(step_size=1e-2)

@jit
def step(i, opt_state, batch):
    params = get_params(opt_state)
    g = grad(mse)(params, batch)
    return opt_update(i, g, opt_state)

#%%
out_shape, params = net_init(rng, in_shape) # re-init model
opt_state = opt_init(params) # init optim 

batch = get_wave(vis_wave_gen, 100)
for i in range(200):
    opt_state = step(i, opt_state, batch)
params = get_params(opt_state)

xb, yb = batch 
plt.plot(xb, net_apply(params, xb), label='prediction')
plt.plot(xb, yb, label='target')
plt.legend()

# %%
### MAML 
alpha = 0.1

# inner loop -- take one gradient step on the data 
def inner_update(params, batch):
    grads = grad(mse)(params, batch)
    sgd_update = lambda param, grad: param - alpha * grad
    inner_params = jax.tree_multimap(sgd_update, params, grads)
    return inner_params

# outer loop 
def maml_loss(params, train_batch, test_batch):
    task_params = inner_update(params, train_batch)
    loss = mse(task_params, test_batch)
    return loss 

@jit
def maml_step(i, opt_state, train_batch, test_batch):
    params = get_params(opt_state)
    g = grad(maml_loss)(params, train_batch, test_batch)
    return opt_update(i, g, opt_state)

## task extractor 
def get_task(n_train, n_test, wave_params=False):
    if not wave_params: 
        batch = get_wave(train_wave_gen, n_train + n_test)
    else: 
        batch, wparams = get_wave(train_wave_gen, n_train + n_test, wave_params=True)

    # extract train/test elements from batch=(xb, yb) with treemap :)
    train_batch = jax.tree_map(lambda l: l[:n_train], batch, is_leaf=lambda node: hasattr(node, 'shape'))
    test_batch = jax.tree_map(lambda l: l[n_train:], batch, is_leaf=lambda node: hasattr(node, 'shape'))

    task = train_batch, test_batch
    if wave_params: task = (*task, wparams)

    return task 

# %%
opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3)
out_shape, params = net_init(rng, in_shape) # re-init model
opt_state = opt_init(params) # init optim 

for i in tqdm(range(20000)):
    train_batch, test_batch = get_task(20, 1)
    opt_state = maml_step(i, opt_state, train_batch, test_batch)
params = get_params(opt_state)

# %%
train_batch, test_batch, wparams = get_task(20, 1, wave_params=True)

# re-create wave smoother for visualization 
phase, amp = wparams 
x = vis_wave_gen(100)
y = np.sin(x + phase) * amp 
plt.plot(x, y, label='targets')

step_params = params.copy()
for i in range(5): # visualize wave at each grad step 
    ypred = net_apply(step_params, x)
    plt.plot(x, ypred, label=f'step{i}')
    step_params = inner_update(step_params, train_batch)

plt.legend()

# %%
task_batch_size = 5
tasks = [get_task(20, 1) for _ in range(task_batch_size)]
train_batch, test_batch = jax.tree_multimap(lambda *b: np.stack(b), *tasks, is_leaf=lambda node: hasattr(node, 'shape'))

xb, yb = train_batch
for i in range(len(xb)):
    plt.scatter(xb[i], yb[i])

# %%
def batch_maml_loss(params, train_batch, test_batch):
    losses = vmap(partial(maml_loss, params))(train_batch, test_batch)
    loss = losses.mean()
    return loss

@jit
def batch_maml_step(i, opt_state, train_batch, test_batch):
    params = get_params(opt_state)
    g = grad(batch_maml_loss)(params, train_batch, test_batch)
    return opt_update(i, g, opt_state)

# %%
task_batch_size = 4

opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3)
out_shape, params = net_init(rng, in_shape) # re-init model
opt_state = opt_init(params) # init optim 

for i in tqdm(range(20000)):
    # get batch of tasks 
    tasks = [get_task(20, 1) for _ in range(task_batch_size)]
    train_batch, test_batch = jax.tree_multimap(lambda *b: np.stack(b), *tasks, is_leaf=lambda node: hasattr(node, 'shape'))
    # take gradient step over the mean 
    opt_state = batch_maml_step(i, opt_state, train_batch, test_batch)

params = get_params(opt_state)

# %%
train_batch, test_batch, wparams = get_task(20, 1, wave_params=True)

# re-create wave smoother for visualization 
phase, amp = wparams 
x = vis_wave_gen(100)
y = np.sin(x + phase) * amp 
plt.plot(x, y, label='targets')
plt.scatter(*train_batch, label='train')

step_params = params.copy()
for i in range(5): # visualize wave at each grad step 
    ypred = net_apply(step_params, x)
    plt.plot(x, ypred, label=f'step{i}')
    step_params = inner_update(step_params, train_batch)

plt.legend()

# %%
