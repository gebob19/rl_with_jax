#%%
import jax 
import jax.numpy as np 

#%%
def f(w, x, y):
    return np.exp(w * x - y)

#%%
jax.grad(f)(np.ones(()),np.ones(()), np.ones(()))

#%%
#%%
#%%