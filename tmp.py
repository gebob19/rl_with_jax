#%%
import ray 
ray.init()

#%%
@ray.remote
class Worker:
    def __init__(self):
        self.i = 0 
    
    def rollout(self, n_steps):
        for _ in range(n_steps):
            self.i += 1 
    
    def get_value(self): return self.i 

#%%
worker = Worker.remote()

#%%
ray.get(worker.rollout.remote(10))

#%%
ray.get(worker.get_value.remote())

#%%


#%%
#%%