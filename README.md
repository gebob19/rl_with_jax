# RL Implementations in JAX 

Single-file implementations focused on clarity rather than proper code standards :)

DQN: `qlearn.py`

REINFORCE: `reinforce/`
- `policy_grad.py` = Pytorch  
- `reinforce_cont.py` = JAX with *continious* actions
- `reinforce_jax.py` = JAX with *discrete* actions 
- `reinforce_torchVSjax.py` = Time Comparison between Pytorch & JAX (torch = faster)

PPO: `ppo/`
- `ppo_disc.py`/`ppo_multi_disc.py` = JAX with *discrete* actions (single & multiprocessing)
- `ppo.py`/`ppo_multi.py` = JAX with *continious* actions (single & multiprocessing)

MAML: `maml/`
- `maml_wave.py` = JAX on sin waves 

DDPG: `ddpg/`
- `ddpg_jax.py` = JAX with continious actions
- `ddpg_td3.py` = DDPG with params from TD3 paper (better than DDPG)

A2C: `a2c/`
- `a2c.py`/`a2c_multi.py` = JAX with single & multiprocessing 
Note: A2C doesn't work that great -- should use PPO 
