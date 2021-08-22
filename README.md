# RL Implementations in JAX 

Single-file implementations focused on clarity rather than proper code standards :)

| Algo      | Path       | Discrete Actions | Continuous Actions | Multi-CPU  | Other                                                              |
|-----------|------------|------------------|--------------------|------------|--------------------------------------------------------------------|
| TRPO      | trpo/      | trpo.py          | cont.py            |            |                                                                    |
| PPO       | ppo/       | ppo_disc.py      | ppo.py             | *_multi.py |                                                                    |
| MAML      | maml/      |                  |                    |            | *SineWave* = maml_wave.py                                            |
| DQN       |            | dqn.py           |                    |            |                                                                    |
| REINFORCE | reinforce/ | reinforce_jax.py | reinforce_cont.py  |            | *Pytorch* = policy_grad.py <br/> *Time Comparison* = reinforce_torchVSjax.py |
| DDPG      | ddpg/      |                  | ddpg_jax.py        |            | *TD3_DDPG* = ddpg_td3.py                                             |
| A2C       | a2c/       | a2c.py           |                    | *_multi.py |                                                                    |

