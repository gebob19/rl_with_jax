#%%
import gym
from numpy.lib.shape_base import column_stack
import torch 
import torch.nn as nn 
import matplotlib.pyplot as plt 
from torch.distributions.categorical import Categorical 
from numpngw import write_apng
import numpy as np 
from absl import flags, app
from tqdm import tqdm 
import wandb
import pprint 
from torch.utils.tensorboard import SummaryWriter

#%%
FLAGS = flags.FLAGS
flags.DEFINE_integer('T', 200, 'max number of steps per rollout')
flags.DEFINE_integer('n_steps', 1000 * 100, '') # 1k episodes -- 100 steps per episode 
flags.DEFINE_integer('batch_size', 64, '')
flags.DEFINE_float('lr', 1e-3, '')
flags.DEFINE_integer('seed', 0, '')
flags.DEFINE_integer('grad_clip_v', -1, '')

flags.DEFINE_string('exp_name', '', '')
flags.DEFINE_string('runs_folder', 'runs/', '')

flags.DEFINE_boolean('random_seed', False, '')
flags.DEFINE_boolean('view_result', False, '')

def main(_):
    T = FLAGS.T 
    n_steps = FLAGS.n_steps 
    batch_size = FLAGS.batch_size 
    lr = FLAGS.lr 
    seed = np.random.randint(low=0, high=100) if FLAGS.random_seed else FLAGS.seed 
    grad_clip_v = FLAGS.grad_clip_v 

    config = {
        'n_steps': n_steps, 
        'batch_size': batch_size,
        'n_timesteps_per_rollout': T, 
        'seed': seed, 
        'lr': lr,
        'grad_clip_value': grad_clip_v, 
    }

    print('Config: ')
    pprint.pprint(config)

    if FLAGS.exp_name:
        writer = SummaryWriter(f'{FLAGS.runs_folder}/{FLAGS.exp_name}')
    else: 
        writer = SummaryWriter()

    for c in config: 
        writer.add_text(c, str(config[c]), 0)

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make('CartPole-v0')

    policy = nn.Sequential(
        nn.Linear(env.observation_space.shape[0], 32),
        nn.ReLU(), 
        nn.Linear(32, 32), 
        nn.ReLU(), 
        nn.Linear(32, env.action_space.n), 
        nn.Softmax(dim=-1),
    )

    optim = torch.optim.SGD(policy.parameters(), lr=lr)
    optim.zero_grad()

    epi_count = 0 
    step_count = 0
    
    pbar = tqdm(total=n_steps)
    while step_count < n_steps: 
        log_action_probs = []
        rewards = []

        obs = env.reset()
        for _ in range(T):
            tobs = torch.from_numpy(obs).float()
            a_space = policy(tobs)
            a_space = Categorical(a_space)
            
            a = a_space.sample()
            prob = a_space.log_prob(a)
            
            entropy = a_space.entropy()
            writer.add_scalar('policy/entropy', entropy.item(), step_count)
            
            obs2, r, done, _ = env.step(a.numpy())
            step_count += 1
            pbar.update(1)

            rewards.append(torch.tensor(r).float())
            log_action_probs.append(prob)

            if done: break
            obs = obs2

        metrics = {
            'reward/min': min(rewards),
            'reward/max': max(rewards),
            'reward/total': sum(rewards),
        }

        # rewards to go 
        gamma = 0.99
        for i in range(len(rewards) - 1)[::-1]:
            rewards[i] = rewards[i] + gamma * rewards[i+1]
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / rewards.std()

        metrics['reward/rescaled_min'] = min(rewards)
        metrics['reward/rescaled_max'] = max(rewards)
        metrics['reward/rescaled_total'] = sum(rewards)

        # record metrics 
        for m in metrics:
            writer.add_scalar(m, metrics[m], epi_count)

        log_action_probs = torch.stack(log_action_probs)
        
        rollout_loss = -(log_action_probs * rewards).sum() # this is key
        rollout_loss.backward()

        if (epi_count+1) % batch_size == 0:

            for i, p in enumerate(policy.parameters()):
                writer.add_histogram(f'w{i}/weight', p.data, epi_count)
                writer.add_histogram(f'w{i}/grad', p.grad, epi_count)

            if grad_clip_v != -1: 
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_v)
                for i, p in enumerate(policy.parameters()):
                    writer.add_histogram(f'w{i}/clipped_grad', p.grad, epi_count)

            optim.step()
            optim.zero_grad()
        
        epi_count += 1 
    pbar.close()

    if FLAGS.view_result:  ## evaluation 
        imgs = []
        eval_reward = 0 
        obs = env.reset()
        while True: 
            img = env.render(mode='rgb_array')
            imgs.append(img)

            tobs = torch.from_numpy(obs).float()
            a_space = policy(tobs)
            a_space = Categorical(a_space)        
            a = a_space.sample()
            
            obs2, r, done, _ = env.step(a.numpy())
            
            eval_reward += r 
            obs = obs2

            if done: break 

        print(f'Total Reward: {eval_reward:.2f} #frames: {len(imgs)}')
        print('writing...')
        write_apng('cartpole.png', imgs, delay=20)

if __name__ == '__main__':
    app.run(main)

# %%
#%%
# %%
# %%
