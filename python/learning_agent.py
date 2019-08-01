from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import torch.multiprocessing as mp
import multiprocessing
import numpy as np
from rl_agent.actor_critic import RL_Agent
import deepmind_lab
from rl_agent.model import Model, Adv_Model
from rl_agent.utils import *
import threading


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--length', type=int, default=200000,
                    help='Number of steps to run the agent')
parser.add_argument('--width', type=int, default=84,
                    help='Horizontal size of the observations')
parser.add_argument('--height', type=int, default=84,
                    help='Vertical size of the observations')
parser.add_argument('--fps', type=int, default=60,
                    help='Number of frames per second')
parser.add_argument('--runfiles_path', type=str, default=None,
                    help='Set the runfiles path to find DeepMind Lab data')
parser.add_argument('--level_script', type=str, default='contributed/dmlab30/language_select_described_object',
                    help='The environment level script to load')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--tau', type=int, default=0.99, help='Training hyperparameter')
parser.add_argument('--gamma', type=int, default=0.99, help='Discounted factor')
parser.add_argument('--clip-grad-norm', type=int, default=100, help='Clip gradient')
parser.add_argument('--num-episodes', type=int, default=90000, help='Number of training episodes')
parser.add_argument('--num-workers', type=int, default=32, help = 'Number of workers')
parser.add_argument('--prior', type=bool, default=False, help = 'Train prior')
parser.add_argument('--posterior', type=bool, default=False, help = 'Train posterior')
args = parser.parse_args()
if args.runfiles_path:
    deepmind_lab.set_runfiles_path(args.runfiles_path)


manager = multiprocessing.Manager() 
reward_buffer = manager.list()
loss_buffer = manager.list()
def train(rank, args, shared_model, reward_buffer, loss_buffer, shared_optimizer=None):
    env = deepmind_lab.Lab( 
    args.level_script, ['RGB_INTERLEAVED', 'INSTR'],
    config={
        'fps': str(args.fps),
        'width': str(args.width),
        'height': str(args.height),
        'mixerSeed': str(args.seed+rank)
    })
    env.reset()
    split = args.num_workers//3
    if rank < split:
        agent = RL_Agent(env, ACTIONS, args, shared_model, shared_optimizer, device ='cuda:0', train_prior = args.prior , train_posterior = args.posterior)
    elif rank < split*2:
        agent = RL_Agent(env, ACTIONS, args, shared_model, shared_optimizer, device ='cuda:1', train_prior = args.prior , train_posterior = args.posterior)
    else:
        agent = RL_Agent(env, ACTIONS, args, shared_model, shared_optimizer, device = 'cuda:2', train_prior = args.prior , train_posterior = args.posterior)
    while len(reward_buffer)<args.num_episodes:
        loss, reward = agent.train()
        reward_buffer.append(reward)
        loss_buffer.append(loss)
        print('Episode {}, loss {}, reward {}'.format(len(reward_buffer),loss,reward))
        
# Start the Reinforcement Learning agent
#shared_model = Adv_Model(len(ACTIONS))
shared_model = Model(len(ACTIONS))
shared_model.load_state_dict(torch.load('/home/km/working_dir_SF/test_model3.pth'))
shared_model.share_memory()
#shared_optimizer = SharedRMSprop(shared_model.parameters(), lr=0.0001, eps = 0.1, weight_decay = 0.99)
#shared_optimizer.share_memory()
# Train the agent
processes = []
'''
for rank in range(args.num_workers):
    print('Build actor {}'.format(rank))
    processes.append(threading.Thread(target=train, args=(rank, args, shared_model, shared_optimizer)))
   
for p in processes:
    p.start()
'''

for rank in range(0, args.num_workers):
        print('Build actor {}'.format(rank))
        p = mp.Process(target=train, args=(rank, args, shared_model, reward_buffer, loss_buffer))
        p.start()
        processes.append(p)
for p in processes:
        p.join()
torch.save(shared_model.state_dict(), '/home/km/working_dir_SF/test_model4.pth')
np.save('/home/km/working_dir_SF/reward_90000.npy', np.array(reward_buffer))
np.save('/home/km/working_dir_SF/loss_90000.npy', np.array(loss_buffer))