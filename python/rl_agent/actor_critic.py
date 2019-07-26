import torch
import random
import numpy as np
import datetime
import time
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.autograd import Variable
from collections import namedtuple
from rl_agent.model import Model
from rl_agent.utils import *



class RL_Agent(object):
    def __init__(self, env, actions, args, shared_model, optimizer=None, device = 'cpu'):
        self.ACTIONS = actions
        self.shared_model = shared_model
        self.model = Model(len(self.ACTIONS))
        self.model.to(device)
        self.device = device
        if optimizer is None:
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.0001, eps = 0.1, weight_decay = 0.99)
        else:
            self.optimizer = optimizer
        
        self.memory = ReplayMemory(300)
        self.args = args
        self.env = env
        
    def sync_to(self):
        for param, shared_param in zip(self.model.parameters(),self.shared_model.parameters()):
            if shared_param.grad is not None:
                return
            if param.grad is not None:
                shared_param._grad = param.grad.cpu()
        
    def sync_from(self):
        self.model.load_state_dict(self.shared_model.state_dict())
        
    def optimize_model(self, values, log_probs, rewards, entropies):
        R = values[-1].data
        '''
        ########################
        add .data to varaiable R
        ########################
        '''
        gae = torch.zeros(1, 1).type(torch.FloatTensor).to(self.device)
        # Base A3C Loss
        policy_loss, value_loss = 0, 0

        # Performing update
        for i in reversed(range(len(rewards))):
            # Value function loss
            R = self.args.gamma * R + rewards[i]
            value_loss = value_loss + 0.5 * (R - values[i]).pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + self.args.gamma * \
                    values[i + 1] - values[i]
            gae = gae * self.args.gamma * self.args.tau + delta_t

            # Computing policy loss
            policy_loss = policy_loss - \
                log_probs[i] * gae.data - 0.01 * entropies[i]
            '''
            ################
            add .data to gae
            ################
            '''
                      

        # Auxiliary loss
        language_prediction_loss = 0 
        tae_loss = 0
        reward_prediction_loss = 0
        value_replay_loss = 0

        # Non-skewed sampling from experience buffer
        if len(rewards)>=15:
            auxiliary_sample = self.memory.sample(11)
            auxiliary_batch = Transition(*zip(*auxiliary_sample))

            # Language Prediction Loss
            # TODO #

            # TAE Loss
            visual_input = auxiliary_batch.state[:10]
            visual_input = torch.cat([t.visual for t in visual_input], 0)

            visual_target = auxiliary_batch.state[1:11]
            visual_target = torch.cat([t.visual for t in visual_target], 0)

            action_logit = torch.cat(auxiliary_batch.action_logit[:10], 0)

            tae_output = self.model.tAE(visual_input, action_logit)
            tae_loss = torch.sum((tae_output - visual_target).pow(2))

            # Skewed-Sampling from experience buffer # TODO
            skewed_sample = self.memory.sample_rp(13)  # memory.skewed_sample(13)
            skewed_batch = Transition(*zip(*skewed_sample))

            # Reward Prediction loss
            batch_rp_input = []
            batch_rp_output = []

            for i in range(10):
                rp_input = skewed_batch.state[i : i+3]
                rp_output = skewed_batch.reward[i+3]

                batch_rp_input.append(rp_input)
                batch_rp_output.append(rp_output)

            rp_predicted = self.model.reward_predictor(batch_rp_input)
            reward_prediction_loss = \
                            torch.sum((rp_predicted - Variable(torch.FloatTensor(batch_rp_output)).to(self.device)).pow(2))
            index = np.random.randint(0, 10)
            R_vr = auxiliary_batch.value[index+1].data * self.args.gamma + auxiliary_batch.reward[index]
            value_replay_loss = 0.5 * torch.squeeze((R_vr - auxiliary_batch.value[index]).pow(2))

        self.optimizer.zero_grad()
        
        # Value function replay
        
        '''
        ###########################
        add .data to value[index+1]
        ###########################
        '''
        # Back-propagation
       
        total_loss = (policy_loss + 0.5 * value_loss +  \
                     reward_prediction_loss +  tae_loss +  \
                      value_replay_loss)
        #print(policy_loss.data, value_loss.data, reward_prediction_loss.data, tae_loss.data, value_replay_loss.data)
        total_loss.backward()
        '''
        #############################################################################################
        change retain_graph to False after changing the input in Action_M from h, c to h.data, c.data
        #############################################################################################
        '''
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

        # Apply updates
        self.sync_to()
        self.optimizer.step()
        self.sync_from()
        
        return total_loss.data[0][0]
        

    def process_state(self, state):

        img = np.expand_dims(np.transpose(state['RGB_INTERLEAVED'], (2, 0, 1)), 0)
        #order = np.expand_dims((state['ORDER']), 0)
        
        img = torch.FloatTensor(img.astype(float)/255).to(self.device)
        '''
        ####################
        rescale img to [0,1]
        ####################
        '''
        order = torch.LongTensor([[word2id[word] for word in state['INSTR'].split()]]).to(self.device)
        
        return State(img, order)

    def train(self):

            h1, c1 = (Variable(torch.randn(1, 256)).to(self.device), 
                        Variable(torch.randn(1, 256)).to(self.device)) 
        
            h2, c2 = (Variable(torch.randn(1, 256)).to(self.device), 
                        Variable(torch.randn(1, 256)).to(self.device)) 
            state = self.process_state(self.env.observations())
            episode_length = 0
            values = []
            log_probs = []
            rewards = []
            entropies = []
            '''
            Move the initialization of lists to the outer loop
            '''
            while True:
                episode_length += 1
                logit, value, h1, c1, h2, c2 = self.model(state, h1, c1, h2, c2)
                # Calculate entropy from action probability distribution
                prob = F.softmax(logit,dim=1)
                log_prob = F.log_softmax(logit,dim=1)
                entropy = -(log_prob * prob).sum(1)


                # Take an action from distribution
                action = prob.multinomial(1).data
                log_prob = log_prob.gather(1, action)       

                # Perform the action on the environment
                reward = self.env.step(self.ACTIONS[action.cpu().numpy()[0][0]], num_steps=4)
                reward = max(min(reward, 1), -1)

                if not self.env.is_running():
                    print('Environment stops early')
                    self.env.reset() # Environment timed-out 

                next_state = self.process_state(self.env.observations())

                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)
                # Push to experience replay buffer
                # THERE IS NO Terminal state in the buffer, ONLY transition
                self.memory.push(state, logit, next_state, reward, value)

                # move to next state
                state = next_state

                # Go to next episode
                if (episode_length >= 300) | (reward != 0):
                    if reward ==0:
                        self.env.reset()
                        final_value = torch.FloatTensor([[0]]).to(self.device)
                    else:
                        _, final_value, _, _, _, _ = self.model(next_state, h1, c1, h2, c2)
                    values.append(final_value)
                    loss = self.optimize_model(values, log_probs, rewards, entropies)   
                    self.memory.clear()
                    return loss, reward

                    