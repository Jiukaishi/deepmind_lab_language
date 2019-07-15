import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import namedtuple
from utils import *
from model import Model
'''
vocab = {
        'apple2',
        'ball',
        'balloon',
        'banana',
        'bottle',
        'cake',
        'can',
        'car',
        'cassette',
        'chair',
        'cherries',
        'cow',
        'flower',
        'fork',
        'fridge',
        'guitar',
        'hair_brush',
        'hammer',
        'hat',
        'ice_lolly',
        'jug',
        'key',
        'knife',
        'ladder',
        'mug',
        'pencil',
        'pig',
        'pincer',
        'plant',
        'saxophone',
        'shoe',
        'spoon',
        'suitcase',
        'tennis_racket',
        'tomato',
        'toothbrush',
        'tree',
        'tv',
        'wine_glass',
        'zebra',
        'chequered',
        'crosses',
        'diagonal_stripe',
        'discs',
        'hex',
        'pinstripe',
        'solid',
        'spots',
        'swirls',
        'small',
        'medium',
        'large',
        'the',
        'Pick',
        'in',
        'room',
        'red', 
        'green',
        'blue', 
        'cyan',
        'magenta', 
        'yellow',
        'object',
    }
'''

vocab = {
        'the',
        'Pick',
        'in',
        'room',
        'red', 
        'green',
        'blue', 
        'cyan',
        'magenta', 
        'yellow',
        'object',
    }

word2id = dict(zip(vocab, range(len(vocab))))
   

def _action(*entries):
    return np.array(entries, dtype=np.intc)

class RL_Agent(object):
    def __init__(self, env, args):
        self.ACTIONS = [
            _action(-20, 0, 0, 0, 0, 0, 0),
            _action(20, 0, 0, 0, 0, 0, 0),
            _action(0, 10, 0, 0, 0, 0, 0),
            _action(0, -10, 0, 0, 0, 0, 0),
            _action(0, 0, -1, 0, 0, 0, 0),
            _action(0, 0, 1, 0, 0, 0, 0),
            _action(0, 0, 0, 1, 0, 0, 0),
            _action(0, 0, 0, -1, 0, 0, 0),
            _action(0, 0, 0, 0, 1, 0, 0),
            _action(0, 0, 0, 0, 0, 1, 0),
            _action(0, 0, 0, 0, 0, 0, 1)]
        
        self.model = Model(len(self.ACTIONS))
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.0001)
        self.model.cuda()
        
        self.memory = ReplayMemory(300)
        self.args = args
        self.env = env
        
       
    def optimize_model(self, values, log_probs, rewards, entropies):
        R = values[-1].data 
        '''
        ########################
        add .data to varaiable R
        ########################
        '''
        gae = torch.zeros(1, 1).type(torch.cuda.FloatTensor)
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
        skewed_sample = self.memory.sample(13)  # memory.skewed_sample(13)
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

        self.optimizer.zero_grad()
        reward_prediction_loss = \
                torch.sum((rp_predicted - Variable(torch.cuda.FloatTensor(batch_rp_output))).pow(2))

        # Value function replay
        index = np.random.randint(0, 10)
        R_vr = auxiliary_batch.value[index+1].data + self.args.gamma * auxiliary_batch.reward[index]
        value_replay_loss = 0.5 * torch.squeeze((R_vr - auxiliary_batch.value[index]).pow(2))
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
        total_loss.backward(retain_graph = False)
        '''
        #############################################################################################
        change retain_graph to False after changing the input in Action_M from h, c to h.data, c.data
        #############################################################################################
        '''
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.args.clip_grad_norm)

        # Apply updates
        self.optimizer.step()
        
        return total_loss.data[0][0]
        

    def process_state(self, state):

        img = np.expand_dims(np.transpose(state['RGB_INTERLEAVED'], (2, 0, 1)), 0)
        #order = np.expand_dims((state['ORDER']), 0)
        
        img = torch.from_numpy(img).type(torch.cuda.FloatTensor)
        order = torch.tensor([[word2id[word] for word in state['INSTR'].split()]]).type(torch.cuda.LongTensor)
        
        return State(img, order)

    def train(self):
        loss_buffer = []
        rewards_buffer = []

        for episode in range(self.args.num_episodes):
            print("STARTED EPISODE", episode)
            self.env.reset()
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
                #with torch.no_grad():
                logit, value = self.model(state)
                logit = logit.data
                # Calculate entropy from action probability distribution
                prob = F.softmax(logit)
                log_prob = F.log_softmax(logit)
                entropy = -(log_prob * prob).sum(1)
                

                # Take an action from distribution
                action = prob.multinomial(1).data
                log_prob = log_prob.gather(1, action)       

                # Perform the action on the environment
                reward = self.env.step(self.ACTIONS[action.cpu().numpy()[0][0]], num_steps=4)

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
                if episode_length >= 300 or reward!=0: 
                    '''
                    ##############################################################
                    redefine the episode and do the backward after each episode
                    ##############################################################
                    '''
                    _, final_value = self.model(next_state)
                    values.append(final_value)
                    if episode_length>=15:
                        loss = self.optimize_model(values, log_probs, rewards, entropies)   
                        loss_buffer.append(loss)
                        print('Episode {} / {} has completed. Episode loss is {}. Episode reward is {}.'.
                                    format(episode, self.args.num_episodes, loss, reward))
                    self.memory.clear()

                    rewards_buffer.append(reward)

                    break
               
        np.save('/home/km/rl_loss.npy', np.array(loss_buffer))
        np.save('/home/km/rl_rewards.npy', np.array(rewards_buffer))