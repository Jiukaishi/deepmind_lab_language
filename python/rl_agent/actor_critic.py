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
from rl_agent.model import Model, Adv_Model
from rl_agent.utils import *

def pad_batch(seqs, device):
    seq_lengths = [len(s) for s in seqs]
    max_length = max(seq_lengths)

    if isinstance(seqs[0], torch.Tensor):
        if len(seqs[0].shape) > 1:
            result = [
                torch.nn.functional.pad(
                    seqs[i],
                    (0, 0, 0, max_length - seq_lengths[i])
                )
                for i in range(len(seqs))
            ]
        else:
            result = [
                torch.nn.functional.pad(
                    seqs[i],
                    (0, max_length - seq_lengths[i])
                )
                for i in range(len(seqs))
            ]
    else:
        result = [
            torch.nn.functional.pad(
                torch.stack(seqs[i]),
                (0, 0, 0, max_length - seq_lengths[i])
            )
            for i in range(len(seqs))
        ]
    return torch.stack(result), torch.tensor(seq_lengths, device=device).long()

class RL_Agent(object):
    def __init__(self, env, actions, args, shared_model, optimizer=None, teaching_forcing_ratio =0.5,  device = 'cpu', train_prior = False, train_posterior = False):
        if device != 'cpu':
            torch.cuda.set_device(int(device[-1]))
        self.ACTIONS = actions
        self.prior_criterion = torch.nn.NLLLoss()
        self.shared_model = shared_model
        if train_prior or train_posterior:
            self.model = Adv_Model(len(self.ACTIONS))
        else:
            self.model = Model(len(self.ACTIONS))
        self.sync_from()
        self.model.to(device)
        self.device = device
        self.train_prior = train_prior
        self.train_posterior = train_posterior
        self.teacher_forcing_ratio = teaching_forcing_ratio
        if optimizer is None:
            self.optimizer = optim.RMSprop(self.shared_model.parameters(), lr=0.0001, eps = 0.1, weight_decay = 0.99)
        else:
            self.optimizer = optimizer
        
        self.memory = ReplayMemory(300)
        self.args = args
        self.env = env
        
    def sync_to(self):
        for param, shared_param in zip(self.model.parameters(),self.shared_model.parameters()):
            if param.grad is None:
                continue
            if shared_param.grad is not None: 
                shared_param._grad += param.grad.cpu()
            else:
                shared_param._grad = param.grad.cpu()
                
        
    def sync_from(self):
        self.model.load_state_dict(self.shared_model.state_dict())
        
    def optimize_model(self, values, log_probs, rewards, entropies, advice = None, representation = None):
        R = values[-1].data

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
   

        # Auxiliary loss
        language_prediction_loss = 0 
        tae_loss = 0
        reward_prediction_loss = 0
        value_replay_loss = 0

        # Non-skewed sampling from experience buffer
        if len(rewards)>=13:
            auxiliary_sample = self.memory.sample(11)
            auxiliary_batch = Transition(*zip(*auxiliary_sample))

            

            # TAE Loss
            visual_input = auxiliary_batch.state[:10]
            language_target = torch.cat([t.instruction for t  in visual_input], 0).view(-1)
            visual_input = torch.cat([t.visual for t in visual_input], 0)
            
            visual_target = auxiliary_batch.state[1:11]
            visual_target = torch.cat([t.visual for t in visual_target], 0)

            action_logit = torch.cat(auxiliary_batch.action_logit[:10], 0)

            tae_output = self.model.tAE(visual_input, action_logit)
            tae_loss = torch.sum((tae_output - visual_target).pow(2))
            
            # Language Prediction Loss
            lp_output = self.model.language_predictor(self.model.vision_m(visual_input))
            language_prediction_loss = torch.nn.CrossEntropyLoss()(lp_output, language_target)
        
        if len(rewards)>=15:
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
         # Value function replay
        index = np.random.randint(0, len(values))
        R_vr = values[index+1].data * self.args.gamma + rewards[index]
        value_replay_loss = 0.5 * torch.squeeze((R_vr - values[index]).pow(2))

              
        # advice reconstruction
        if self.train_prior:
                    for i in range(advice.shape[0]):
                        use_teacher_forcing = bool(np.random.rand() < self.teacher_forcing_ratio)
                        advice_scaffold = advice[i] if use_teacher_forcing else None
                        advice_pred, _ = self.sample_decoder(decoder_t='prior', advice_scaffold=advice_scaffold)
                        min_length = min(advice[i].shape[0], advice_pred.shape[0])
                        language_prediction_loss += self.prior_criterion(
                            advice_pred[:min_length], advice[i][:min_length]
                        )
        if self.train_posterior:
            #representaion, representation_length = pad_batch(representation, device=self.device)
            representation_length = torch.tensor([representation[i].size(0) for i in range(len(representation))], device = self.device)
            representation = torch.stack(representation)
            post_representation = self.model.posterior_forward(representation, representation_length)
            for i in range(advice.shape[0]):
                use_teacher_forcing = bool(np.random.rand() < self.teacher_forcing_ratio)
                advice_scaffold = advice[i] if use_teacher_forcing else None
                advice_pred, _ = self.sample_decoder(decoder_t='posterior', hidden=post_representation[i].view(1, 1, -1), advice_scaffold=advice_scaffold)

                min_length = min(advice[i].shape[0], advice_pred.shape[0])
                language_prediction_loss += self.prior_criterion(
                    advice_pred[:min_length], advice[i][:min_length]
                )
         
        self.model.zero_grad()    
       
        # Back-propagation
        
       
        total_loss = (policy_loss + 0.5 * value_loss +  \
                     reward_prediction_loss +  tae_loss +  \
                      value_replay_loss + language_prediction_loss)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

        # Apply updates
        self.sync_to()
        self.optimizer.step()
        self.sync_from()
        
        return total_loss.data[0][0]
        
    def sample_decoder(self, decoder_t='prior', hidden=None, advice_scaffold=None, randomize=False):
        # advice_scaffold: Input true advice tensor to scaffold predicted advice
        # (aka teacher forcing)

        assert decoder_t in ['prior', 'posterior']
        if decoder_t in 'prior':
            decoder = self.model.prior_decoder
        else:
            decoder = self.model.post_decoder
        
        if hidden is None:
            decoder_hidden = torch.zeros(1, 1, self.model.embed_n, device=self.device)
        else:
            decoder_hidden = hidden
        decoder_input = torch.tensor([[word2id[START]]], device=self.device)
        #encoder_outputs = torch.zeros(1, 1, self.embed_n, device=self.device)# no encoder output

        output_probs = []
        outputs = []

        if advice_scaffold is not None:
            max_length = advice_scaffold.size(0)
        else:
            max_length = self.model.max_advice_length

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            if advice_scaffold is not None:
                decoder_input = advice_scaffold[di]
            else:
                if randomize:
                    probs = torch.exp(decoder_output).squeeze().detach().cpu()
                    decoder_input = np.random.choice(self.model.vocab_size, p=probs.numpy())
                    decoder_input = torch.tensor(decoder_input, device=self.device).long()
                else:
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input
            output_probs.append(decoder_output)
            outputs.append(decoder_input)

            if decoder_input.item() == word2id[STOP]:
                break

        return torch.cat(output_probs), torch.tensor(outputs, device=self.device)

    def process_state(self, state):

        img = np.expand_dims(np.transpose(state['RGB_INTERLEAVED'], (2, 0, 1)), 0)
        #order = np.expand_dims((state['ORDER']), 0)
        
        img = torch.FloatTensor(img.astype(float)/255).to(self.device)
        #order = [START] + state['INSTR'].split() + [STOP]
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
            #advice = []
            #representation = []
            while True:
                #advice.append(state.instruction)
                episode_length += 1
                logit, value, h1, c1, h2, c2 = self.model(state, h1, c1, h2, c2)
                # Calculate entropy from action probability distribution
                prob = F.softmax(logit,dim=-1)
                log_prob = F.log_softmax(logit,dim=-1)
                entropy = -(log_prob * prob).sum(1,keepdim=True)
                entropies.append(entropy)
                #representation.append(mix_out)

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
                
                # Push to experience replay buffer
                self.memory.push(state, logit, next_state, reward, value)

                # move to next state
                state = next_state

                # Go to next episode
                if (episode_length >= 300) | (reward != 0):
                    if reward ==0:                  
                        final_value = torch.zeros(1,1).to(self.device)
                    else:
                        _, final_value, _, _, _, _ = self.model(next_state, h1, c1, h2, c2)
                    values.append(final_value)
                    self.env.reset()
                    loss = self.optimize_model(values, log_probs, rewards, entropies)   
                    self.memory.clear()
                    return loss, reward
                if episode_length % 50 == 0:
                    h1, c1, h2, c2 = h1.data, c1.data, h2.data, c2.data

                    