import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from collections import namedtuple
from rl_agent.network_modules import *
from rl_agent.utils import *
from seq2seq import *

State = namedtuple('State', ('visual', 'instruction'))

class Model(nn.Module):
    def __init__(self, action_space):
        super(Model, self).__init__()
        
        # Core modules
        self.vision_m = Vision_M()
        self.language_m = Language_M()
        self.mixing_m = Mixing_M()
        self.action_m = Action_M()
        
        # Action selection and Value Critic
        self.policy = Policy(action_space=action_space)
        
        # Auxiliary networks
        self.tAE = temporal_AutoEncoder(self.policy, self.vision_m)
        self.language_predictor = Language_Prediction(self.language_m)
        self.reward_predictor = RewardPredictor(self.vision_m, self.language_m, self.mixing_m)
        
        
    def forward(self, x, h1, c1, h2, c2):
        '''
        Argument:
        
            img: environment image, shape [batch_size, 84, 84, 3]
            instruction: natural language instruction [batch_size, seq]
        '''
        
        vision_out = self.vision_m(x.visual)
        language_out = self.language_m(x.instruction)
        mix_out = self.mixing_m(vision_out, language_out)
        h1, c1, h2, c2 = self.action_m(mix_out, h1, c1, h2, c2)
        
        action_prob, value = self.policy(h2)
        
        return action_prob, value, h1, c1, h2, c2
    
    
class Adv_Model(Model):
    def __init__(self, action_space):
        super(Adv_Model, self).__init__(action_space)
        self.embed_n = 128
        self.state_hidden_n = 64
        self.vocab_size = 13
        self.max_advice_length = 9
        self.prior_decoder = DecoderRNN(self.embed_n, self.vocab_size)
        self.post_rnn = nn.GRU(3264, 3264, batch_first=True)
        self.post_decoder = DecoderRNN(3264, self.vocab_size)
     
    def posterior_forward(self, representations, representation_lengths):

        representations, _ = self.post_rnn(representations, None)

        row_indices = torch.arange(0, representation_lengths.size(0)).long()
        representations = representations[row_indices, representation_lengths-1, :]        
        return representations
    
