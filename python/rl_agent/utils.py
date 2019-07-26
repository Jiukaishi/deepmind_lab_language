import numpy as np
import torch
from torch.autograd import Variable
from collections import namedtuple

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

Transition = namedtuple('Transition',
                        ('state', 'action_logit', 'next_state', 'reward', 'value'))

State = namedtuple('State', ('visual', 'instruction'))

def _action(*entries):
    return np.array(entries, dtype=np.intc)


ACTIONS = [
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

class SharedRMSprop(torch.optim.RMSprop):
    def __init__(self, params, lr=1e-3, alpha = 0.99, eps=1e-8,
                 weight_decay=0, momentum=0):
        super(SharedRMSprop, self).__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                
def mse_loss(predicted, target):
    return torch.sum((predicted - target) ** 2)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, *args):
        self.memory.append(Transition(*args))
        
        while len(self.memory) > self.capacity:
            self.memory.pop(0)

    def sample(self, batch_size):
        start_index = np.random.randint(0, len(self.memory) - batch_size)
        return self.memory[start_index : start_index + batch_size]
    
    def sample_rp(self, batch_size):
        from_zero  = np.random.randint(2)
        if from_zero:
            return self.memory[len(self.memory) - batch_size : len(self.memory)]
        else:
            start_index = np.random.randint(0, len(self.memory) -batch_size -1)
            return self.memory[start_index : start_index + batch_size]
        
    def __len__(self):
        return(len(self.memory))
    
    def full(self):
        if (len(self.memory) >= self.capacity):
            return True
        return False
    
    def clear(self):
        self.memory = []
    
class FakeEnvironment(object):
    def __init__(self):
        pass
    
    def reset(self):
        pass
    
    def step(self, action):
        return self.generate_state(), 2, False, 1
        
    def observations(self):
        vision = np.random.randint(0, 100, (84, 84, 3))
        instruction = np.random.randint(0, 10, (4))
        
        return {'RGB_INTERLACED' : vision, 'ORDER' : instruction}