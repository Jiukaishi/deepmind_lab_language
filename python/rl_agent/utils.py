import numpy as np
import torch
import random
from torch.autograd import Variable
from collections import namedtuple
from collections import defaultdict
from collections import deque
import math

START = "<s>"
STOP = "</s>"
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
}
word2id = dict(zip(vocab, range(len(vocab))))

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'value'))

State = namedtuple('State', ('visual', 'instruction'))

        
    
def _action(*entries):
    return np.array(entries, dtype=np.intc)


ACTIONS = [
    _action(0, 0, 0, 1, 0, 0, 0),    # Forward
    _action(0, 0, 0, -1, 0, 0, 0),   # Backward
    _action(0, 0, -1, 0, 0, 0, 0),   # Strafe Left
    _action(0, 0, 1, 0, 0, 0, 0),    # Strafe Right
    _action(-20, 0, 0, 0, 0, 0, 0),  # Look Left
    _action(20, 0, 0, 0, 0, 0, 0),   # Look Right
    _action(-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
    _action(20, 0, 0, 1, 0, 0, 0),   # Look Right + Forward
    _action(0, 0, 0, 0, 1, 0, 0),    # Fire.
]

class SharedRMSprop(torch.optim.Optimizer):
    """Implements RMSprop algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=7e-4,
                 alpha=0.99,
                 eps=0.1,
                 weight_decay=0,
                 momentum=0,
                 centered=False):
        defaults = defaultdict(
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered)
        super(SharedRMSprop, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['grad_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['square_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['momentum_buffer'] = p.data.new().resize_as_(
                    p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['square_avg'].share_memory_()
                state['step'].share_memory_()
                state['grad_avg'].share_memory_()
                state['momentum_buffer'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'RMSprop does not support sparse gradients')
                state = self.state[p]

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = square_avg.addcmul(-1, grad_avg,
                                             grad_avg).sqrt().add_(
                                                 group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    p.data.add_(-group['lr'], buf)
                else:
                    p.data.addcdiv_(-group['lr'], grad, avg)

        return loss
                
def mse_loss(predicted, target):
    return torch.sum((predicted - target) ** 2)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen = capacity)
        self._zero_reward_indices = deque()
        self._non_zero_reward_indices = deque()
        self._top_frame_index = 0
        
    def push(self, episode_length, *args):
        self.memory.append(Transition(*args))
        frame_index = self._top_frame_index + len(self.memory)
        was_full = self.full()
        if episode_length > 3:
            if args[3] == 0:
                self._zero_reward_indices.append(frame_index)
            else:
                self._non_zero_reward_indices.append(frame_index)
                
        if was_full:
            self._top_frame_index += 1

            cut_frame_index = self._top_frame_index + 3
            # Cut frame if its index is lower than cut_frame_index.
            if len(self._zero_reward_indices) > 0 and \
                    self._zero_reward_indices[0] < cut_frame_index:
                self._zero_reward_indices.popleft()

            if len(self._non_zero_reward_indices) > 0 and \
                    self._non_zero_reward_indices[0] < cut_frame_index:
                self._non_zero_reward_indices.popleft()

        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        
        
    def sample_vr(self):   
        index = np.random.randint(len(self._zero_reward_indices) + len(self._non_zero_reward_indices))
         
        if index < len(self._non_zero_reward_indices):
            index = np.random.randint(len(self._non_zero_reward_indices))
            end_index = self._non_zero_reward_indices[index]
        else:
            index = np.random.randint(len(self._zero_reward_indices))
            end_index = self._zero_reward_indices[index]
            
        start_index = end_index - 3 - self._top_frame_index
        sampled_frames = []
        for i in range(4):
            sampled_frames.append(self.memory[start_index+i])
            
        return sampled_frames
            
    
    def sample_rp(self, batch_size):
        rp_sample = []
        for i in range(batch_size):
            from_zero  = np.random.randint(2)

            if len(self._zero_reward_indices) == 0:
                # zero rewards container was empty
                from_zero = False
            elif len(self._non_zero_reward_indices) == 0:
                # non zero rewards container was empty
                from_zero = True

            if from_zero:
                index = np.random.randint(len(self._zero_reward_indices))
                end_index = self._zero_reward_indices[index]
            else:
                index = np.random.randint(len(self._non_zero_reward_indices))
                end_index = self._non_zero_reward_indices[index]

            start_index = end_index - 3 - self._top_frame_index
            sampled_frames = []
            for i in range(4):
                sampled_frames.append(self.memory[start_index+i])
            rp_sample.append(Transition(*zip(*sampled_frames)))
            
        return rp_sample

        
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