from torch.nn import init
import torch
from torch.utils.data.sampler import Sampler
import copy
import random
import numpy as np

class SubsetSequentialSampler(Sampler):
    def __init__(self, indices, minibatch_size=2):
        self.minibatch_size = minibatch_size
        if not any(isinstance(el, list) for el in indices):
            self.indices = [indices]
        else:
            self.indices = indices
        self.env_len = len(self.indices[0])

    def __iter__(self):
        if len(self.indices) > 1:
            l_indices = copy.deepcopy(self.indices)

            l_iter = list()
            for _ in range(0, self.env_len, self.minibatch_size):
                for i in range(len(l_indices)):
                    l_iter.extend(l_indices[i][:self.minibatch_size])
                    del l_indices[i][:self.minibatch_size]
        else:
            l_iter = copy.deepcopy(self.indices[0])
        return iter(l_iter)

    def __len__(self):
        return sum([len(el) for el in self.indices])

class SubsetRamdomSampler(Sampler):
    def __init__(self, indices, minibatch_size=2, same_order_in_groups=True):
        self.minibatch_size = minibatch_size
        self.same_order_in_groups = same_order_in_groups
        if not any(isinstance(el, list) for el in indices):
            self.indices = [indices]
        else:
            self.indices = indices
        self.env_len = len(self.indices[0])

    def __iter__(self):
        if len(self.indices) > 1:
            if self.same_order_in_groups:
                l_shuffled = copy.deepcopy(self.indices)
                random.shuffle(l_shuffled[0])
                for i in range(1, len(self.indices)):
                    l_shuffled[i] = [el + i * self.env_len for el in l_shuffled[0]]
            else:
                l_shuffled = copy.deepcopy(self.indices)
                for l in l_shuffled:
                    random.shuffle(l)

            l_iter = list()
            for _ in range(0, self.env_len, self.minibatch_size):
                for i in range(len(l_shuffled)):
                    l_iter.extend(l_shuffled[i][:self.minibatch_size])
                    del l_shuffled[i][:self.minibatch_size]
        else:
            l_shuffled = copy.deepcopy(self.indices[0])
            random.shuffle(l_shuffled)
            l_iter = l_shuffled
        return iter(l_iter)

    def __len__(self):
        return sum([len(el) for el in self.indices])

def fix_seed(seed):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def init_weights(net, init_config={}):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            for param_name, param_config in init_config.items():
                if hasattr(m, param_name):
                    param = getattr(m, param_name)
                    if param is not None:
                        init_type = param_config.get('type')
                        if init_type == 'normal':
                            param_gain = param_config.get('gain')
                            init.normal_(param.data, 0.0, 1/ np.sqrt(param_gain))
                        elif init_type == 'xavier':
                            param_gain = param_config.get('gain')
                            init.xavier_normal_(param.data, gain=param_gain)
                        elif init_type == 'kaiming':
                            init.kaiming_normal_(param.data, a=0, mode='fan_in')
                        elif init_type == 'orthogonal':
                            param_gain = param_config.get('gain')
                            init.orthogonal_(param.data, gain=param_gain)
                        elif init_type == 'constant':
                            init.constant_(param.data, 0.0)
                        elif init_type == 'default':
                            pass
                        else:
                            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
            if hasattr(m, 'bias_context') and m.bias_context is not None:
                init.constant_(m.bias_context.data, 0.0)
    net.apply(init_func)

def set_requires_grad(module, tf=False):
    module.requires_grad = tf
    for param in module.parameters():
        param.requires_grad = tf

def count_parameters(model, mode='ind'):
    if mode == 'ind':
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif mode == 'layer':
        return sum(1 for p in model.parameters() if p.requires_grad)
    elif mode == 'row':
        n_mask = 0
        for p in model.parameters():
            if p.dim() == 1:
                n_mask += 1
            else:
                n_mask += p.size(0) 
        return n_mask