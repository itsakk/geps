from torch.nn import init
import torch
import numpy as np
import math

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
                            init.kaiming_normal_(param.data, a=math.sqrt(5), mode='fan_in')
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
            if hasattr(m, 'bias_context') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
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