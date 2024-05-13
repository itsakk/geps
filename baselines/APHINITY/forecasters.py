import torch
import torch.nn as nn
from torchdiffeq import odeint
from models import *
from solvers import *
from functools import partial

class Forecaster(nn.Module):
    def __init__(self, dataset_name, in_dim, out_dim, n_env, augment_type, is_complete, method, options):
        super().__init__()         

        self.derivative_estimator = DerivativeEstimator(dataset_name, in_dim, out_dim, n_env, augment_type = augment_type, is_complete = is_complete)
        self.method = method
        self.options = options
        self.int_ = odeint 
        
    def forward(self, y, t, env, epsilon=0):
        if epsilon < 1e-3:
            epsilon = 0

        if epsilon == 0:
            res = self.int_(partial(self.derivative_estimator, env = env), y0=y[..., 0], t=t, method=self.method, options=self.options)
        else:
            eval_points = np.random.random(len(t)) < epsilon
            eval_points[-1] = False
            eval_points = eval_points[1:]
            start_i, end_i = 0, None
            res = []
            for i, eval_point in enumerate(eval_points):
                if eval_point:
                    end_i = i + 1
                    t_seg = t[start_i:end_i + 1]
                    res_seg = self.int_(partial(self.derivative_estimator, env = env), y0=y[..., start_i], t=t_seg,
                                        method=self.method, options=self.options)
                    if len(res) == 0:
                        res.append(res_seg)
                    else:
                        res.append(res_seg[1:])
                    start_i = end_i
            t_seg = t[start_i:]
            res_seg = self.int_(partial(self.derivative_estimator, env = env), y0=y[..., start_i], t=t_seg, method=self.method,
                                options=self.options)
            if len(res) == 0:
                res.append(res_seg)
            else:
                res.append(res_seg[1:])
            res = torch.cat(res, dim=0)
        return torch.movedim(res, 0, -1)

class DerivativeEstimator(nn.Module):

    def __init__(self, dataset_name, in_dim, out_dim, n_env, augment_type = 'serie', is_complete = False):
        super().__init__()
        self.augment_type = augment_type
        self.model_aug = get_nn_model(dataset_name, in_dim, out_dim)
        self.model_phy = get_numerical_solver(dataset_name, n_env, is_complete)

    def forward(self, t, u, env):
        if self.augment_type == 'serie':
            u = self.model_phy(t, u, env)
            u = self.model_aug(u)
            return u
        elif self.augment_type == 'additive':
            res_phy = self.model_phy(t, u, env)
            res_aug = self.model_aug(u)
            return res_phy + res_aug