import torch
import torch.nn as nn
from functools import partial
from torchdiffeq import odeint
from fuels.model.solvers import get_numerical_solver
from fuels.model.networks import get_nn_model
import numpy as np

class Derivative(nn.Module):
    def __init__(self, dataset_name, in_dim, out_dim, code_c, factor, n_env, is_complete, type_augment):
        super().__init__()
        self.type_augment = type_augment
        self.codes = nn.Parameter(0 * torch.ones(n_env, code_c))
        self.model_phy = get_numerical_solver(dataset_name, code_c, is_complete, n_env)
        self.model_aug = get_nn_model(dataset_name, in_dim, out_dim, factor, self.codes)

    def forward(self, t, y0, env):
        if self.type_augment == 'serie':
            y_phy = self.model_phy(t, y0, env, self.codes)
            y_nn = self.model_aug(y_phy, env)
            return y_nn
        elif self.type_augment == 'additive':
            y_phy = self.model_phy(t, y0, env, self.codes)
            y_nn = self.model_aug(y0, env)
            return y_nn + y_phy
        elif self.type_augment == 'residual':
            y_phy = self.model_phy(t, y0, env, self.codes)
            y_nn = self.model_aug(y_phy, env)
            return y_phy + y_nn
        elif self.type_augment == 'phy':
            return self.model_phy(t, y0, env, self.codes)
        else:
            return self.model_aug(y0, env)

class Forecaster(nn.Module):
    def __init__(self, dataset_name, in_dim, out_dim, code_c, factor, n_env, is_complete, type_augment, method, options):
        super().__init__()
        self.method = method
        self.options = options
        self.int_ = odeint
        self.derivative = Derivative(dataset_name, in_dim, out_dim, code_c, factor, n_env, is_complete, type_augment)
        
    def forward(self, y, t, env, epsilon=0):
        if epsilon < 1e-3:
            epsilon = 0

        if epsilon == 0:
            res = self.int_(partial(self.derivative, env = env), y0=y[..., 0], t=t, method=self.method, options=self.options)
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
                    res_seg = self.int_(partial(self.derivative, env = env), y0=y[..., start_i], t=t_seg,
                                        method=self.method, options=self.options)
                    if len(res) == 0:
                        res.append(res_seg)
                    else:
                        res.append(res_seg[1:])
                    start_i = end_i
            t_seg = t[start_i:]
            res_seg = self.int_(partial(self.derivative, env = env), y0=y[..., start_i], t=t_seg, method=self.method,
                                options=self.options)
            if len(res) == 0:
                res.append(res_seg)
            else:
                res.append(res_seg[1:])
            res = torch.cat(res, dim=0)
        return torch.movedim(res, 0, -1)