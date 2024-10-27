import torch
import torch.nn as nn
from torchdiffeq import odeint
from models import *
import numpy as np

class Forecaster(nn.Module):
    def __init__(self, dataset_name, in_dim, out_dim, n_env, method, options):
        super().__init__()

        n_left = 1
        n_right = n_env

        if dataset_name == 'pendulum':
            self.left_model  = nn.ModuleList([MLP(in_dim, out_dim) for _ in range(n_left)])
            self.right_model = nn.ModuleList([MLP(in_dim, out_dim) for _ in range(n_right)])
        if dataset_name == 'gs':
            self.left_model  = nn.ModuleList([CNN2D(in_dim, out_dim, 3) for _ in range(n_left)])
            self.right_model = nn.ModuleList([CNN2D(in_dim, out_dim, 3) for _ in range(n_right)])
        if dataset_name in ['burgers', 'combined']:
            self.left_model  = nn.ModuleList([CNN1D(in_dim, out_dim, 7) for _ in range(n_left)])
            self.right_model = nn.ModuleList([CNN1D(in_dim, out_dim, 7) for _ in range(n_right)])          
        if dataset_name == 'kolmo':
            self.left_model  = nn.ModuleList([FNO2d(10, 10, out_dim, in_dim) for _ in range(n_left)])
            self.right_model = nn.ModuleList([FNO2d(10, 10, out_dim, in_dim) for _ in range(n_right)])                

        self.derivative_estimator = DerivativeEstimatorMultiEnv(self.left_model, self.right_model, n_env=n_env)
        self.method = method
        self.options = options
        self.int_ = odeint 
        
    def forward(self, y, t, env, epsilon = 0):
        self.derivative_estimator.set_env(env)
        if epsilon < 1e-3:
            epsilon = 0

        if epsilon == 0:
            res = self.int_(self.derivative_estimator, y0=y[..., 0], t=t, method=self.method, options=self.options)
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
                    res_seg = self.int_(self.derivative_estimator, y0=y[..., start_i], t=t_seg,
                                        method=self.method, options=self.options)
                    if len(res) == 0:
                        res.append(res_seg)
                    else:
                        res.append(res_seg[1:])
                    start_i = end_i
            t_seg = t[start_i:]
            res_seg = self.int_(self.derivative_estimator, y0=y[..., start_i], t=t_seg, method=self.method,
                                options=self.options)
            if len(res) == 0:
                res.append(res_seg)
            else:
                res.append(res_seg[1:])
            res = torch.cat(res, dim=0)
        return torch.movedim(res, 0, -1)
    
class DerivativeEstimatorMultiEnv(nn.Module):

    def __init__(self, left_model, right_model, n_env):
        super().__init__()

        assert isinstance(left_model, nn.ModuleList)
        assert isinstance(right_model, nn.ModuleList)

        self.left_model = left_model
        self.right_model = right_model
        self.env = None
        self.n_env = n_env

    def set_env(self, env):
        self.env = env

    def forward(self, t, u):
        left_res, right_res = None, None
        left_res = self.left_model[0](u)
        right_res = self.right_model[self.env](u)
        return left_res + right_res