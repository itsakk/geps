import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from model import get_nn_model
import numpy as np

class Derivative(nn.Module):
    def __init__(self, dataset_name, in_dim, out_dim, ctx_dim):
        super().__init__()
        self.model_nn = get_nn_model(dataset_name, in_dim, out_dim, ctx_dim)

    def forward(self, t, y0):
        return self.model_nn(y0)

class Forecaster(nn.Module):
    def __init__(self, dataset_name, in_dim, out_dim, method, options, ctx_dim):
        super().__init__()
        self.method = method
        self.options = options
        self.int_ = odeint
        self.derivative = Derivative(dataset_name, in_dim, out_dim, ctx_dim)

    def forward(self, y, t, epsilon=0):
        if epsilon < 1e-3:
            epsilon = 0

        if epsilon == 0:
            res = self.int_(self.derivative, y0=y[..., 0], t=t, method=self.method, options=self.options)
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
                    res_seg = self.int_(self.derivative, y0=y[..., start_i], t=t_seg,
                                        method=self.method, options=self.options)
                    if len(res) == 0:
                        res.append(res_seg)
                    else:
                        res.append(res_seg[1:])
                    start_i = end_i
            t_seg = t[start_i:]
            res_seg = self.int_(self.derivative, y0=y[..., start_i], t=t_seg, method=self.method,
                                options=self.options)
            if len(res) == 0:
                res.append(res_seg)
            else:
                res.append(res_seg[1:])
            res = torch.cat(res, dim=0)
        return torch.movedim(res, 0, -1)