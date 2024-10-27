import torch
from geps.model.layers import *
import einops

def l2_penalty(model):
    # L2 penalty
    l2_penalty = 0
    for layer in model.children():
        if isinstance(layer, GEPSLinear):
            l2_penalty += torch.norm(layer.A, p=2) ** 2 + torch.norm(layer.B, p=2) ** 2
        elif isinstance(layer, Swish):
            l2_penalty += torch.norm(layer.linear.weight, p=2) ** 2
    l2_penalty += torch.norm(model.derivative.model_aug.codes, p=2) ** 2
    return l2_penalty

class RelativeL2(nn.Module):
    def forward(self, x, y):
        x = einops.rearrange(x, "b ... -> b (...)")
        y = einops.rearrange(y, "b ... -> b (...)")
        diff_norms = torch.linalg.norm(x - y, ord=2, dim=-1)
        y_norms = torch.linalg.norm(y, ord=2, dim=-1)
        return (diff_norms / y_norms).mean()