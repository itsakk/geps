import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Sin(nn.Module):
    def __init__(self, code, factor):
        super().__init__()
        self.factor = factor
        self.agnostic_a = nn.Parameter(torch.tensor([1.]))
        self.weight = nn.Parameter(torch.empty(1, code))

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x, codes):
        agnostic_a = self.agnostic_a + self.factor * F.linear(codes, self.weight)
        if len(x.shape) == 3:
            agnostic_a = agnostic_a[..., None]
        if len(x.shape) == 4:
            agnostic_a = agnostic_a[..., None, None]
        return torch.sin(agnostic_a*x)

class GaussianActivation(nn.Module):
    def __init__(self, code, factor):
        super().__init__()
        self.factor = factor
        self.agnostic_a = nn.Parameter(torch.tensor([1.]))
        self.weight = nn.Parameter(torch.empty(1, code))

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x, codes):
        agnostic_a = self.agnostic_a + self.factor * F.linear(codes, self.weight)
        if len(x.shape) == 3:
            agnostic_a = agnostic_a[..., None]
        if len(x.shape) == 4:
            agnostic_a = agnostic_a[..., None, None]
        return torch.exp(-x**2/(2*agnostic_a**2))

class Quadratic(nn.Module):
    def __init__(self, code, factor):
        super().__init__()
        self.factor = factor
        self.agnostic_a = nn.Parameter(torch.tensor([1.]))
        self.weight = nn.Parameter(torch.empty(1, code))

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x, codes):
        agnostic_a = self.agnostic_a + self.factor * F.linear(codes, self.weight)
        if len(x.shape) == 3:
            agnostic_a = agnostic_a[..., None]
        if len(x.shape) == 4:
            agnostic_a = agnostic_a[..., None, None]
        return 1/(1+(agnostic_a*x)**2)

class MultiQuadratic(nn.Module):
    def __init__(self, code, factor):
        super().__init__()
        self.factor = factor
        self.agnostic_a = nn.Parameter(torch.tensor([1.]))
        self.weight = nn.Parameter(torch.empty(1, code))

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x, codes):
        agnostic_a = self.agnostic_a + self.factor * F.linear(codes, self.weight)
        if len(x.shape) == 3:
            agnostic_a = agnostic_a[..., None]
        if len(x.shape) == 4:
            agnostic_a = agnostic_a[..., None, None]
        return 1/(1+(agnostic_a*x)**2)**0.5

class Laplacian(nn.Module):
    def __init__(self, code, factor):
        super().__init__()
        self.factor = factor
        self.agnostic_a = nn.Parameter(torch.tensor([1.]))
        self.weight = nn.Parameter(torch.empty(1, code))

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x, codes):
        agnostic_a = self.agnostic_a + self.factor * F.linear(codes, self.weight)
        if len(x.shape) == 3:
            agnostic_a = agnostic_a[..., None]
        if len(x.shape) == 4:
            agnostic_a = agnostic_a[..., None, None]
        return torch.exp(-torch.abs(x)/agnostic_a)

class SuperGaussian(nn.Module):
    def __init__(self, code, factor):
        super().__init__()
        self.factor = factor
        self.agnostic_a = nn.Parameter(torch.tensor([1.]))
        self.agnostic_b = nn.Parameter(torch.tensor([1.]))
        self.weight_a = nn.Parameter(torch.empty(1, code))
        self.weight_b = nn.Parameter(torch.empty(1, code))

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_a, -bound, bound)
        nn.init.uniform_(self.weight_b, -bound, bound)

    def forward(self, x, codes):
        agnostic_a = self.agnostic_a + self.factor * F.linear(codes, self.weight_a)
        agnostic_b = self.agnostic_b + self.factor * F.linear(codes, self.weight_b)
        if len(x.shape) == 3:
            agnostic_a = agnostic_a[..., None]
            agnostic_b = agnostic_b[..., None]
        if len(x.shape) == 4:
            agnostic_a = agnostic_a[..., None, None]
            agnostic_b = agnostic_b[..., None, None]
        return torch.exp(-x**2/(2*agnostic_a**2))**agnostic_b

class ExpSin(nn.Module):
    def __init__(self, code, factor):
        super().__init__()
        self.factor = factor
        self.agnostic_a = nn.Parameter(torch.tensor([1.]))
        self.weight = nn.Parameter(torch.empty(1, code))

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x, codes):
        agnostic_a = self.agnostic_a + self.factor * F.linear(codes, self.weight)
        if len(x.shape) == 3:
            agnostic_a = agnostic_a[..., None]
        if len(x.shape) == 4:
            agnostic_a = agnostic_a[..., None, None]
        return torch.exp(-torch.sin(agnostic_a*x))

class Swish(nn.Module):

    def __init__(self, code, factor):
        super().__init__()
        self.factor = factor
        self.agnostic_beta = nn.Parameter(torch.tensor([0.5]))
        self.weight = nn.Parameter(torch.empty(1, code))

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x, codes):
        softplus = F.softplus(self.agnostic_beta + self.factor * F.linear(codes, self.weight))
        if len(x.shape) == 3:
            softplus = softplus[..., None]
        if len(x.shape) == 4:
            softplus = softplus[..., None, None]
        return (x * torch.sigmoid_(x * softplus)).div_(1.1)