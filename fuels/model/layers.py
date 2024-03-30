import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class ROCAConv1D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        padding_mode: str = 'circular',
        stride: int = 1,
        factor: int = 1,
        bias: bool = True,
        code: int = 2,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.padding = padding
        self.padding_mode = padding_mode
        self.factor = factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, kernel_size), **factory_kwargs))
        self.A = nn.Parameter(torch.empty(in_channels, code, kernel_size))
        self.B = nn.Parameter(torch.empty(out_channels, code, kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.empty((out_channels), **factory_kwargs))
            self.bias_context = nn.Parameter(torch.empty((code, out_channels), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias_context', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_channels * self.kernel_size)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.A, -bound, bound)
        nn.init.uniform_(self.B, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.uniform_(self.bias_context, -bound, bound)

    def stride_input(self, inputs, kernel_size, stride):
        batch_size, channels, seq_len = inputs.shape
        batch_stride, channel_stride, seq_stride = inputs.stride()
        out_seq_len = int((seq_len - kernel_size) / stride + 1)
        new_shape = (batch_size, channels, out_seq_len, kernel_size)
        new_strides = (batch_stride, channel_stride, stride * seq_stride, seq_stride)
        return torch.as_strided(inputs, size=new_shape, stride=new_strides)

    def forward(self, input: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:

        # Adaptation rule
        weights = torch.einsum("ick,  bcr-> birk", self.A, codes)
        context_weights = torch.einsum("birk, ork -> boik", weights, self.B)
        combined_weight = self.weight + self.factor * context_weights
        context_bias = torch.matmul(torch.diagonal(codes, dim1=-2, dim2=-1), self.bias_context).view(-1, self.out_channels)
        combined_bias = self.bias + self.factor * context_bias

        # Padding
        inputs = F.pad(input, (self.padding, self.padding), mode = self.padding_mode)

        # Convolution
        input_windows = self.stride_input(inputs, self.kernel_size, self.stride)
        return torch.einsum('bihw,boiw->boh', input_windows, combined_weight) + combined_bias[..., None]
    
class ROCAConv2D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        padding_mode: str = 'circular',
        stride: int = 1,
        code: int = 2,
        factor: int = 1,
        bias: bool = True,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.padding = padding
        self.padding_mode = padding_mode
        self.factor = factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, kernel_size, kernel_size), **factory_kwargs))
        self.A = nn.Parameter(torch.empty(in_channels, code, kernel_size, kernel_size))
        self.B = nn.Parameter(torch.empty(out_channels, code, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.empty((out_channels), **factory_kwargs))
            self.bias_context = nn.Parameter(torch.empty((code, out_channels), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias_context', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_channels * self.kernel_size)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.A, -bound, bound)
        nn.init.uniform_(self.B, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.uniform_(self.bias_context, -bound, bound)

    def stride_input(self, inputs, kernel_size, stride):
        batch_size, channels, h, w = inputs.shape
        batch_stride, channel_stride, rows_stride, columns_stride = inputs.stride()
        out_h = int((h - kernel_size) / stride + 1)
        out_w = int((w - kernel_size) / stride + 1)
        new_shape = (batch_size, channels, out_h, out_w, kernel_size, kernel_size)
        new_strides = (batch_stride, channel_stride, stride * rows_stride, stride * columns_stride, rows_stride, columns_stride)
        return torch.as_strided(inputs, size=new_shape, stride=new_strides)

    def forward(self, input: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:

        # Adaptation rule
        weights = torch.einsum("ickl,  bcr-> birkl", self.A, codes)
        context_weights = torch.einsum("birkl, orkl -> boikl", weights, self.B)
        combined_weight = self.weight + self.factor * context_weights
        context_bias = torch.matmul(torch.diagonal(codes, dim1=-2, dim2=-1), self.bias_context).view(-1, self.out_channels)
        combined_bias = self.bias + self.factor * context_bias

        # Padding
        inputs = F.pad(input, (self.padding, self.padding, self.padding, self.padding), mode = self.padding_mode) 

        # Convolution       
        input_windows = self.stride_input(inputs, self.kernel_size, self.stride)
        return torch.einsum('bchwkt,bfckt->bfhw', input_windows, combined_weight) + combined_bias[..., None, None]
    
class ROCALinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 code: int = 2, factor: int = 1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.factor = factor
        self.code = code
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((in_features, out_features), **factory_kwargs))
        self.A = nn.Parameter(torch.empty((in_features, code), **factory_kwargs))
        self.B = nn.Parameter(torch.empty((code, out_features), **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.empty((out_features), **factory_kwargs))
            self.bias_context = nn.Parameter(torch.empty((code, out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias_context', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.A, -bound, bound)
        nn.init.uniform_(self.B, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.uniform_(self.bias_context, -bound, bound)

    def forward(self, input: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        weights = torch.einsum("ir,  brp-> bip", self.A, codes)
        context_weights = torch.einsum("bir, rj -> bij", weights, self.B)
        context_bias = torch.matmul(torch.diagonal(codes, dim1=-2, dim2=-1), self.bias_context).view(-1, self.out_features)
        combined_weight = self.weight + self.factor * context_weights
        combined_bias = self.bias + self.factor * context_bias
        return torch.bmm(input.unsqueeze(1), combined_weight).squeeze(1) + combined_bias

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