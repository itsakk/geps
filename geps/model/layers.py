import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class GEPSConv1D(nn.Module):

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
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
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
    
class GEPSConv2D(nn.Module):

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
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
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
    
class GEPSLinear(nn.Module):

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
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
                nn.init.uniform_(self.bias, -bound, bound)
                nn.init.uniform_(self.bias_context, -bound, bound)

    def forward(self, input: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        weights = torch.einsum("ir,  brp-> bip", self.A, codes)
        context_weights = torch.einsum("bir, rj -> bij", weights, self.B)
        context_bias = torch.matmul(torch.diagonal(codes, dim1=-2, dim2=-1), self.bias_context).view(-1, self.out_features)
        combined_weight = self.weight + self.factor * context_weights
        combined_bias = self.bias + self.factor * context_bias
        return torch.bmm(input.unsqueeze(1), combined_weight).squeeze(1) + combined_bias
    
class GEPSLinear1D(nn.Module):

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
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
                nn.init.uniform_(self.bias, -bound, bound)
                nn.init.uniform_(self.bias_context, -bound, bound)

    def forward(self, input: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        weights = torch.einsum("ir,  brp-> bip", self.A, codes)
        context_weights = torch.einsum("bir, rj -> bij", weights, self.B)
        context_bias = torch.matmul(torch.diagonal(codes, dim1=-2, dim2=-1), self.bias_context).view(-1, self.out_features)
        combined_weight = self.weight + self.factor * context_weights
        combined_bias = self.bias + self.factor * context_bias
        return torch.bmm(input, combined_weight) + combined_bias.unsqueeze(1)


class GEPSSpectralConv1d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, code, factor = 1):
        super(GEPSSpectralConv1d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.factor = factor

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        
        self.A_1 = nn.Parameter(torch.empty((in_channels, code, self.modes1)))
        self.B_1 = nn.Parameter(torch.empty((code, out_channels, self.modes1)))

    # Complex multiplication
    def compl_mul1d(self, a, b):
        res = torch.einsum("bix,biox->box", a, b)
        return res

    def forward(self, x, codes):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[2])

        weights_1 = torch.einsum("icm, bcr-> birm", self.A_1, codes)
        context_weights_1 = torch.einsum("birm, rom -> biom", weights_1, self.B_1)
        weights1 = self.weights1 + self.factor * context_weights_1

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], weights1)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=[x.size(-1)], dim=[2])
        return x
    
class GEPSSpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, code, factor = 1):
        super(GEPSSpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2
        self.factor = factor

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        
        self.A_1 = nn.Parameter(torch.empty((in_channels, code, self.modes1, self.modes2)))
        self.B_1 = nn.Parameter(torch.empty((code, out_channels, self.modes1, self.modes2)))
        self.A_2 = nn.Parameter(torch.empty((in_channels, code, self.modes1, self.modes2)))
        self.B_2 = nn.Parameter(torch.empty((code, out_channels, self.modes1, self.modes2)))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (batch, in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,bioxy->boxy", input, weights)

    def forward(self, x, codes):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        weights_1 = torch.einsum("icmn,  bcr-> birmn", self.A_1, codes)
        context_weights_1 = torch.einsum("birmn, romn -> biomn", weights_1, self.B_1)
        weights1 = self.weights1 + self.factor * context_weights_1

        weights_2 = torch.einsum("icmn,  bcr-> birmn", self.A_2, codes)
        context_weights_2 = torch.einsum("birmn, romn -> biomn", weights_2, self.B_2)
        weights2 = self.weights2 + self.factor * context_weights_2

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2),
                             x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(
                x_ft[:, :, :self.modes1, :self.modes2], weights1)
        
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(
                x_ft[:, :, -self.modes1:, :self.modes2], weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x