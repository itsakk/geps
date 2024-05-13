from fuels.model.layers import *
from fuels.model.activations import *
import torch
import torch.nn as nn
import einops

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, factor, codes):
        super().__init__()

        self.codes = codes
        self.code_dim = self.codes.shape[-1]
        self.linear1 = ROCALinear(in_dim, out_dim, True, self.code_dim)
        self.act1 = Swish(self.code_dim, factor)
        self.linear2 = ROCALinear(out_dim, out_dim, True, self.code_dim)
        self.act2 = Swish(self.code_dim, factor)
        self.linear3 = ROCALinear(out_dim, out_dim, True, self.code_dim)
        self.act3 = Swish(self.code_dim, factor)
        self.linear4 = ROCALinear(out_dim, in_dim, True, self.code_dim)

    def forward(self, x, env):
        codes = self.codes[env.long()]
        diag_codes = torch.diag_embed(codes)
        x1 = self.linear1(x, diag_codes)
        x1 = self.act1(x1, codes)
        x1 = self.linear2(x1, diag_codes)
        x1 = self.act2(x1, codes)
        x1 = self.linear3(x1, diag_codes)
        x1 = self.act3(x1, codes)
        x1 = self.linear4(x1, diag_codes)
        return x1
    
class CNN1D(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size, factor, codes):
        super().__init__()
        
        self.codes = codes
        self.code_dim = self.codes.shape[-1]
        self.padding = kernel_size // 2
        self.conv1 = ROCAConv1D(in_dim, out_dim, kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)
        self.act1 = Swish(self.code_dim, factor)
        self.conv2 = ROCAConv1D(out_dim, out_dim, kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)
        self.act2 = Swish(self.code_dim, factor)
        self.conv3 = ROCAConv1D(out_dim, out_dim, kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)
        self.act3 = Swish(self.code_dim, factor)
        self.conv4 = ROCAConv1D(out_dim, in_dim, kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)

    def forward(self, x, env):
        codes = self.codes[env.long()]
        diag_codes = torch.diag_embed(codes)
        x1 = self.conv1(x, diag_codes)
        x1 = self.act1(x1, codes)
        x1 = self.conv2(x1, diag_codes)
        x1 = self.act2(x1, codes)
        x1 = self.conv3(x1, diag_codes)
        x1 = self.act3(x1, codes)
        x1 = self.conv4(x1, diag_codes)
        return 0.0005 * x1

class CNN2D(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size, factor, codes):
        super().__init__()

        self.codes = codes
        self.code_dim = self.codes.shape[-1]
        self.padding = kernel_size // 2
        self.conv1 = ROCAConv2D(in_dim, out_dim, kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)
        self.act1 = Swish(self.code_dim, factor)
        self.conv2 = ROCAConv2D(out_dim, out_dim, kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)
        self.act2 = Swish(self.code_dim, factor)
        self.conv3 = ROCAConv2D(out_dim, out_dim, kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)
        self.act3 = Swish(self.code_dim, factor)
        self.conv4 = ROCAConv2D(out_dim, in_dim, kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)

    def forward(self, x, env):
        codes = self.codes[env.long()]
        diag_codes = torch.diag_embed(codes)
        x1 = self.conv1(x, diag_codes)
        x1 = self.act1(x1, codes)
        x1 = self.conv2(x1, diag_codes)
        x1 = self.act2(x1, codes)
        x1 = self.conv3(x1, diag_codes)
        x1 = self.act3(x1, codes)
        x1 = self.conv4(x1, diag_codes)
        return 0.0005 * x1

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, num_channels, factor, codes):
        super(FNO2d, self).__init__()

        self.codes = codes
        self.code_dim = self.codes.shape[-1]
        self.num_channels = num_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.kernel_size = 7
        self.padding = self.kernel_size // 2
        self.fc0 = ROCALinear1D(2 + num_channels, self.width, True, self.code_dim)

        self.conv0 = ROCASpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2, self.code_dim)
        self.conv1 = ROCASpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2, self.code_dim)
        self.conv2 = ROCASpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2, self.code_dim)
        self.conv3 = ROCASpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2, self.code_dim)
        
        self.w0 = ROCAConv2D(self.width, self.width, self.kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)
        self.w1 = ROCAConv2D(self.width, self.width, self.kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)
        self.w2 = ROCAConv2D(self.width, self.width, self.kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)
        self.w3 = ROCAConv2D(self.width, self.width, self.kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)

        self.fc1 = ROCALinear1D(self.width, 128, True, self.code_dim)
        self.fc2 = ROCALinear1D(128, num_channels, True, self.code_dim)

        self.act1 = Swish(self.code_dim, factor)
        self.act2 = Swish(self.code_dim, factor)
        self.act3 = Swish(self.code_dim, factor)
        self.act4 = Swish(self.code_dim, factor)

    def forward(self, x, env):
        b, c, size_x, size_y = x.shape 
        codes = self.codes[env.long()]
        diag_codes = torch.diag_embed(codes)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        x = einops.rearrange(x, "b c x y-> b (x y) c")
        x = self.fc0(x, diag_codes)
        x = einops.rearrange(x, "b (x y) c -> b c x y", x = size_x, y = size_y)

        x1 = self.conv0(x, diag_codes)
        x2 = self.w0(x, diag_codes)
        x = x1 + x2
        x = self.act1(x, codes)

        x1 = self.conv1(x, diag_codes)
        x2 = self.w1(x, diag_codes)
        x = x1 + x2
        x = self.act2(x, codes)

        x1 = self.conv2(x, diag_codes)
        x2 = self.w2(x, diag_codes)
        x = x1 + x2
        x = self.act3(x, codes)

        x1 = self.conv3(x, diag_codes)
        x2 = self.w3(x, diag_codes)
        x = x1 + x2

        x = einops.rearrange(x, "b c x y -> b (x y) c")
        x = self.fc1(x, diag_codes)
        x = self.act4(x, codes)
        x = self.fc2(x, diag_codes)
        x = einops.rearrange(x, "b (x y) c -> b c x y", x = size_x, y = size_y)
        return 0.0005 * x

    def get_grid(self, shape, device):
        if len(shape) == 4:
            batchsize, c, size_x, size_y = shape[0], shape[1], shape[2], shape[3]
            gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
            gridx = gridx.reshape(1, 1, size_x, 1).repeat(
                [batchsize, 1, 1, size_y])
            gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
            gridy = gridy.reshape(1, 1, 1, size_y).repeat(
                [batchsize, 1, size_x, 1])
            return torch.cat((gridx, gridy), dim=1).to(device)
    
def get_nn_model(dataset_name, in_dim, out_dim, factor, codes):
    if dataset_name in ['lv', 'pendulum']:
        model = MLP(in_dim, out_dim, factor, codes)
    elif dataset_name == 'burgers':
        model = CNN1D(in_dim, out_dim, 7, factor, codes)
    elif dataset_name == 'gs':
        model = CNN2D(in_dim, out_dim, 3, factor, codes)
    elif dataset_name == "kolmo":
        model = FNO2d(10, 10, out_dim, in_dim, factor, codes)
    return model