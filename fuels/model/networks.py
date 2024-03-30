from fuels.model.layers import *
import torch
import torch.nn as nn

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
        self.conv1 = ROCAConv1D(in_dim, out_dim, kernel_size, self.padding, padding_mode = "constant", bias = True, code = self.code_dim)
        self.act1 = Swish(self.code_dim, factor)
        self.conv2 = ROCAConv1D(out_dim, out_dim, kernel_size, self.padding, padding_mode = "constant", bias = True, code = self.code_dim)
        self.act2 = Swish(self.code_dim, factor)
        self.conv3 = ROCAConv1D(out_dim, out_dim, kernel_size, self.padding, padding_mode = "constant", bias = True, code = self.code_dim)
        self.act3 = Swish(self.code_dim, factor)
        self.conv4 = ROCAConv1D(out_dim, in_dim, kernel_size, self.padding, padding_mode = "constant", bias = True, code = self.code_dim)

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

def get_nn_model(dataset_name, in_dim, out_dim, factor, codes):

    if dataset_name in ['lv', 'pendulum']:
        model = MLP(in_dim, out_dim, factor, codes)
    elif dataset_name == 'burgers':
        model = CNN1D(in_dim, out_dim, 7, factor, codes)
    elif dataset_name == 'gs':
        model = CNN2D(in_dim, out_dim, 3, factor, codes)
    elif dataset_name == 'kolmo':
        model = CNN2D(in_dim, out_dim, 3, factor, codes)
    return model