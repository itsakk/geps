from geps.model.layers import *
from geps.model.activations import *
import torch
import torch.nn as nn
import einops

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, factor, codes):
        super().__init__()

        self.codes = codes
        self.code_dim = self.codes.shape[-1]
        self.linear1 = GEPSLinear(in_dim, out_dim, True, self.code_dim)
        self.act1 = Swish(self.code_dim, factor)
        self.linear2 = GEPSLinear(out_dim, out_dim, True, self.code_dim)
        self.act2 = Swish(self.code_dim, factor)
        self.linear3 = GEPSLinear(out_dim, out_dim, True, self.code_dim)
        self.act3 = Swish(self.code_dim, factor)
        self.linear4 = GEPSLinear(out_dim, in_dim, True, self.code_dim)

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
        self.conv1 = GEPSConv1D(in_dim, out_dim, kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)
        self.act1 = Swish(self.code_dim, factor)
        self.conv2 = GEPSConv1D(out_dim, out_dim, kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)
        self.act2 = Swish(self.code_dim, factor)
        self.conv3 = GEPSConv1D(out_dim, out_dim, kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)
        self.act3 = Swish(self.code_dim, factor)
        self.conv4 = GEPSConv1D(out_dim, in_dim, kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)

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
        self.conv1 = GEPSConv2D(in_dim, out_dim, kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)
        self.act1 = Swish(self.code_dim, factor)
        self.conv2 = GEPSConv2D(out_dim, out_dim, kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)
        self.act2 = Swish(self.code_dim, factor)
        self.conv3 = GEPSConv2D(out_dim, out_dim, kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)
        self.act3 = Swish(self.code_dim, factor)
        self.conv4 = GEPSConv2D(out_dim, in_dim, kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)

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

class FNO1d(nn.Module):
    def __init__(self, modes, width, num_channels, factor, codes):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """
        self.codes = codes
        self.code_dim = self.codes.shape[-1]
        self.modes1 = modes
        self.width = width

        self.fc0 = GEPSLinear1D(1 + num_channels, self.width, True, self.code_dim, factor = factor)
        self.conv0 = GEPSSpectralConv1d_fast(self.width, self.width, self.modes1, code = self.code_dim, factor = factor)
        self.conv1 = GEPSSpectralConv1d_fast(self.width, self.width, self.modes1, code = self.code_dim, factor = factor)
        self.conv2 = GEPSSpectralConv1d_fast(self.width, self.width, self.modes1, code = self.code_dim, factor = factor)
        self.conv3 = GEPSSpectralConv1d_fast(self.width, self.width, self.modes1, code = self.code_dim, factor = factor)
        self.conv4 = GEPSSpectralConv1d_fast(self.width, self.width, self.modes1, code = self.code_dim, factor = factor)
        self.conv5 = GEPSSpectralConv1d_fast(self.width, self.width, self.modes1, code = self.code_dim, factor = factor)
        self.conv6 = GEPSSpectralConv1d_fast(self.width, self.width, self.modes1, code = self.code_dim, factor = factor)
        self.conv7 = GEPSSpectralConv1d_fast(self.width, self.width, self.modes1, code = self.code_dim, factor = factor)


        self.w0 = GEPSConv1D(self.width, self.width, kernel_size=1, padding = 0, padding_mode='constant', code = self.code_dim, factor = factor)
        self.w1 = GEPSConv1D(self.width, self.width, kernel_size=1, padding = 0, padding_mode='constant', code = self.code_dim, factor = factor)
        self.w2 = GEPSConv1D(self.width, self.width, kernel_size=1, padding = 0, padding_mode='constant', code = self.code_dim, factor = factor)
        self.w3 = GEPSConv1D(self.width, self.width, kernel_size=1, padding = 0, padding_mode='constant', code = self.code_dim, factor = factor)
        self.w4 = GEPSConv1D(self.width, self.width, kernel_size=1, padding = 0, padding_mode='constant', code = self.code_dim, factor = factor)
        self.w5 = GEPSConv1D(self.width, self.width, kernel_size=1, padding = 0, padding_mode='constant', code = self.code_dim, factor = factor)
        self.w6 = GEPSConv1D(self.width, self.width, kernel_size=1, padding = 0, padding_mode='constant', code = self.code_dim, factor = factor)
        self.w7 = GEPSConv1D(self.width, self.width, kernel_size=1, padding = 0, padding_mode='constant', code = self.code_dim, factor = factor)


        self.fc1 = GEPSLinear1D(self.width, 128, True, self.code_dim, factor = factor)
        self.fc2 = GEPSLinear1D(128, num_channels, True, self.code_dim, factor = factor)

        self.act1 = Swish(self.code_dim, factor = factor)
        self.act2 = Swish(self.code_dim, factor = factor)
        self.act3 = Swish(self.code_dim, factor = factor)
        self.act4 = Swish(self.code_dim, factor = factor)
        self.act5 = Swish(self.code_dim, factor = factor)
        self.act6 = Swish(self.code_dim, factor = factor)
        self.act7 = Swish(self.code_dim, factor = factor)
        self.act8 = Swish(self.code_dim, factor = factor)

    def forward(self, x, env):
        grid = self.get_grid(x.shape, x.device)
        codes = self.codes[env.long()]
        diag_codes = torch.diag_embed(codes)

        x = torch.cat((x, grid), dim=1)
        x = einops.rearrange(x, "b c x -> b x c")
        x = self.fc0(x, diag_codes)
        x = einops.rearrange(x, "b x c -> b c x")

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
        x = self.act4(x, codes)

        x1 = self.conv4(x, diag_codes)
        x2 = self.w4(x, diag_codes)
        x = x1 + x2
        x = self.act5(x, codes)

        x1 = self.conv5(x, diag_codes)
        x2 = self.w5(x, diag_codes)
        x = x1 + x2
        x = self.act6(x, codes)

        x1 = self.conv6(x, diag_codes)
        x2 = self.w6(x, diag_codes)
        x = x1 + x2
        x = self.act7(x, codes)

        x1 = self.conv7(x, diag_codes)
        x2 = self.w7(x, diag_codes)
        x = x1 + x2
        x = self.act8(x, codes)

        x = einops.rearrange(x, "b c x -> b x c")
        x = self.fc1(x, diag_codes)
        x = self.act4(x, codes)
        x = self.fc2(x, diag_codes)
        x = einops.rearrange(x, "b x c -> b c x")
        return x
        
    def get_grid(self, shape, device):
        batchsize, _, size_x = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x).repeat(
            [batchsize, 1, 1])
        return gridx.to(device)
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
        self.fc0 = GEPSLinear1D(2 + num_channels, self.width, True, self.code_dim)

        self.conv0 = GEPSSpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2, self.code_dim)
        self.conv1 = GEPSSpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2, self.code_dim)
        self.conv2 = GEPSSpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2, self.code_dim)
        self.conv3 = GEPSSpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2, self.code_dim)
        
        self.w0 = GEPSConv2D(self.width, self.width, self.kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)
        self.w1 = GEPSConv2D(self.width, self.width, self.kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)
        self.w2 = GEPSConv2D(self.width, self.width, self.kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)
        self.w3 = GEPSConv2D(self.width, self.width, self.kernel_size, self.padding, padding_mode = "circular", bias = True, code = self.code_dim)

        self.fc1 = GEPSLinear1D(self.width, 128, True, self.code_dim)
        self.fc2 = GEPSLinear1D(128, num_channels, True, self.code_dim)

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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        GEPSConv1D(in_channels, out_channels, kernel_size= 3, padding = 1, padding_mode='circular', code = self.code_dim),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        GEPSConv1D(in_channels, out_channels, kernel_size= 3, padding = 1, padding_mode='circular', code = self.code_dim),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, code_dim=None):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = GEPSConv1D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, padding_mode='circular', code=code_dim)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
        self.conv2 = GEPSConv1D(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular', code=code_dim)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = downsample
        self.stride = 1

    def forward(self, x, codes):
        residual = x
        
        out = self.conv1(x, codes)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out, codes)
        out = self.bn2(out)
        
        if self.downsample:
            residual = self.downsample(x, codes)
        
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, codes=None):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.codes = codes
        self.code_dim = self.codes.shape[-1]

        self.conv1 = GEPSConv1D(1, 64, kernel_size=7, stride=1, padding=3, padding_mode='circular', code=self.code_dim)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        
        # Defining the layers manually
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[3], stride=1)
        
        self.avgpool = nn.AvgPool1d(7, stride=1)
        self.conv2 = GEPSConv1D(256, 1, kernel_size=7, stride=1, padding=3, padding_mode='circular', code=self.code_dim)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []

        # Create new convolution and batch norm layers for downsampling
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample_conv = GEPSConv1D(self.inplanes, planes, kernel_size=3, stride=stride, padding_mode='circular', code=self.code_dim).cuda()
            downsample_bn = nn.BatchNorm1d(planes).cuda()
            downsample = lambda x, codes: downsample_bn(downsample_conv(x, codes))
        
        # First block with downsample
        layers.append(block(self.inplanes, planes, stride, downsample, code_dim=self.code_dim))
        self.inplanes = planes
        
        # Remaining blocks (no downsample needed)
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, code_dim=self.code_dim))

        return nn.ModuleList(layers)

    def forward(self, x, env):
        codes = self.codes[env.long()]
        codes = torch.diag_embed(codes)
        
        x = self.conv1(x, codes)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Iterate through layers
        for layer in self.layer0:
            x = layer(x, codes)
        for layer in self.layer1:
            x = layer(x, codes)
        for layer in self.layer2:
            x = layer(x, codes)
        for layer in self.layer3:
            x = layer(x, codes)
        x = F.relu(x)
        x = self.conv2(x, codes)
        return 0.0005 * x

def get_nn_model(dataset_name, in_dim, out_dim, factor, codes):
    if dataset_name in ['lv', 'pendulum']:
        model = MLP(in_dim, out_dim, factor, codes)
    elif dataset_name == 'burgers':
        model = CNN1D(in_dim, out_dim, 7, factor, codes)
    elif (dataset_name == 'gs') or (dataset_name == 'gs_multi'):
        model = CNN2D(in_dim, out_dim, 3, factor, codes)
    elif dataset_name == "kolmo":
        model = FNO2d(10, 10, out_dim, in_dim, factor, codes)
    elif dataset_name == 'combined':
        model = CNN1D(in_dim, out_dim, 7, factor, codes)
    elif dataset_name == 'combined_big':
        model = ResNet(ResidualBlock, [3, 4, 6, 3], codes)
    return model