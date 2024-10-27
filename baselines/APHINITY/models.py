import torch
import einops
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)
    
class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2),
                             x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(
                x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(
                x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, num_channels):
        super(FNO2d, self).__init__()

        self.num_channels = num_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2
        self.fc0 = nn.Linear(2 + num_channels, self.width)

        self.conv0 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)

        self.act0 = Swish()
        self.act1 = Swish()
        self.act2 = Swish()
        self.act3 = Swish()
        
    def forward(self, x):
        b, c, size_x, size_y = x.shape 
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        x = einops.rearrange(x, "b c x y-> b (x y) c")
        x = self.fc0(x)
        x = einops.rearrange(x, "b (x y) c -> b c x y", x = size_x, y = size_y)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.act0(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.act1(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.act2(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = einops.rearrange(x, "b c x y -> b (x y) c")
        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = einops.rearrange(x, "b (x y) c -> b c x y", x = size_x, y = size_y)
        return x

    def get_grid(self, shape, device):
        batchsize, c, size_x, size_y = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(torch.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat(
            [batchsize, 1, 1, size_y])
        gridy = torch.tensor(torch.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat(
            [batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)
    
class CNN2D(nn.Module):
    def __init__(self, state_c, hidden_c, kernel_size):

        super().__init__()
        self.padding = kernel_size // 2
        self.conv1 = nn.Conv2d(state_c, hidden_c, kernel_size, padding = self.padding, padding_mode = "circular", bias = True)
        self.act1 = Swish()
        self.conv2 = nn.Conv2d(hidden_c, hidden_c, kernel_size, padding = self.padding, padding_mode = "circular", bias = True)
        self.act2 = Swish()
        self.conv3 = nn.Conv2d(hidden_c, hidden_c, kernel_size, padding = self.padding, padding_mode = "circular", bias = True)
        self.act3 = Swish()
        self.conv4 = nn.Conv2d(hidden_c, state_c, kernel_size, padding = self.padding, padding_mode = "circular", bias = True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.act1(x1)
        x1 = self.conv2(x1)
        x1 = self.act2(x1)
        x1 = self.conv3(x1)
        x1 = self.act3(x1)
        x1 = self.conv4(x1)
        return 0.0005 * x1

class CNN1D(nn.Module):
    def __init__(self, state_c, hidden_c, kernel_size):
        super().__init__()

        self.padding = kernel_size // 2
        self.conv1 = nn.Conv1d(state_c, hidden_c, kernel_size, padding = self.padding, padding_mode = "circular", bias = True)
        self.act1 = Swish()
        self.conv2 = nn.Conv1d(hidden_c, hidden_c, kernel_size, padding = self.padding, padding_mode = "circular", bias = True)
        self.act2 = Swish()
        self.conv3 = nn.Conv1d(hidden_c, hidden_c, kernel_size, padding = self.padding, padding_mode = "circular", bias = True)
        self.act3 = Swish()
        self.conv4 = nn.Conv1d(hidden_c, state_c, kernel_size, padding = self.padding, padding_mode = "circular", bias = True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.act1(x1)
        x1 = self.conv2(x1)
        x1 = self.act2(x1)
        x1 = self.conv3(x1)
        x1 = self.act3(x1)
        x1 = self.conv4(x1)
        return 0.0005 * x1
    
class MLP(nn.Module):
    """
    Feed-forward neural network with context parameters.
    """

    def __init__(self,
                 in_dim,
                 out_dim,
                 ):
        super(MLP, self).__init__()

        # fully connected layers
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.lin3 = nn.Linear(out_dim, out_dim)
        self.lin4 = nn.Linear(out_dim, in_dim)

        self.act1 = Swish()
        self.act2 = Swish()
        self.act3 = Swish()

    def forward(self, x):
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = self.lin3(x)
        x = self.act3(x)
        x = self.lin4(x)
        return x
    
def get_nn_model(dataset_name, in_dim, out_dim):
    if dataset_name == 'pendulum':
        model = MLP(in_dim, out_dim)
    elif dataset_name in ['burgers', 'combined']:
        model = CNN1D(in_dim, out_dim, 7)
    elif dataset_name == 'gs':
        model = CNN2D(in_dim, out_dim, 3)
    elif dataset_name == "kolmo":
        model = FNO2d(10, 10, out_dim, in_dim)
    return model