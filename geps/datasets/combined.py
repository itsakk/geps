import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import einops
import h5py
from itertools import product
from torch.utils.data import Dataset, DataLoader

class CombinedDataset(Dataset):
    def __init__(self, file_name, group, dt):
        self.file_name = file_name
        self.group = group
        self.dt = int(1 / dt)

        self.file = h5py.File(self.file_name, 'r')
        self.length = len(self.file[f'{self.group}/states'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        state = torch.tensor(self.file[f'{self.group}/states'][idx], dtype=torch.float32)[..., ::self.dt]
        env = torch.tensor(self.file[f'{self.group}/env'][idx], dtype=torch.float32)
    
        t = torch.linspace(0, 1, state.shape[-1])
        return {'states': state, 't': t, 'env': env}
    
def psdiff(x, order=1, period=None, axis=-1):
    """
    Calculate the n-th order difference along given axis in the frequency domain.
    
    Args:
    x (torch.Tensor): Input tensor
    order (int): The order of the difference (default is 1)
    period (float): The period of the Fourier transform (default is 2*pi)
    axis (int): Axis along which the diff is taken (default is -1)
    
    Returns:
    torch.Tensor: The n-th order difference in the frequency domain
    """
    if period is None:
        period = 2 * torch.pi

    # Compute FFT
    ft = torch.fft.fft(x, dim=axis)
    
    # Generate frequency array
    n = x.shape[axis]
    freq = torch.fft.fftfreq(n, d=1.0/n)
    
    # Reshape freq to match the dimensionality of ft
    shape = [1] * len(x.shape)
    shape[axis] = -1
    freq = freq.reshape(shape)
    
    # Compute the multiplier
    multiplier = (2j * torch.pi * freq / period) ** order
    
    # Apply the multiplier
    ft_diff = ft * multiplier.cuda()
    
    # Inverse FFT
    return torch.fft.ifft(ft_diff, dim=axis).real

def initial_conditions(seed, N, L):
    x = np.linspace(0, (1 - 1.0 / N) * L, N)[:, None]
    np.random.seed(seed)
    A, phi, l = params()
    u = np.sum(A * np.sin(2 * np.pi * l * x / L + phi), -1)
    return torch.tensor(u).cuda()

def params():
    N = 3
    lmin, lmax = 1, 3
    A = np.random.rand(1, N) - 0.5
    phi = 2.0 * np.pi * np.random.rand(1, N)
    l = np.random.randint(lmin, lmax, (1, N))
    return A, phi, l

def generate_multiple_ics(seeds, N, L):
    ics = []
    results = [initial_conditions(seed, N, L) for seed in seeds]
    for result in results:
        ics.append(result)
    return torch.stack(ics)

class Combined_equation(nn.Module):
    """
    The Korteweg-de Vries equation:
    ut + 0.5*u**2 + uxxx = 0

    Advection:
    ut + beta u_x = 0

    The Kuramoto-Sivashinsky equation:
    ut + (0.5*u**2 + ux + nu*uxxx)x = 0

    The heat equation 
    ut - nu * uxx = 0
    which we use to get data for the Burgers' equation via the Cole-Hopf transformation

    Combined equation:
    ut - (alpha * u***2 + beta *ux + gamma * uxx * delta uxxx)x
    """
    def __init__(self,
                 alpha=None,
                 beta=None,
                 delta=None,
                 gamma=None,
                 L=None):
        super().__init__()
        self.alpha=alpha
        self.beta=beta
        self.delta=delta
        self.gamma=gamma
        self.L=L
        
    def update_parameters(self, alpha, beta, delta, gamma):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma

    def _f(self, t, u):
        if (self.beta == 0) & (self.delta == 0) & (self.gamma == 0):
            ux = psdiff(u, period=self.L)
        else: 
            ux = u * psdiff(u, period=self.L)
        uxx = psdiff(u, order=2, period=self.L)
        uxxx = psdiff(u, order=3, period=self.L)
        uxxxx = psdiff(u,order=4, period=self.L)
        dudt = - self.alpha*ux + self.beta*uxx - self.delta*uxxx - self.gamma*uxxxx
        return dudt

def solve_pde(model_phy, y0, time_horizon, num_ts):
    states = odeint(model_phy._f, y0=y0, t=torch.linspace(0., time_horizon, num_ts).cuda(), method = 'scipy_solver', options = dict(solver = 'Radau'))
    states = states.unsqueeze(1).float()
    states = einops.rearrange(states, 't c b x -> b c x t')
    return states.numpy(force = True)

def main(num_samples, group='train', file_name='combined.h5'):
    
    # temporal horizon and delta t
    num_ts = 100
    N = 128
    L = 32
    time_horizon = 30
    # Number of desired environments
    num_envs = 1200 + 10
    # Sampled ranges for alphas, beta, delta, gamma
    np.random.seed(42)
    alpha= np.linspace(0.5, 1, num = 11)
    beta= np.linspace(0.0, 0.5, num = 11)
    delta= np.linspace(0.0, 1., num = 11)
    gamma = np.linspace(0.0, 1., num = 11)

    # Generate the Cartesian product
    all_combinations = list(product(alpha, beta, delta, gamma))

    # Randomly sample 1200 unique combinations
    sampled_combinations = np.random.choice(len(all_combinations), num_envs, replace=False)
    params = [all_combinations[i] for i in sampled_combinations]
    print(len(params))

    params = params[1200:]

    if group == 'train':
        seed = 0
    elif group == 'val':
        seed = 1000000
    elif group == 'test':
        seed = 2000000

    model_phy = Combined_equation(alpha = 0, beta = 0, gamma = 0,  delta = 0, L = L)

    # Precompute the total number of samples
    total_samples = len(params) * num_samples

    h5f = h5py.File(file_name, "a")
    dataset = h5f.create_group(group)

    h5f_states = dataset.create_dataset("states", (total_samples, 1, N, num_ts), dtype=np.float32)
    h5f_alphas = dataset.create_dataset("alpha", (total_samples,), dtype=np.float32)
    h5f_betas = dataset.create_dataset("beta", (total_samples,), dtype=np.float32)
    h5f_gammas = dataset.create_dataset("gamma", (total_samples,), dtype=np.float32)
    h5f_deltas = dataset.create_dataset("delta", (total_samples,), dtype=np.float32)
    h5f_seeds = dataset.create_dataset("seed", (total_samples,), dtype=np.int32)
    h5f_envs = dataset.create_dataset("env", (total_samples,), dtype=np.int32)

    sample_idx = 0
    batch = num_samples

    for j, param_set in enumerate(params):
        print(j)
        alpha, beta, delta, gamma = param_set
        model_phy.update_parameters(alpha, beta, delta, gamma)
  

        for i in range(0, num_samples, batch):
            y0 = generate_multiple_ics(np.arange(seed, seed + batch), N, L)
            states = solve_pde(model_phy, y0, time_horizon, num_ts)

            end_idx = sample_idx + batch
            # Ensure we do not exceed the total_samples boundary
            if end_idx > total_samples:
                end_idx = total_samples
                batch = end_idx - sample_idx

            # Store the trajectory and parameters in the structured array
            h5f_states[sample_idx:end_idx] = states[:batch]  # Ensure correct slicing
            h5f_alphas[sample_idx:end_idx] = alpha
            h5f_betas[sample_idx:end_idx] = beta
            h5f_deltas[sample_idx:end_idx] = delta
            h5f_gammas[sample_idx:end_idx] = gamma
            h5f_envs[sample_idx:end_idx] = j
            h5f_seeds[sample_idx:end_idx] = np.arange(seed, seed + batch)

            sample_idx = end_idx  # Move to the next index
            seed += batch  # Increment seed by batch size

    h5f.close()
    print(f"Data saved to {file_name}")

if __name__ == "__main__":
    main(10, group='val', file_name='/home/kassai/code/FUELS/exp_gen/adapt/combined_big/combined_big_val.h5')