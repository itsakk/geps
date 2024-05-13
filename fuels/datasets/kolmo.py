import torch
import shelve
from torch.utils.data import Dataset
from torchdiffeq import odeint
import scipy as sp
import numpy as np
import torch.nn.functional as F
import einops

MAX = torch.iinfo(torch.int32).max

def filter(signal, N, output_size):
    new_signal = torch.zeros((output_size, output_size))
    factor = int(signal.shape[0] / output_size)
    for i in range(output_size):
        for j in range(output_size):
            new_signal[i][j] = torch.mean(signal[i*factor:(i+1)*factor, j*factor:(j+1)*factor])
    return new_signal

class Turb2d(Dataset):

    def __init__(self, n_data_per_env, t_horizon, params, dt=0.002, dt_filt=0.02, warmup = 20, 
                 N=1024, N_filt=256, path=None, group='train'):
        super().__init__()
        self.n_data_per_env = n_data_per_env
        self.time_horizon = t_horizon
        self.num_env = len(params)
        self.dt = dt
        self.dt_filt = dt_filt
        self.warmup = warmup
        self.N = N
        self.N_filt = N_filt
        self.len = n_data_per_env * self.num_env
        self.indices = [list(range(env * n_data_per_env, (env + 1) * n_data_per_env)) for env in range(self.num_env)]
        self.params_eq = params
        self.test = (group == 'test')
        self.max = torch.iinfo(torch.int32).max
        self.data = shelve.open(path)

    def energy_spectrum(self, N, fdx, k0 = 10):

        kx = 2 * np.pi * sp.fft.fftfreq(N, fdx)
        ky = 2 * np.pi * sp.fft.fftfreq(N, fdx)

        kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')
        kzz = np.sqrt(kxx**2 + kyy**2)
        energy = 4/3*np.pi**(-1/2)*(kzz/k0)**4*1/k0*np.exp(-(kzz/k0)**2)
        energy[np.abs(energy)<1e-34] = 0
        return energy, kzz

    def _get_initial_condition(self, index, N, fdx):
        np.random.seed(index if not self.test else self.max - index)
        energy, kzz = self.energy_spectrum(N, fdx)
        ## axis psi
        random_psi = np.zeros((N,N))

        if N%2 == 1:
            random_psi_x = np.random.rand(N//2,N//2)
            random_psi_y = np.random.rand(N//2,N//2)

            random_psi[1:N//2+1,1:N//2+1] = random_psi_x+random_psi_y
            random_psi[:N//2:-1,1:N//2+1] = -random_psi_x+random_psi_y
            random_psi[1:N//2+1,:N//2:-1] = random_psi_x-random_psi_y
            random_psi[:N//2:-1,:N//2:-1] = -random_psi_x-random_psi_y
        else:
            random_psi_x = np.random.rand(N//2-1,N//2-1)
            random_psi_y = np.random.rand(N//2-1,N//2-1)

            random_psi[1:N//2,1:N//2] = random_psi_x+random_psi_y
            random_psi[:N//2:-1,1:N//2] = -random_psi_x+random_psi_y
            random_psi[1:N//2,:N//2:-1] = random_psi_x-random_psi_y
            random_psi[:N//2:-1,:N//2:-1] = -random_psi_x-random_psi_y
        
        random_phases = np.exp(2 * np.pi * 1j * random_psi)
        velocities = N*N*(kzz*energy/np.pi)**(1/2)*random_phases
        u_result = sp.fft.ifft2(velocities).real
        return u_result
    
    def ps(self, N, dx, w):
        w = w.unsqueeze(0).unsqueeze(0)
        w_pad = F.pad(w, pad = (0, 1, 0, 1), mode = "circular").squeeze(0).squeeze(0)
        wf = torch.fft.fft2(1/N**2*w_pad)

        kxx, kyy = torch.meshgrid(2*np.pi*torch.fft.fftfreq(self.N+1, self.fdx, device="cuda"), 2*np.pi*torch.fft.fftfreq(self.N+1, self.fdx, device="cuda"))
        uf = -wf/(1e-12 + kxx**2 + kyy**2)
        uf[0,0] = 0
        return (torch.fft.ifft2(N**2*uf).real)[:self.N,:self.N]

    def derive_x(self, a):
        return (
            torch.roll(a, -1, dims=1)
            - torch.roll(a, +1, dims=1)
        )

    def derive_y(self, a):
        return (
            torch.roll(a, -1, dims=0)
            - torch.roll(a, +1, dims=0)
        )
    
    def laplacian2D(self, a):
        return (
            -4 * a
            + torch.roll(a, +1, dims=0)
            + torch.roll(a, -1, dims=0)
            + torch.roll(a, +1, dims=1)
            + torch.roll(a, -1, dims=1)
        ) / (self.fdx ** 2)

    def sin_forcing(self, U, k=4, F=1):
        return F * (torch.sin(k * self.yy) * self.xx - 0.1 * U)

    def exp_forcing(self, k=4, F=1):
        return F * (torch.exp(-k * self.yy ** 2))

    def periodic_forcing(self, k=4, t=0, F=1):
        return F * (torch.sin(k * self.xx) + torch.cos(k * t))

    def forcing(self, force_func, U, t, k, F=1):
        if force_func == 'sin':
            return self.sin_forcing(U, k, F)
        elif force_func == 'exp':
            return self.exp_forcing(k, F)
        elif force_func == 'periodic':
            return self.periodic_forcing(k, t, F)
        else:
            return 0

    def _f(self, t, x):
        x = x.reshape(self.N, self.N)
        psi = self.ps(self.N, self.fdx, -x)

        J1 = (self.derive_y(psi) * self.derive_x(x) - self.derive_x(psi) * self.derive_y(x)) / (4 * self.fdx ** 2)
        J2 = (torch.roll(x, -1, dims=1) * self.derive_y(torch.roll(psi, -1, dims=1)) -
              torch.roll(x, 1, dims=1) * self.derive_y(torch.roll(psi, 1, dims=1)) -
              torch.roll(x, -1, dims=0) * self.derive_x(torch.roll(psi, -1, dims=0)) +
              torch.roll(x, 1, dims=0) * self.derive_x(torch.roll(psi, 1, dims=0))) / (4 * self.fdx ** 2)
        J3 = (torch.roll(torch.roll(x, -1, dims=1), -1, dims=0) * (torch.roll(psi, -1, dims=0) - torch.roll(psi, -1, dims=1)) -
              torch.roll(torch.roll(x, 1, dims=1), 1, dims=0) * (torch.roll(psi, 1, dims=1) - torch.roll(psi, 1, dims=0)) -
              torch.roll(torch.roll(x, 1, dims=1), -1, dims=0) * (torch.roll(psi, -1, dims=0) - torch.roll(psi, 1, dims=1)) +
              torch.roll(torch.roll(x, -1, dims=1), 1, dims=0) * (torch.roll(psi, -1, dims=1) - torch.roll(psi, 1, dims=0))) / (4 * self.fdx ** 2)

        return (-(J1 + J2 + J3) / 3 + self.mu * self.laplacian2D(x)) + self.forcing(self.force_func, x, t, self.k)

    def __getitem__(self, index):
        env = index // self.n_data_per_env
        env_index = index % self.n_data_per_env

        if self.data.get(str(index)) is None:

            domain_size = self.params_eq[env]['domain'] * torch.pi

            self.fdx = domain_size / self.N
            self.mu = self.params_eq[env]['mu']
            self.force_func = self.params_eq[env]['force']
            self.k = self.params_eq[env]['k']

            x = torch.linspace(0, domain_size, self.N).cuda()
            y = torch.linspace(0, domain_size, self.N).cuda()
            self.xx, self.yy = torch.meshgrid(x, y, indexing = 'ij')

            print(f'generating {env},{env_index}')
            y0 = self._get_initial_condition(env_index, self.N, self.fdx)
            y0 = torch.Tensor(y0[:self.N, :self.N]).cuda()
            states = odeint(self._f, y0=y0, t=torch.arange(0., self.time_horizon, self.dt_filt).cuda(), method= 'rk4', options = dict(step_size = self.dt)).cuda()

            states = states[self.warmup:]
            new_state = torch.zeros((states.shape[0], self.N_filt, self.N_filt)).cuda()
            for i in range(states.shape[0]):
                new_state[i] = filter(states[i], N=self.N, output_size=self.N_filt)
                
            states = new_state
            states = einops.rearrange(states, "t n m -> n m t").cpu().detach().unsqueeze(0)
            self.data[str(index)] = states
        else:
            states = self.data[str(index)]
        return {'states': states, 't': torch.arange(0, self.time_horizon - self.warmup * self.dt_filt, self.dt_filt).float(), 'env': env}

    def __len__(self):
        return int(self.len)