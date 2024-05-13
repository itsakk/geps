import shelve
import torch
import einops

import numpy as np
import scipy as sp
import torch.nn.functional as F

from torch.utils.data import Dataset
from torchdiffeq import odeint

MAX = np.iinfo(np.int32).max

def box_filter(signal,kernel_size):
        kernel = torch.ones(kernel_size).cuda() / kernel_size
        filtered_states = F.conv1d(signal, kernel.unsqueeze(0).unsqueeze(0), stride = kernel_size)
        return filtered_states

class Burgers(Dataset):

    def __init__(self, n_data_per_env, t_horizon, params, dt_filt=1e-3, 
                 N=16384, N_filt=256, path=None, group='train'):
        super().__init__()
        self.n_data_per_env = n_data_per_env
        self.time_horizon = t_horizon
        self.num_env = len(params)
        self.dt_filt = dt_filt
        self.N = N
        self.N_filt = N_filt
        self.len = n_data_per_env * self.num_env
        self.indices = [list(range(env * n_data_per_env, (env + 1) * n_data_per_env)) for env in range(self.num_env)]
        self.params_eq = params
        self.test = (group == 'test')
        self.max = torch.iinfo(torch.int32).max
        self.data = shelve.open(path)

    def _derive1(self, a):
        return(
                torch.roll(a, 1)
                -a
            )

    def _mean_f(self, a):
        return(
                torch.roll(a, 1)
                +a
            )/2

    def _laplacian1D(self, a):
        return (
                - 2 * a
                + torch.roll(a, 1) 
                + torch.roll(a, -1)
            ) / (self.fdx ** 2)

    def _laplacian1Dbis(self, a):
        return (
            + 2 * a
            + torch.roll(a,+1) 
            + torch.roll(a,-1)
            ) / (self.fdx ** 2)

    def _fluxRight(self, a):
        phi1 = torch.abs(self._laplacian1D(a))/torch.abs(self._laplacian1Dbis(a))
        phi2 = torch.roll(phi1,+1)
        k2 = 1
        epsilon2 = k2*torch.maximum(phi1,phi2) 
        epsilon4 = torch.maximum(torch.tensor(0),1/60 - 1/5*epsilon2)
        return(
                + self._mean_f(1/2*a**2)
                - 1/6*self._derive1(self._derive1(self._mean_f(1/2*a**2)))
                + 1/30*self._derive1(self._derive1(self._derive1(self._derive1(self._mean_f(1/2*a**2)))))
                - torch.abs(self._mean_f(a)) *(epsilon2*self._derive1(a) + epsilon4*(self._derive1(self._derive1(self._derive1(self._derive1(self._derive1(a)))))))
        )

    def exp_forcing(self, w = 4, F = 1):
        return F * torch.exp(-w * self.x ** 2)
    

    def periodic_forcing(self, w = 4, t = 0, F = 1):
        return F * (torch.sin(w * self.x) + torch.cos(w * t))

    def forcing(self, force_func, U, t, w, F):
        if force_func == 'sin':
            return self.periodic_forcing(w, t, F)
        elif force_func == 'exp':
            return self.exp_forcing(w, F)
        elif force_func == 'rand':
            return self.randn_forcing(U, F)
        else:
            return 0
        
    def _f(self, t, x):
        return -(self._fluxRight(x)-self._fluxRight(torch.roll(x, -1)))/self.fdx + self.mu * self._laplacian1D(x) + self.forcing(self.force_func, x, t, self.w, self.F)
    
    def energy_spectrum(self, N, fdx, k0 = 5):
        ku = 2*np.pi*sp.fft.fftfreq(N, fdx)
        ku = 2/3*np.pi**(-1/2)*(ku/k0)**4*1/k0*np.exp(-(ku/k0)**2)
        ku[np.abs(ku)<1e-34] = 0
        return ku

    def _get_initial_condition(self, index, N, fdx):
        np.random.seed(index if not self.test else self.max - index)
        energy = self.energy_spectrum(N, fdx) # 1e6
        random_psi = np.random.rand(N//2-1)
        random_psi = np.concatenate((np.array([0]),random_psi,np.random.rand(1),-random_psi[::-1]))
        random_phases = np.exp(2 * np.pi * 1j * random_psi)
        velocities = N*(2*energy)**(1/2)*random_phases
        u_result = sp.fft.ifft(velocities).real
        return u_result

    def __getitem__(self, index):
        env = index // self.n_data_per_env
        env_index = index % self.n_data_per_env

        if self.data.get(str(index)) is None:
            self.mu = self.params_eq[env]['mu']
            self.F = self.params_eq[env]['F']
            self.w = self.params_eq[env]['w']
            self.force_func = self.params_eq[env]['force']
            self.fdx = (self.params_eq[env]['domain'] * np.pi) / self.N
            self.x = torch.linspace(0, self.params_eq[env]['domain'] * np.pi, self.N).cuda()
            print(f'generating {env},{env_index}')
            
            y0 = self._get_initial_condition(env_index, self.N, self.fdx)
            y0 = torch.from_numpy(y0).cuda()
            states = odeint(self._f, y0=y0, t=torch.arange(0., self.time_horizon, self.dt_filt).cuda(), method= 'scipy_solver', options = dict(solver = 'RK45')).cuda()

            states = states.unsqueeze(1).float()

            states = box_filter(signal = states, kernel_size= self.N // self.N_filt)
            states = einops.rearrange(states, 't c x -> c x t').cpu().detach()
            self.data[str(index)] = states
        else:
            states = self.data[str(index)]
        return {'states': states, 't': torch.arange(0, self.time_horizon, self.dt_filt).float(), 'env': env}
    
    def __len__(self):
        return int(self.len)