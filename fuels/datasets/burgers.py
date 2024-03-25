import numpy as np
import shelve
from torch.utils.data import Dataset
from scipy.integrate import solve_ivp
import torch
from collections import OrderedDict
import scipy as sp
from functools import partial
import shelve

MAX = np.iinfo(np.int32).max

def box_filter(signal,kernel_size):
        new = np.zeros(int(signal.shape[0]/kernel_size))
        for i in range(int(signal.shape[0]/kernel_size)-1):
               new[i] = np.mean(signal[i*kernel_size:(i+1)*kernel_size])
        return new

class BurgersF(Dataset):

    def __init__(self, n_data_per_env, t_horizon, params, dt, dx=2.,
                 path=None, method='RK45', group='train'):
        super().__init__()
        self.n_data_per_env = n_data_per_env
        self.num_env = len(params)
        self.time_horizon = t_horizon
        self.dx = dx
        self.len = n_data_per_env * self.num_env
        self.indices = [list(range(env * n_data_per_env, (env + 1) * n_data_per_env)) for env in range(self.num_env)]
        self.N = 16384
        self.fdx = 2 * np.pi /(self.N)
        self.num_env = len(params)
        self.params_eq = params
        self.test = (group == 'test')
        self.max = np.iinfo(np.int32).max
        self.data = shelve.open(path)
        self.dt = dt

    def _derive1(self,a):
        return(
            np.roll(a,1)
            -a
        )

    def _mean_f(self,a):
        return(
            np.roll(a,1)
            +a
        )/2

    def _laplacian1Dbis(self, a):
        return (
            + 2 * a
            + np.roll(a,+1) 
            + np.roll(a,-1)
            ) / (self.fdx ** 2)

    def _laplacian1D(self, a):
        return (
            - 2 * a
            + np.roll(a,+1) 
            + np.roll(a,-1)
        ) / (self.fdx ** 2)
    
    def _fluxRight(self, a):
        phi1 = np.abs(self._laplacian1D(a))/np.abs(self._laplacian1Dbis(a))
        phi2 = np.roll(phi1,+1)
        k2 = 1
        epsilon2 = k2*np.maximum(phi1,phi2) 
        epsilon4 = np.maximum(0,1/60 - 1/5*epsilon2)

        return(
                + self._mean_f(1/2*a**2)
                - 1/6*self._derive1(self._derive1(self._mean_f(1/2*a**2)))
                + 1/30*self._derive1(self._derive1(self._derive1(self._derive1(self._mean_f(1/2*a**2)))))
                - np.abs(self._mean_f(a)) *(epsilon2*self._derive1(a) + epsilon4*(self._derive1(self._derive1(self._derive1(self._derive1(self._derive1(a)))))))
            )
    
    def forcing(self, forcing_func, y0, t, F, w):
        if forcing_func == 'rand':
            force = F * np.random.randn(*y0.shape)
        elif forcing_func == 'exp':
            force = F * np.exp(-w*y0**2)
        elif forcing_func == "periodic":
            force = F * (np.sin(w*y0) + np.cos(w*t))
        else:
            force = 0
        return force
        
    def _f(self,t, x, env = 0):
        mu = self.params_eq[env]['mu']
        F_ = self.params_eq[env]['F']
        w = self.params_eq[env]['w']
        force_func = self.params_eq[env]['force']
        return  -(self._fluxRight(x)-self._fluxRight(np.roll(x,-1)))/self.fdx + mu*self._laplacian1D(x) + self.forcing(force_func, x, t, F_, w)
    
    def energy_spectrum(self, N,k0 = 5):
            ku = 2*np.pi*sp.fft.fftfreq(N,self.fdx)
            ku = 2/3*np.pi**(-1/2)*(ku/k0)**4*1/k0*np.exp(-(ku/k0)**2)
            ku[np.abs(ku)<1e-34] = 0
            return ku

    def _get_initial_condition(self, index):
        np.random.seed(index if not self.test else self.max - index)
        energy = self.energy_spectrum(self.N) # 1e6
        random_psi = np.random.rand(self.N//2-1)
        random_psi = np.concatenate((np.array([0]),random_psi,np.random.rand(1),-random_psi[::-1]))
        random_phases = np.exp(2 * np.pi * 1j * random_psi)
        velocities = self.N*(2*energy)**(1/2)*random_phases
        u_result = sp.fft.ifft(velocities).real
        return u_result
    
    def __getitem__(self, index):
        env = index // self.n_data_per_env
        env_index = index % self.n_data_per_env
        if self.data.get(str(index)) is None:
            print(f'generating {env},{env_index}')
            y0 = self._get_initial_condition(env_index)

            states = solve_ivp(partial(self._f, env = env), (0., self.time_horizon), y0=y0, method='RK45', t_eval=np.arange(0., self.time_horizon, 1e-5)).y

            #temporal sampling
            states = states[:,::int(1e5*self.dt)]
            states = states[:,:int(self.time_horizon/self.dt)]

            #spatial filtering
            states = np.apply_along_axis(box_filter,axis=0,arr=states,kernel_size=int(int(self.N)*self.dx))
            
            self.data[str(index)] = states
            states = torch.from_numpy(states).float()
            states = states.resize(1,states.size()[0],states.size()[1])
        else:
            states = torch.from_numpy(self.data[str(index)]).float()
            states = states.resize(1,states.size()[0],states.size()[1])
            
        return {'states': states, 't': torch.arange(0, self.time_horizon, self.dt).float(), 'env': env}
    
    def __len__(self):
        return int(self.len)