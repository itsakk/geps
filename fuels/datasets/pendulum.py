import numpy as np
import math, shelve
from torch.utils.data import Dataset
from scipy.integrate import solve_ivp
import torch

MAX = np.iinfo(np.int32).max

class DampedPendulum(Dataset):

    def __init__(self, path, ndata_per_env, time_horizon, dt, params, group='train'):
        super().__init__()
        self.time_horizon = float(time_horizon)  # total time
        self.dt = float(dt)  # time step
        self.num_env = len(params)
        self.len = ndata_per_env * self.num_env
        self.group = group
        self.ndata_per_env = ndata_per_env
        self.data = shelve.open(path)
        self.params = params
        self.indices = [list(range(env * ndata_per_env, (env + 1) * ndata_per_env)) for env in range(self.num_env)]

    def _get_pde_parameters(self, params, env):
        alpha = params[env]['alpha']
        T0 = params[env]['T0']
        omega0_square = (2 * math.pi / T0) ** 2
        return omega0_square, alpha

    def _f(self, t, x, omega0_square, alpha):  # coords = [q,p]
        q, p = np.split(x, 2)
        dqdt = p
        dpdt = -omega0_square * np.sin(q) - alpha * p
        return np.concatenate([dqdt, dpdt], axis=-1)

    def _get_initial_condition(self, seed):
        np.random.seed(seed if self.group == 'train' else MAX-seed)
        y0 = np.random.rand(2) * 2.0 - 1
        radius = np.random.rand() + 1.3
        y0 = y0 / np.sqrt((y0 ** 2).sum()) * radius
        return y0

    def __getitem__(self, index):
        t_eval = torch.from_numpy(np.arange(0, self.time_horizon, self.dt))
        env = index // self.ndata_per_env
        omega0_square, alpha = self._get_pde_parameters(self.params, env)

        if self.data.get(str(index)) is None:
            y0 = self._get_initial_condition(index)
            states = solve_ivp(fun=self._f, t_span=(0, self.time_horizon), args = (omega0_square, alpha), y0=y0, method='DOP853', t_eval=t_eval, rtol=1e-10).y
            self.data[str(index)] = states
            states = torch.from_numpy(states).float()
        else:
            states = torch.from_numpy(self.data[str(index)]).float()

        return {'states': states, 't': t_eval.float(),'env': env, 'index' : index, 'omega0_square' : omega0_square, 'alpha' : alpha}

    def __len__(self):
        return self.len
    



class DampedDrivenPendulum(Dataset):

    def __init__(self, path, ndata_per_env, time_horizon, dt, params, group='train'):
        super().__init__()
        self.time_horizon = float(time_horizon)  # total time
        self.dt = float(dt)  # time step
        self.num_env = len(params)
        self.len = ndata_per_env * self.num_env
        self.group = group
        self.ndata_per_env = ndata_per_env
        self.data = shelve.open(path)
        self.params = params
        self.indices = [list(range(env * ndata_per_env, (env + 1) * ndata_per_env)) for env in range(self.num_env)]

    def _get_pde_parameters(self, params, env):
        alpha = params[env]['alpha']
        w0 = params[env]['w0']
        wf = params[env]['wf']
        f0 = params[env]['f0']
        
        # w02 = w0 ** 2
        w02 = w0

        return w02, alpha, wf, f0

    def _f(self, t, x, w02, alpha, wf, f0):  # coords = [q,p]
        q, p = np.split(x, 2)
        dqdt = p
        dpdt = -w02 * np.sin(q) - alpha * p + f0 * np.cos(wf * t)                     
        return np.concatenate([dqdt, dpdt], axis=-1)

    def _get_initial_condition(self, seed):
        np.random.seed(seed if self.group == 'train' else MAX-seed)
        y0 = np.random.rand(2) #* 2.0 - 1
        radius = np.random.rand() + 1.3
        y0 = y0 / np.sqrt((y0 ** 2).sum()) * radius
        return y0 
        # return np.array([0.1,0])

    def __getitem__(self, index):
        t_eval = torch.from_numpy(np.arange(0, self.time_horizon, self.dt))
        env = index // self.ndata_per_env
        w02, alpha, wf, f0 = self._get_pde_parameters(self.params, env)

        if self.data.get(str(index)) is None:
            y0 = self._get_initial_condition(index)
            states = solve_ivp(fun=self._f, t_span=(0, self.time_horizon), args = (w02, alpha, wf,f0), y0=y0, method='DOP853', t_eval=t_eval, rtol=1e-10).y
            
            self.data[str(index)] = states
            states = torch.from_numpy(states).float()
        else:
            states = torch.from_numpy(self.data[str(index)]).float()

        return {'states': states, 't': t_eval.float(),'env': env, 'index' : index, 'w02' : w02, 'alpha' : alpha, 'wf' : wf, 'f0' : f0}

    def __len__(self):
        return self.len