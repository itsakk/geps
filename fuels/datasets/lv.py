import torch
from torch.utils.data import Dataset
from scipy.integrate import solve_ivp
from functools import partial
import numpy as np
import shelve

class LotkaVolterraDataset(Dataset):
    def __init__(self, path, n_data_per_env, t_horizon, params, dt, random_influence=0.2, method='RK45', group='train',
                 rdn_gen=1.):
        super().__init__()
        self.n_data_per_env = n_data_per_env
        self.num_env = len(params)
        self.len = n_data_per_env * self.num_env
        self.t_horizon = float(t_horizon)
        self.dt = dt
        self.random_influence = random_influence
        self.params_eq = params
        self.group = group
        self.max = np.iinfo(np.int32).max
        self.indices = [list(range(env * n_data_per_env, (env + 1) * n_data_per_env)) for env in range(self.num_env)]
        self.method = method
        self.rdn_gen = rdn_gen
        self.data = shelve.open(path)

    def _f(self, t, x, env=0):
        alpha = self.params_eq[env]['alpha']
        beta = self.params_eq[env]['beta']
        gamma = self.params_eq[env]['gamma']
        delta = self.params_eq[env]['delta']
        d = np.zeros(2)
        d[0] = alpha * x[0] - beta * x[0] * x[1]
        d[1] = delta * x[0] * x[1] - gamma * x[1]
        return d

    def _get_init_cond(self, seed):
        np.random.seed(seed if self.group == 'train' else self.max - seed)
        return np.random.random(2) + self.rdn_gen

    def __getitem__(self, index):
        env = index // self.n_data_per_env
        env_index = index % self.n_data_per_env
        t = torch.arange(0, self.t_horizon, self.dt).float()
        out = {'t': t, 'env': env}
        if self.data.get(str(index)) is None:
            y0 = self._get_init_cond(env_index)
            y = solve_ivp(partial(self._f, env=env), (0., self.t_horizon), y0=y0, method=self.method,
                          t_eval=np.arange(0., self.t_horizon, self.dt))
            y = torch.from_numpy(y.y).float()
            out['states'] = y
            self.data[str(index)] = y.numpy()
        else:
            out['states'] = torch.from_numpy(self.data[str(index)])

        out['index'] = index
        out['param'] = torch.tensor(list(self.params_eq[env].values()))
        return out

    def __len__(self):
        return self.len