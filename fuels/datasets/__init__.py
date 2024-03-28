import torch
import math
from torch.utils.data import DataLoader       
from fuels.utils import SubsetRamdomSampler, SubsetSequentialSampler
from fuels.datasets.pendulum import DampedDrivenPendulum
from fuels.datasets.lv import LotkaVolterraDataset
from fuels.datasets.gs import GrayScottDataset
from fuels.datasets.burgers import BurgersF
from fuels.datasets.pendulum_cases import *

def DataLoaderODE(dataset, minibatch_size, n_env, is_train=True):
    if is_train:
        sampler = SubsetRamdomSampler(indices=dataset.indices, minibatch_size=minibatch_size)
    else:
        sampler = SubsetSequentialSampler(indices=dataset.indices, minibatch_size=minibatch_size)
    dataloader_params = {
        'dataset': dataset,
        'batch_size': minibatch_size * n_env,
        'num_workers': 0,
        'sampler': sampler,
        'pin_memory': True,
        'drop_last': False
    }
    return DataLoader(**dataloader_params)

def param_lv(buffer_filepath, batch_size_train=25, batch_size_val=25):

    dataset_train_params = {
        "path": buffer_filepath + '_train',
        "n_data_per_env": batch_size_train, 
        "t_horizon": 10, 
        "dt": 0.5, 
        "method": "RK45", 
        "group": "train",
        "params": [
            {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 0.5},
            {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.5},
            {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.5},
            {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 0.75},
            {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 1.0},
            {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.75},
            {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 1.0},
            {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.75},
            
            ]
    }

    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = batch_size_val
    dataset_test_params["group"] = "test"
    dataset_test_params['path'] = buffer_filepath + '_test'
    dataset_test_params['t_horizon'] = 20

    dataset_train = LotkaVolterraDataset(**dataset_train_params)
    dataset_test = LotkaVolterraDataset(**dataset_test_params)
    num_env = len(dataset_train_params["params"])

    dataloader_train = DataLoaderODE(dataset_train, batch_size_train, num_env, is_train=True)
    dataloader_test = DataLoaderODE(dataset_test, batch_size_val, num_env, is_train=False)

    # True parameters
    betas = torch.Tensor([0.5, 0.75, 1.0, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0])
    deltas = torch.Tensor([0.5, 0.5, 0.5, 0.75, 1.0, 0.75, 1.0, 0.75, 1.0])

    params = torch.cat((betas.unsqueeze(-1), deltas.unsqueeze(-1)), axis = 1)
    return dataloader_train, dataloader_test, params

def param_pendulum(buffer_filepath, batch_size_train=25, batch_size_val=25):

    dataset_train_params = {
        'ndata_per_env': batch_size_train,
        'time_horizon': 200,
        'dt': 0.5, 
        'group': 'train',
        'path': buffer_filepath + '_train',
        'IC': 0.5,  # Random IC in [0,0.5]rad ≅ [0,30]deg
        'params' : [# DAMPED
                    {"alpha": 0.05, "w0": 0.50, "wf":0.00, "f0": 0.0}, # Underdamped
                    {"alpha": 0.50, "w0": 0.50, "wf":0.00, "f0": 0.0}, # Critically damped
                    {"alpha": 1.50, "w0": 0.50, "wf":0.00, "f0": 0.0}, # Overdamped
                    
                    # DRIVEN
                    {"alpha": 0.00, "w0": 0.50, "wf":0.05, "f0": 0.1}, # Subresonance
                    {"alpha": 0.00, "w0": 0.50, "wf":0.50, "f0": 0.1}, # Resonance
                    {"alpha": 0.00, "w0": 0.70, "wf":0.75, "f0": 0.2}, # Beats
                    {"alpha": 0.00, "w0": 0.50, "wf":1.00, "f0": 0.1}] # Superresonance
                    
    }
    
    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params['ndata_per_env'] = batch_size_val
    dataset_test_params['group'] = 'test'
    dataset_test_params['path'] = buffer_filepath+'_test'
    dataset_test_params['time_horizon'] = 400
    
    dataset_train = DampedDrivenPendulum(**dataset_train_params)
    dataset_test  = DampedDrivenPendulum(**dataset_test_params)

    dataloader_train = DataLoaderODE(dataset_train, batch_size_train , dataset_train.num_env, is_train=True)
    dataloader_test  = DataLoaderODE(dataset_test, batch_size_val, dataset_train.num_env, is_train=False)

    # True parameters
    alphas = torch.Tensor([vals["alpha"] for vals in dataset_train_params["params"]])
    w0s = torch.Tensor([vals["w0"] for vals in dataset_train_params["params"]])
    wfs = torch.Tensor([vals["wf"] for vals in dataset_train_params["params"]])
    f0s = torch.Tensor([vals["f0"] for vals in dataset_train_params["params"]])
    
    params = torch.cat((w0s.unsqueeze(-1), alphas.unsqueeze(-1),wfs.unsqueeze(-1),f0s.unsqueeze(-1)), axis = 1)
    return dataloader_train, dataloader_test, params 

def param_gs(buffer_filepath, batch_size_train=1, batch_size_val=32):
    dataset_train_params = {
        "n_data_per_env": batch_size_train, 
        "t_horizon": 200, #90
        "dt": 10, # 2 
        "warm_up": 0,
        "method": "RK45",
        "size": 32, 
        "n_block": 3, 
        "dx": 1, 
        "group": "train",
        'path': buffer_filepath + '_train',
        "params": [
            {"f": 0.03, "k": 0.062, "r_u": 0.2097, "r_v": 0.105},
            {"f": 0.039, "k": 0.058, "r_u": 0.2097, "r_v": 0.105},
            {"f": 0.03, "k": 0.058, "r_u": 0.2097, "r_v": 0.105},
            {"f": 0.039, "k": 0.062, "r_u": 0.2097, "r_v": 0.105}
        ]
    }
    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = batch_size_val
    dataset_test_params["path"] = buffer_filepath + '_test'
    dataset_test_params["group"] = "test"
    dataset_test_params['t_horizon'] = 400

    dataset_train, dataset_test = GrayScottDataset(**dataset_train_params), GrayScottDataset(**dataset_test_params)
    dataloader_train = DataLoaderODE(dataset_train, batch_size_train, dataset_train.num_env, is_train=True)
    dataloader_test = DataLoaderODE(dataset_test, batch_size_val, dataset_train.num_env, is_train=False)

    f = torch.Tensor([ 0.03, 0.039, 0.03, 0.039]) 
    k = torch.Tensor([0.062, 0.058, 0.058, 0.062]) 
    params = torch.cat((f.unsqueeze(-1), k.unsqueeze(-1)), axis = 1)
    return dataloader_train, dataloader_test, params

def param_burgers(buffer_filepath, batch_size_train=16, batch_size_val=16):
    dataset_train_params = {
        "n_data_per_env": batch_size_train, 
        "t_horizon": 0.05,
        "dx": 1/256,
        "dt": 1e-3,
        "group": "train",
        'path': buffer_filepath + '_train',
        "params": [
            {"mu": 1e-4},
            {"mu": 3e-4},
            {"mu": 5e-4},
            {"mu": 7e-4}#
        ],
    }
    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = batch_size_val
    dataset_test_params["path"] = buffer_filepath + '_test'
    dataset_test_params["group"] = "test"
    dataset_test_params['t_horizon'] = 0.1

    dataset_train, dataset_test = BurgersF(**dataset_train_params), BurgersF(**dataset_test_params)
    dataloader_train = DataLoaderODE(dataset_train, batch_size_train, dataset_train.num_env, is_train=True) #  // len(dataset_train_params['params'])
    dataloader_test = DataLoaderODE(dataset_test, batch_size_val, dataset_train.num_env, is_train=False) # // len(dataset_train_params['params'])

    params = torch.Tensor([ 1e-4, 3e-4, 5e-4, 7e-4]).unsqueeze(-1) 
    return dataloader_train, dataloader_test, params

def init_dataloaders(dataset, batch_size_train, batch_size_val, buffer_filepath=None):
    assert buffer_filepath is not None
    
    if dataset == 'pendulum':
        return param_pendulum(buffer_filepath, batch_size_train, batch_size_val)
    elif dataset == 'lv':
        return param_lv(buffer_filepath, batch_size_train, batch_size_val)
    elif dataset == 'gs':
        return param_gs(buffer_filepath, batch_size_train, batch_size_val)
    elif dataset == 'burgers':
        return param_burgers(buffer_filepath, batch_size_train, batch_size_val)
    
def param_adapt_pendulum(buffer_filepath, batch_size_train=25, batch_size_val=25):

    dataset_train_params = {
        'ndata_per_env': batch_size_train,
        'time_horizon': 200,
        'dt': 0.5, 
        'group': 'train',
        'path': buffer_filepath+'_train_ada',
        'IC': 1.0,  # Random IC in [0,1]rad ≅ [0,60]deg
        'params' : [# DRIVEN-DAMPED: transient + steady state
                    {"alpha": 0.5, "w0": 1.00, "wf":0.30, "f0": 0.1},
                    {"alpha": 0.5, "w0": 1.00, "wf":2.00, "f0": 0.1},
                    # DRIVEN-DAMPED: chaotic
                    {"alpha": 0.10, "w0": 1.00, "wf":1.00, "f0": 1.5},
                    {"alpha": 0.10, "w0": 0.50, "wf":1.23, "f0": 1.0}]
    }
    
    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params['ndata_per_env'] = batch_size_val
    dataset_test_params['group'] = 'test'
    dataset_test_params['path'] = buffer_filepath+'_test_ada'
    dataset_test_params['time_horizon'] = 400
    
    dataset_train = DampedDrivenPendulum(**dataset_train_params)
    dataset_test  = DampedDrivenPendulum(**dataset_test_params)

    dataloader_train = DataLoaderODE(dataset_train, batch_size_train , dataset_train.num_env, is_train=True)
    dataloader_test  = DataLoaderODE(dataset_test, batch_size_val, dataset_train.num_env, is_train=False)

    # True parameters
    alphas = torch.Tensor([vals["alpha"] for vals in params_test])
    w0s = torch.Tensor([vals["w0"] for vals in params_test])
    wfs = torch.Tensor([vals["wf"] for vals in params_test])
    f0s = torch.Tensor([vals["f0"] for vals in params_test])
    
    params = torch.cat((w0s.unsqueeze(-1), alphas.unsqueeze(-1),wfs.unsqueeze(-1),f0s.unsqueeze(-1)), axis = 1)
    return dataloader_train, dataloader_test, params      

def param_adapt_lv(buffer_filepath, batch_size_train=1, batch_size_val=32):

    betas = [0.3, 0.3, 1.125, 1.125]
    deltas = [0.3, 1.125, 0.3, 1.125]

    dataset_train_params = {
        "path": buffer_filepath + '_train_ada',
        "n_data_per_env": 1, 
        "t_horizon": 10, 
        "dt": 0.5, 
        "method": "RK45", 
        "group": "train",
        "params": [{"alpha": 0.5, "beta": beta_i, "gamma": 0.5, "delta": delta_i} for beta_i, delta_i in zip(betas, deltas)]
    }
    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = batch_size_val
    dataset_test_params["group"] = "test"
    dataset_test_params['path'] = buffer_filepath + '_test_ada'
    dataset_test_params['t_horizon'] = 20

    dataset_train = LotkaVolterraDataset(**dataset_train_params)
    dataset_test = LotkaVolterraDataset(**dataset_test_params)
    num_env = len(dataset_train_params["params"])

    dataloader_train = DataLoaderODE(dataset_train, batch_size_train, num_env, is_train=True)
    dataloader_test = DataLoaderODE(dataset_test, batch_size_val, num_env, is_train=False)

    betas = torch.Tensor(betas)
    deltas = torch.Tensor(deltas)

    # True parameters
    params = torch.cat((betas.unsqueeze(-1), deltas.unsqueeze(-1)), axis = 1)
    return dataloader_train, dataloader_test, params


def param_adapt_gs(buffer_filepath, batch_size_train=1, batch_size_val=32):
    dataset_train_params = {
        "n_data_per_env": batch_size_train, 
        "t_horizon": 200, #90
        "dt": 10,
        "warm_up": 0,
        "method": "RK45",
        "size": 32, 
        "n_block": 3, 
        "dx": 1, 
        "group": "train",
        'path': buffer_filepath + '_train_ada',
        "params": [
            {"f": 0.042, "k": 0.057, "r_u": 0.2097, "r_v": 0.105},
            {"f": 0.033, "k": 0.057, "r_u": 0.2097, "r_v": 0.105},
            {"f": 0.033, "k": 0.06, "r_u": 0.2097, "r_v": 0.105},
            {"f": 0.042, "k": 0.06 ,"r_u": 0.2097, "r_v": 0.105}
        ]
    }

    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = batch_size_val
    dataset_test_params["path"] = buffer_filepath + '_test_ada'
    dataset_test_params["group"] = "test"
    dataset_test_params['t_horizon'] = 400

    dataset_train, dataset_test = GrayScottDataset(**dataset_train_params), GrayScottDataset(**dataset_test_params)
    dataloader_train = DataLoaderODE(dataset_train, batch_size_train, dataset_train.num_env, is_train=True)
    dataloader_test = DataLoaderODE(dataset_test, batch_size_val, dataset_train.num_env, is_train=False)

    f = torch.Tensor([ 0.042, 0.033, 0.033, 0.042]) 
    k = torch.Tensor([0.057, 0.057, 0.060, 0.060]) 
    params = torch.cat((f.unsqueeze(-1), k.unsqueeze(-1)), axis = 1)
    return dataloader_train, dataloader_test, params

def param_adapt_burgers(buffer_filepath, batch_size_train=4, batch_size_val=32):
    dataset_train_params = {
        "n_data_per_env": batch_size_train, 
        "t_horizon": 0.05,
        "dx": 1/256,
        "dt": 1e-3,
        "group": "train",
        'path': buffer_filepath + '_train_ada',
        "params": [
            {"mu": 5e-3},
            {"mu": 7e-3},
            {"mu": 1e-5},
            {"mu": 5e-5}
        ],
    }
    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = batch_size_val
    dataset_test_params["path"] = buffer_filepath + '_test_ada'
    dataset_test_params["group"] = "test"
    dataset_test_params['t_horizon'] = 0.1

    dataset_train, dataset_test = BurgersF(**dataset_train_params), BurgersF(**dataset_test_params)
    dataloader_train = DataLoaderODE(dataset_train, batch_size_train, dataset_train.num_env, is_train=True) #  // len(dataset_train_params['params'])
    dataloader_test = DataLoaderODE(dataset_test, batch_size_val, dataset_train.num_env, is_train=False) # // len(dataset_train_params['params'])

    params = torch.Tensor([5e-3, 7e-3, 1e-5, 5e-5]).unsqueeze(-1)
    return dataloader_train, dataloader_test, params

def init_adapt_dataloaders(dataset, batch_size_train = 1, batch_size_val = 32, buffer_filepath=None):
    assert buffer_filepath is not None

    if dataset == 'pendulum':
        return param_adapt_pendulum(buffer_filepath, batch_size_train, batch_size_val)
    elif dataset == 'lv':
        return param_adapt_lv(buffer_filepath, batch_size_train, batch_size_val)
    elif dataset == 'gs':
        return param_adapt_gs(buffer_filepath, batch_size_train, batch_size_val)
    elif dataset == 'burgers':
        return param_adapt_burgers(buffer_filepath, batch_size_train, batch_size_val)