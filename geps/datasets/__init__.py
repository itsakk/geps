import torch
from torch.utils.data import DataLoader 
from geps.datasets.pendulum import DampedDrivenPendulum
from geps.datasets.lv import LotkaVolterraDataset
from geps.datasets.gs import GrayScottDataset
from geps.datasets.burgers import Burgers
from geps.datasets.kolmo import Turb2d
import numpy as np
from itertools import product
from geps.datasets.combined import *

def DataLoaderODE(dataset, minibatch_size, shuffle=True):
    dataloader_params = {
        'dataset': dataset,
        'batch_size': minibatch_size,
        'num_workers': 0,
        'shuffle': shuffle,
        'pin_memory': True,
        'drop_last': False
    }
    return DataLoader(**dataloader_params)

def param_lv(buffer_filepath, batch_size_train=25, batch_size_val=25):

    dataset_train_params = {
        "path": buffer_filepath + '_train',
        "n_data_per_env": 4, 
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
    dataset_test_params["n_data_per_env"] = 32
    dataset_test_params["group"] = "test"
    dataset_test_params['path'] = buffer_filepath + '_test'
    dataset_test_params['t_horizon'] = 20

    dataset_train = LotkaVolterraDataset(**dataset_train_params)
    dataset_test = LotkaVolterraDataset(**dataset_test_params)
    num_env = len(dataset_train_params["params"])

    dataloader_train = DataLoaderODE(dataset_train, batch_size_train, shuffle=True)
    dataloader_test = DataLoaderODE(dataset_test, batch_size_val, shuffle=False)
    
    # True parameters
    betas = torch.Tensor([0.5, 0.75, 1.0, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0])
    deltas = torch.Tensor([0.5, 0.5, 0.5, 0.75, 1.0, 0.75, 1.0, 0.75, 1.0])

    params = torch.cat((betas.unsqueeze(-1), deltas.unsqueeze(-1)), axis = 1)
    return dataloader_train, dataloader_test, params

def param_pendulum(buffer_filepath, batch_size_train=25, batch_size_val=25):

    dataset_train_params = {
        'ndata_per_env': 8,
        'time_horizon': 25,
        'dt': 0.5, 
        'group': 'train',
        'path': buffer_filepath + '_train',
        'IC': 0.25,  # Random IC in [0,0.5]rad ≅ [0,30]deg
        'params' : [# DAMPED
                    {"alpha": 0.20, "w0": 0.50, "wf":0.00, "f0": 0.0}, # Underdamped
                    {"alpha": 0.50, "w0": 0.50, "wf":0.00, "f0": 0.0}, # Critically damped
                    # {"alpha": 1.50, "w0": 0.50, "wf":0.00, "f0": 0.0}, # Overdamped
                    
                    # # DRIVEN
                    # {"alpha": 0.00, "w0": 0.50, "wf":0.05, "f0": 0.05}, # Subresonance
                    # {"alpha": 0.00, "w0": 0.50, "wf":0.50, "f0": 0.1}, # Resonance
                    {"alpha": 0.00, "w0": 0.70, "wf":0.75,  "f0": 0.2}, # Beats
                    {"alpha": 0.00, "w0": 0.50, "wf":1.00, "f0": 0.1}  # Superresonance
        ]
    }
    
    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params['ndata_per_env'] = 32
    dataset_test_params['group'] = 'test'
    dataset_test_params['path'] = buffer_filepath+'_test'
    dataset_test_params['time_horizon'] = 50
    
    dataset_train = DampedDrivenPendulum(**dataset_train_params)
    dataset_test  = DampedDrivenPendulum(**dataset_test_params)

    dataloader_train = DataLoaderODE(dataset_train, batch_size_train, shuffle = True)
    dataloader_test  = DataLoaderODE(dataset_test, batch_size_val, shuffle = False)

    # True parameters
    alphas = torch.Tensor([vals["alpha"] for vals in dataset_train_params["params"]])
    w0s = torch.Tensor([vals["w0"] for vals in dataset_train_params["params"]])
    wfs = torch.Tensor([vals["wf"] for vals in dataset_train_params["params"]])
    f0s = torch.Tensor([vals["f0"] for vals in dataset_train_params["params"]])
    
    params = torch.cat((w0s.unsqueeze(-1), alphas.unsqueeze(-1),wfs.unsqueeze(-1),f0s.unsqueeze(-1)), axis = 1)
    return dataloader_train, dataloader_test, params

def param_gs(buffer_filepath, batch_size_train=1, batch_size_val=32):
    dataset_train_params = {
        "n_data_per_env": 256, 
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
    dataset_test_params["n_data_per_env"] = 32
    dataset_test_params["path"] = buffer_filepath + '_test'
    dataset_test_params["group"] = "test"
    dataset_test_params['t_horizon'] = 400

    dataset_train, dataset_test = GrayScottDataset(**dataset_train_params), GrayScottDataset(**dataset_test_params)
    dataloader_train = DataLoaderODE(dataset_train, batch_size_train, shuffle=False)
    dataloader_test = DataLoaderODE(dataset_test, batch_size_val, shuffle=False)

    f = torch.Tensor([ 0.03, 0.039, 0.03, 0.039])
    k = torch.Tensor([0.062, 0.058, 0.058, 0.062]) 
    params = torch.cat((f.unsqueeze(-1), k.unsqueeze(-1)), axis = 1)
    return dataloader_train, dataloader_test, params

# def param_gs_multienv(buffer_filepath, batch_size_train=1, batch_size_val=32, n=1024):
    
#     # Define the range for f and k
#     f_range = (0.034, 0.041)
#     k_range = (0.0575, 0.0615)
#     np.random.seed(123)
#     # Sample f and k from a uniform distribution
#     f_samples = np.random.uniform(f_range[0], f_range[1], 100)
#     k_samples = np.random.uniform(k_range[0], k_range[1], 100)

#     # Generate the Cartesian product
#     all_combinations = list(product(f_samples, k_samples))

#     # Randomly sample 1200 unique combinations
#     sampled_combinations = np.random.choice(len(all_combinations), n, replace=False)
#     params = [all_combinations[i] for i in sampled_combinations]

#     dataset_train_params = {
#         "n_data_per_env": 4,
#         "t_horizon": 200,
#         "dt": 10,
#         "warm_up": 0,
#         "method": "RK45",
#         "size": 32,
#         "n_block": 3,
#         "dx": 1,
#         "group": "train",
#         'path': buffer_filepath + '_train',
#         "params": [
#             {"f": f, "k": k, "r_u": 0.2097, "r_v": 0.105}
#             for f, k in params
#         ]
#     }

#     dataset_test_params = dict()
#     dataset_test_params.update(dataset_train_params)
#     dataset_test_params["n_data_per_env"] = 4
#     dataset_test_params["path"] = buffer_filepath + '_test'
#     dataset_test_params["group"] = "test"
#     dataset_test_params['t_horizon'] = 400

#     dataset_train, dataset_test = GrayScottDataset(**dataset_train_params), GrayScottDataset(**dataset_test_params)
#     dataloader_train = DataLoaderODE(dataset_train, batch_size_train, shuffle=True)
#     dataloader_test = DataLoaderODE(dataset_test, batch_size_val, shuffle=False)

#     params = torch.Tensor(params)
#     return dataloader_train, dataloader_test, params

def param_gs_multienv(buffer_filepath, batch_size_train=1, batch_size_val=32, n=1024):
    
    # Define the range for f and k
    f_range = (0.03, 0.04)
    k_range = (0.058, 0.062)
    np.random.seed(123)
    # Sample f and k from a uniform distribution
    f_samples = np.random.uniform(f_range[0], f_range[1], 100)
    k_samples = np.random.uniform(k_range[0], k_range[1], 100)

    # Generate the Cartesian product
    all_combinations = list(product(f_samples, k_samples))

    # Randomly sample 1200 unique combinations
    sampled_combinations = np.random.choice(len(all_combinations), n, replace=False)
    params = [all_combinations[i] for i in sampled_combinations]

    dataset_train_params = {
        "n_data_per_env": 1,
        "t_horizon": 200,
        "dt": 10,
        "warm_up": 0,
        "method": "RK45",
        "size": 32,
        "n_block": 3,
        "dx": 1,
        "group": "train",
        'path': buffer_filepath + '_train',
        "params": [
            {"f": f, "k": k, "r_u": 0.2097, "r_v": 0.105}
            for f, k in zip(f_samples, k_samples)
        ]
    }

    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = 4
    dataset_test_params["path"] = buffer_filepath + '_test'
    dataset_test_params["group"] = "test"
    dataset_test_params['t_horizon'] = 200

    dataset_train, dataset_test = GrayScottDataset(**dataset_train_params), GrayScottDataset(**dataset_test_params)
    dataloader_train = DataLoaderODE(dataset_train, batch_size_train, shuffle=False)
    dataloader_test = DataLoaderODE(dataset_test, batch_size_val, shuffle=False)

    params = torch.Tensor(list(zip(f_samples, k_samples)))
    return dataloader_train, dataloader_test, params

def param_burgers(buffer_filepath, batch_size_train=16, batch_size_val=16):
    dataset_train_params = {
        "n_data_per_env": 4, 
        "t_horizon": 0.05,
        "N": 16384,
        "N_filt": 256,
        "dt_filt": 1e-3,
        "group": "train",
        'path': buffer_filepath + '_train',
        "params": [
            {"mu": 5e-1, 'force': '', 'F': 5, 'w': 1.5, "domain": 2},
            {"mu": 5e-2, 'force': '', 'F': 5, 'w': 1.5, "domain": 2},
            {"mu": 5e-4, 'force': '', 'F': 5, 'w': 1.5, "domain": 2},
            {"mu": 5e-1, 'force': 'periodic', 'F': 5, 'w': 1.5, "domain": 2},
            {"mu": 5e-2, 'force': 'periodic', 'F': 5, 'w': 1.5, "domain": 2},
            {"mu": 5e-4, 'force': 'periodic', 'F': 5, 'w': 1.5, "domain": 2},
        ],
    }

    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = 32
    dataset_test_params["path"] = buffer_filepath + '_test'
    dataset_test_params["group"] = "test"
    dataset_test_params['t_horizon'] = 0.1

    dataset_train, dataset_test = Burgers(**dataset_train_params), Burgers(**dataset_test_params)
    dataloader_train = DataLoaderODE(dataset_train, batch_size_train, shuffle=True) #  // len(dataset_train_params['params'])
    dataloader_test = DataLoaderODE(dataset_test, batch_size_val, shuffle=False) # // len(dataset_train_params['params'])

    params = torch.Tensor([5e-1, 5e-2, 5e-4, 5e-1, 5e-2, 5e-4]).unsqueeze(-1)
    return dataloader_train, dataloader_test, params

def param_kolmo(buffer_filepath, batch_size_train=16, batch_size_val=16):
    dataset_train_params = {
        "n_data_per_env": 16, 
        "t_horizon": 1.4,
        "N": 512,
        "N_filt": 64,
        "dt": 0.005,
        "dt_filt": 0.05,
        "warmup": 8,
        "group": "train",
        'path': buffer_filepath + '_train',
        "params": [
            {"mu": 1e-3, 'force': '', 'domain': 0.75, 'k': 4},
            {"mu": 1e-3, 'force': '', 'domain': 1., 'k': 4},
            {"mu": 1e-3, 'force': 'sin', 'domain': 0.75, 'k': 4},
            {"mu": 1e-3, 'force': 'sin', 'domain': 1, 'k': 4},
            {"mu": 1e-4, 'force': '', 'domain': 0.75, 'k': 4},
            {"mu": 1e-4, 'force': '', 'domain': 1., 'k': 4},
            {"mu": 1e-4, 'force': 'sin', 'domain': 0.75, 'k': 4},
            {"mu": 1e-4, 'force': 'sin', 'domain': 1, 'k': 4},
        ],
    }
    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = 32
    dataset_test_params["path"] = buffer_filepath + '_test'
    dataset_test_params["group"] = "test"
    dataset_test_params['t_horizon'] = 2.4

    dataset_train, dataset_test = Turb2d(**dataset_train_params), Turb2d(**dataset_test_params)
    dataloader_train = DataLoaderODE(dataset_train, batch_size_train, shuffle=True)
    dataloader_test = DataLoaderODE(dataset_test, batch_size_val, shuffle=False)

    nu = torch.Tensor([1e-3, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4])
    domain = torch.Tensor([0.75, 1.0, 0.75, 1.0, 0.75, 1.0, 0.75, 1.0])
    params = torch.cat((nu.unsqueeze(-1), domain.unsqueeze(-1)), axis = 1)
    return dataloader_train, dataloader_test, params

def load_combined(buffer_filepath, train_batch_size = 32, val_batch_size = 32):
    # Sampled ranges for alphas, beta, delta, gamma
    alpha= [0.5, 0.75]
    beta= [0.075, 0.25]
    delta= [0.25, 0.75]
    gamma = [0.25, 0.75]

    # Generate the Cartesian product
    params = torch.tensor(list(product(alpha, beta, delta, gamma)))
    trainset = CombinedDataset(buffer_filepath + '_train.h5', 'train', dt = 0.1)
    valset = CombinedDataset(buffer_filepath + '_val.h5', 'val', dt = 0.1)
    train_loader = DataLoader(trainset, batch_size= train_batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=val_batch_size, shuffle=True, num_workers=1, pin_memory=True)
    return train_loader, val_loader, params

def load_combined_big(buffer_filepath, train_batch_size = 32, val_batch_size = 32):
    # Number of desired environments
    num_envs = 1200 + 10
    # Sampled ranges for alphas, beta, delta, gamma
    np.random.seed(42)

    # Sampled ranges for alphas, beta, delta, gamma
    alpha= np.linspace(0.5, 1, num = 10)
    beta= np.linspace(0.0, 0.5, num = 10)
    delta= np.linspace(0.0, 1., num = 10)
    gamma = np.linspace(0.0, 1., num = 10)

    # Generate the Cartesian product
    all_combinations = list(product(alpha, beta, delta, gamma))

    # Randomly sample 1200 unique combinations
    sampled_combinations = np.random.choice(len(all_combinations), num_envs, replace=False)
    params = [all_combinations[i] for i in sampled_combinations]
    params[:1200]
    trainset = CombinedDataset(buffer_filepath + '_train.h5', 'train', dt = 0.1)
    valset = CombinedDataset(buffer_filepath + '_val.h5', 'val', dt = 0.1)
    train_loader = DataLoader(trainset, batch_size= train_batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=val_batch_size, shuffle=True, num_workers=1, pin_memory=True)
    return train_loader, val_loader, torch.tensor(params)

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
    elif dataset == 'kolmo':
        return param_kolmo(buffer_filepath, batch_size_train, batch_size_val)
    elif dataset == 'combined':
        return load_combined(buffer_filepath, train_batch_size = batch_size_train, val_batch_size = batch_size_val)
    elif dataset == 'combined_big':
        return load_combined_big(buffer_filepath, train_batch_size = batch_size_train, val_batch_size = batch_size_val)
    elif dataset == 'gs_multienv':
        return param_gs_multienv(buffer_filepath, batch_size_train, batch_size_val)
    

def param_adapt_pendulum(buffer_filepath, batch_size_train=25, batch_size_val=25):
    dataset_train_params = {
        'ndata_per_env': 1,
        'time_horizon': 25,
        'dt': 0.5, 
        'group': 'train',
        'path': buffer_filepath+'_train_ada',
        'IC': 0.25,
        'params' : [# DRIVEN-DAMPED: transient + steady state
                    {"alpha": 0.5, "w0": 1.00, "wf":0.30, "f0": 0.2},
                    {"alpha": 0.5, "w0": 0.75, "wf":0.70, "f0": 0.1},
                    # DRIVEN-DAMPED: chaotic
                    {"alpha": 0.10, "w0": 1.00, "wf":1.00, "f0": 0.15},
                    {"alpha": 0.10, "w0": 0.50, "wf":0.5, "f0": 0.05}]
    }
    
    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params['ndata_per_env'] = 32
    dataset_test_params['group'] = 'test'
    dataset_test_params['path'] = buffer_filepath+'_test_ada'
    dataset_test_params['time_horizon'] = 50
    
    dataset_train = DampedDrivenPendulum(**dataset_train_params)
    dataset_test  = DampedDrivenPendulum(**dataset_test_params)

    dataloader_train = DataLoaderODE(dataset_train, batch_size_train, shuffle=True)
    dataloader_test = DataLoaderODE(dataset_test, batch_size_val, shuffle=False)

    # True parameters
    alphas = torch.Tensor([vals["alpha"] for vals in dataset_train_params["params"]])
    w0s = torch.Tensor([vals["w0"] for vals in dataset_train_params["params"]])
    wfs = torch.Tensor([vals["wf"] for vals in dataset_train_params["params"]])
    f0s = torch.Tensor([vals["f0"] for vals in dataset_train_params["params"]])
    
    params = torch.cat((w0s.unsqueeze(-1), alphas.unsqueeze(-1),wfs.unsqueeze(-1),f0s.unsqueeze(-1)), axis = 1)
    return dataloader_train, dataloader_test, params      

def param_adapt_gs(buffer_filepath, batch_size_train=1, batch_size_val=32):
    dataset_train_params = {
        "n_data_per_env": 1, 
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
    dataset_test_params["n_data_per_env"] = 32
    dataset_test_params["path"] = buffer_filepath + '_test_ada'
    dataset_test_params["group"] = "test"
    dataset_test_params['t_horizon'] = 400

    dataset_train, dataset_test = GrayScottDataset(**dataset_train_params), GrayScottDataset(**dataset_test_params)
    dataloader_train = DataLoaderODE(dataset_train, batch_size_train, shuffle=True)
    dataloader_test = DataLoaderODE(dataset_test, batch_size_val, shuffle=False)

    f = torch.Tensor([ 0.042, 0.033, 0.033, 0.042]) 
    k = torch.Tensor([0.057, 0.057, 0.060, 0.060]) 
    params = torch.cat((f.unsqueeze(-1), k.unsqueeze(-1)), axis = 1)
    return dataloader_train, dataloader_test, params

def param_adapt_burgers(buffer_filepath, batch_size_train=16, batch_size_val=16):
    dataset_train_params = {
        "n_data_per_env": 1, 
        "t_horizon": 0.05,
        "N": 16384,
        "N_filt": 256,
        "dt_filt": 1e-3,
        "group": "train",
        'path': buffer_filepath + '_train_ada',
        "params": [
            {"mu": 1.0, 'force': 'exp', 'F': 5, 'w': 1.5, "domain": 2},
            {"mu": 5e-5, 'force': 'exp', 'F': 5, 'w': 1.5, "domain": 2},
            {"mu": 1.0, 'force': 'exp', 'F': 5, 'w': 3.0, "domain": 2},
            {"mu": 5e-5, 'force': 'exp', 'F': 5, 'w': 3.0, "domain": 2},
        ],
    }

    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = 32
    dataset_test_params["path"] = buffer_filepath + '_test_ada'
    dataset_test_params["group"] = "test_ada"
    dataset_test_params['t_horizon'] = 0.1

    dataset_train, dataset_test = Burgers(**dataset_train_params), Burgers(**dataset_test_params)
    dataloader_train = DataLoaderODE(dataset_train, batch_size_train, shuffle  = True)
    dataloader_test = DataLoaderODE(dataset_test, batch_size_val, shuffle  = False)

    params = torch.Tensor([1.0, 5e-5, 1.0, 5e-5]).unsqueeze(-1) 
    return dataloader_train, dataloader_test, params

def param_adapt_kolmo(buffer_filepath, batch_size_train=16, batch_size_val=16):
    dataset_train_params = {
        "n_data_per_env": 16, 
        "t_horizon": 1.4,
        "N": 512,
        "N_filt": 64,
        "dt": 0.005,
        "dt_filt": 0.05,
        "warmup": 8,
        "group": "train",
        'path': buffer_filepath + '_train_ada',
        "params": [
            {"mu": 5e-4, 'force': 'exp', 'domain': 1.25, 'k': 4},
            {"mu": 5e-4, 'force': 'exp', 'domain': 1.25, 'k': 4},
            {"mu": 5e-4, 'force': 'periodic', 'domain': 1.25, 'k': 4},
            {"mu": 5e-4, 'force': 'periodic', 'domain': 1.25, 'k': 4},
        ],
    }
    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = 32
    dataset_test_params["path"] = buffer_filepath + '_test_ada'
    dataset_test_params["group"] = "test"
    dataset_test_params['t_horizon'] = 2.4

    dataset_train, dataset_test = Turb2d(**dataset_train_params), Turb2d(**dataset_test_params)
    dataloader_train = DataLoaderODE(dataset_train, 16 * 4, shuffle = False)
    dataloader_test = DataLoaderODE(dataset_test, batch_size_val, shuffle = False)

    nu = torch.Tensor([1e-5, 1e-5, 1e-5, 1e-5])
    domain = torch.Tensor([1.5, 2.0, 1.5, 2.0])
    params = torch.cat((nu.unsqueeze(-1), domain.unsqueeze(-1)), dim = -1)
    return dataloader_train, dataloader_test, params

def load_combined_adapt(buffer_filepath, train_batch_size = 32, val_batch_size = 32):
    # Sampled ranges for alphas, beta, delta, gamma
    num_envs = 1200 + 10
    # Sampled ranges for alphas, beta, delta, gamma
    np.random.seed(42)

    # Sampled ranges for alphas, beta, delta, gamma
    alpha= np.linspace(0.5, 1, num = 10)
    beta= np.linspace(0.0, 0.5, num = 10)
    delta= np.linspace(0.0, 1., num = 10)
    gamma = np.linspace(0.0, 1., num = 10)

    # Generate the Cartesian product
    all_combinations = list(product(alpha, beta, delta, gamma))

    # Randomly sample 1200 unique combinations
    sampled_combinations = np.random.choice(len(all_combinations), num_envs, replace=False)
    params = [all_combinations[i] for i in sampled_combinations]
    params = params[1200:]
    trainset = CombinedDataset(buffer_filepath + '_train.h5', 'train', dt = 0.1)
    valset = CombinedDataset(buffer_filepath + '_val.h5', 'val', dt = 0.1)
    train_loader = DataLoader(trainset, batch_size= train_batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=val_batch_size, shuffle=True, num_workers=1, pin_memory=True)
    return train_loader, val_loader, torch.tensor(params)

def init_adapt_dataloaders(dataset, batch_size_train = 1, batch_size_val = 32, buffer_filepath=None):
    assert buffer_filepath is not None

    if dataset == 'pendulum':
        return param_adapt_pendulum(buffer_filepath, batch_size_train, batch_size_val)
    elif dataset == 'gs':
        return param_adapt_gs(buffer_filepath, batch_size_train, batch_size_val)
    elif dataset == 'burgers':
        return param_adapt_burgers(buffer_filepath, batch_size_train, batch_size_val)
    elif dataset == 'kolmo':
        return param_adapt_kolmo(buffer_filepath, batch_size_train, batch_size_val)
    elif dataset == 'combined_big':
        return load_combined_adapt(buffer_filepath, train_batch_size = batch_size_train, val_batch_size = batch_size_val)