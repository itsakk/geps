import os
import torch
import hydra
import wandb

import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from fuels.model.networks import *
from fuels.utils import fix_seed, count_parameters
from fuels.datasets import *
from fuels.model.forecasters import *

@hydra.main(config_path="config/model/", config_name="adapt.yaml")
def main(cfg: DictConfig) -> None:

    entity = cfg.wandb.entity
    project = cfg.wandb.project
    run_id = cfg.wandb.id
    run_name = cfg.wandb.name

    # pretrained model run name
    pretrain_model_run_name = cfg.pretrain.run_name

    #data
    data_dir = cfg.data.dir
    dataset_name = cfg.data.dataset_name
    seed = cfg.data.seed
    path = os.path.join(data_dir, dataset_name)

    # optim
    batch_size_train = cfg.optim.batch_size_train
    batch_size_val = cfg.optim.batch_size_val
    epochs = cfg.optim.epochs
    lr = cfg.optim.lr

    # model decoder
    state_c = cfg.model.state_c
    code_c = cfg.model.code_c
    hidden_c = cfg.model.hidden_c
    factor = cfg.model.factor
    is_complete = cfg.model.is_complete
    pde_params = cfg.model.pde_params
    type_augment = cfg.model.type_augment

    # model forecaster
    method = cfg.model.method
    options = cfg.model.options

    # device
    device = torch.device("cuda")

    if dataset_name == 'gs':
        options = dict(step_size = 1)
        
    # wandb
    run_dir = (
        os.path.join(os.getenv("WANDB_DIR"),
                     f"wandb/{cfg.wandb.dir}/{dataset_name}")
        if cfg.wandb.dir is not None
        else None
    )

    # initialize wandb log
    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        id=run_id,
        dir=run_dir,
        resume='allow',
    )
    
    run_name = wandb.run.name
    wandb.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    # create model folder
    model_dir = Path(os.getenv("WANDB_DIR")) / 'pdegen' / 'new_ver' / 'adapt'/ dataset_name / "model"
    os.makedirs(str(model_dir), exist_ok=True)

    # set seed
    fix_seed(seed)

    # load data
    train, test, params = init_adapt_dataloaders(dataset_name, batch_size_train, batch_size_val, os.path.join(path, dataset_name))
    params = params.to(device)
    pde_params = params if pde_params == True else None

    num_env = len(params)

    model = Forecaster(dataset_name, state_c, hidden_c, code_c, factor, pde_params, num_env, is_complete, type_augment, method, options).to(device)

   # pretrained model path
    pretrained_model_path = Path(os.getenv("WANDB_DIR")) / 'pdegen' / 'new_ver' / 'pretrain' / dataset_name / "model" / pretrain_model_run_name
    checkpoint = torch.load(f"{pretrained_model_path}.pt", map_location=device)

    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model'].items() if (k in model_dict and not "codes" in k)}
    pretrained_dict['derivative.model_phy.params'] = pretrained_dict['derivative.model_phy.params'].mean(dim = 0).repeat(num_env, 1) # torch.tensor(0).repeat(num_env, 2)

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.to(device)

    for name, param in model.named_parameters():
        if param.requires_grad and "codes" not in name and "params" not in name:
            param.requires_grad = False
        print("name, param.requires_grad : ", name, param.requires_grad)
    
    optimizer = optim.Adam(model.parameters(), lr= lr, betas=(0.9, 0.999))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.9, # gamma_step
        patience=250,
        threshold=0.01,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-5,
        eps=1e-08,
        verbose=True,
    )

    ntrain = len(train.dataset)
    ntest = len(test.dataset)
    best_loss = 10**6
    t_in = train.dataset[0]['t'].shape[-1]
    nupdate = 100
    criterion = nn.MSELoss()

    print("ntrain, ntest : ", ntrain, ntest)
    print("t_in : ", t_in)
    print("dataset_name : ", dataset_name)
    print("path : ", path)
    print("train.dataset[0]['states'].shape : ", train.dataset[0]['states'].shape)
    print("num_env : ", num_env)
    print(f"n_params forecaster: {count_parameters(model)}")

    for epoch in range(epochs):
        step_show = epoch % nupdate == 0
        step_show_last = epoch == epochs - 1
        loss_train = 0

        for _, data in enumerate(train):
            targets = data["states"].to(device)
            n_samples = len(targets)
            t = data["t"][0].to(device)
            env = data["env"].to(device)
            y0 = targets[..., 0]

            outputs = model(y0, t, env)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
                
            loss_train += loss.item() * n_samples

        loss_train /= ntrain
        scheduler.step(loss_train)

        if True in (step_show, step_show_last):
            with torch.no_grad():
                loss_test_in = 0
                loss_test_out = 0
                for _, data_test in enumerate(test):
                    targets_test = data_test['states'].to(device)
                    n_samples = len(targets_test)
                    t = data_test['t'][0].to(device)
                    env = data_test['env'].to(device)

                    outputs_test = model(targets_test[..., 0], t, env)
                    loss_in = criterion(outputs_test[..., :t_in], targets_test[..., :t_in])
                    loss_out = criterion(outputs_test[..., t_in:], targets_test[..., t_in:])
                    loss_test_in += loss_in.item() * n_samples
                    loss_test_out += loss_out.item() * n_samples

            if is_complete:
                param_error = torch.mean((model.derivative.model_phy.params - params)**2)
            else:
                param_error = torch.mean((model.derivative.model_phy.params[:, 0] - params[:, 0])**2)

            loss_test_in /= ntest
            loss_test_out /= ntest

        if True in (step_show, step_show_last):
            wandb.log(
                {
                    "loss_test_in": loss_test_in,
                    "loss_test_out": loss_test_out,
                    "param_error": param_error,
                    "train_loss": loss_train,
                },
            )

        else:

            wandb.log(
                {
                    "train_loss": loss_train,
                },
                step=epoch,
                commit=not step_show,
            )
        if loss_train < best_loss:
            best_loss = loss_train
            torch.save(
                {
                    "cfg": cfg,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": best_loss,
                },
                f"{model_dir}/{run_name}.pt",
            )

    return best_loss

if __name__ == "__main__":
    main()