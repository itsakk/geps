import os
import torch
import hydra
import wandb

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np

from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from fuels.utils import fix_seed, count_parameters, init_weights
from fuels.datasets import *
from fuels.model.forecasters import *
from fuels.losses import *

@hydra.main(config_path="config/model/", config_name="train.yaml")
def main(cfg: DictConfig) -> None:

    entity = cfg.wandb.entity
    project = cfg.wandb.project
    run_id = cfg.wandb.id
    run_name = cfg.wandb.name

    data_dir = cfg.data.dir
    dataset_name = cfg.data.dataset_name
    seed = cfg.data.seed
    path = os.path.join(data_dir, dataset_name)

    # optim
    batch_size_train = cfg.optim.batch_size_train
    batch_size_val = cfg.optim.batch_size_val
    epochs = cfg.optim.epochs
    lr = cfg.optim.lr
    init_type = cfg.optim.init_type
    regul = cfg.optim.regul

    # model decoder
    state_c = cfg.model.state_c
    code_c = cfg.model.code_c
    hidden_c = cfg.model.hidden_c
    factor = cfg.model.factor
    is_complete = cfg.model.is_complete
    type_augment = cfg.model.type_augment

    # model forecaster
    epsilon = cfg.model.teacher_forcing_init
    epsilon_t = cfg.model.teacher_forcing_decay
    epsilon_freq = cfg.model.teacher_forcing_update
    method = cfg.model.method
    options = cfg.model.options

    if (dataset_name == 'gs') & (type_augment != ''):
        options = dict(step_size = 1)

    # device
    device = torch.device("cuda")

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

    run.tags = (
            ("fuels",)
            + (dataset_name,)
            + (type_augment,)
            + ("train",)
    )

    run_name = wandb.run.name
    wandb.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    # create model folder
    model_dir = Path(os.getenv("WANDB_DIR")) / 'fuels' / 'new_ver' / 'pretrain'/ dataset_name / "model"
    os.makedirs(str(model_dir), exist_ok=True)

    # set seed
    fix_seed(seed)

    # load data
    train, test, params = init_dataloaders(dataset_name, batch_size_train, batch_size_val, os.path.join(path, dataset_name))
    params = params.to(device)
    num_env = len(params)

    model = Forecaster(dataset_name, state_c, hidden_c, code_c, factor, num_env, is_complete, type_augment, method, options).to(device)
    init_weights(model, init_config=init_type)

    optimizer = optim.Adam(model.parameters(), lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.9, # gamma_step
        patience=50,
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
    criterion_eval = RelativeL2()
    param_error = 0

    print("ntrain, ntest : ", ntrain, ntest)
    print("t_in : ", t_in)
    print("dataset_name : ", dataset_name)
    print("path : ", path)
    print("train.dataset[0]['states'].shape : ", train.dataset[0]['states'].shape)
    print("num_env : ", num_env)
    print(f"n_params forecaster: {count_parameters(model)}")
    print("params : ", params)

    epsilon_t = 0
    for epoch in range(epochs):
        outputs_pendulum = []
        targets_pendulum = []

        step_show = epoch % nupdate == 0
        step_show_last = epoch == epochs - 1
        loss_train = 0

        if epoch % epsilon_freq == 0:
            epsilon_t = epsilon_t * epsilon

        for _, data in enumerate(train):
            targets = data["states"].to(device)
            n_samples = len(targets)
            t = data["t"][0].to(device)
            env = data["env"].to(device)
            outputs = model(targets, t, env, epsilon_t)

            loss = criterion(outputs, targets)
            
            if regul:
                l2_loss = l2_penalty(model)

            loss_total = loss # + l2_loss + param_error
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            optimizer.zero_grad()
                
            loss_train += loss.item() * n_samples

        loss_train /= ntrain
        scheduler.step(loss_train)
     
        if True in (step_show, step_show_last):
            with torch.no_grad():
                loss_test_in = 0
                loss_test_out = 0
                err_per_env = []
                for _, data_test in enumerate(test):
                    targets_test = data_test['states'].to(device)
                    n_samples = len(targets_test)
                    t = data_test['t'][0].to(device)
                    env = data_test['env'].to(device)

                    outputs_test = model(targets_test, t, env)

                    loss_in = criterion_eval(outputs_test[..., :t_in], targets_test[..., :t_in])
                    loss_out = criterion_eval(outputs_test[..., t_in:], targets_test[..., t_in:])
                    loss_test_in += loss_in.item() * n_samples
                    loss_test_out += loss_out.item() * n_samples
                    err_per_env.append(loss_in.item())
                    
                    if dataset_name == 'pendulum':
                        random = np.random.randint(32)
                        outputs_pendulum.append(outputs_test[random])
                        targets_pendulum.append(targets_test[random])

            if type_augment != '':
                param_error = torch.mean(abs((model.derivative.model_phy.estimated_params - params)))
            #err_per_env = torch.stack(err_per_env).reshape(num_env, -1).mean(dim=1).tolist()
            
            #print("err_per_env : ", err_per_env)
            loss_test_in /= ntest
            loss_test_out /= ntest

        if True in (step_show, step_show_last):
            if dataset_name == 'pendulum':
                fig, axs = plt.subplots(num_env, 1, figsize=(10, 15))  # Create a figure with 4 subplots

                for i in range(num_env):  # Loop over the first 4 data samples
                    axs[i].plot(targets_pendulum[i].cpu().detach().numpy()[0, :], label='target')
                    axs[i].plot(outputs_pendulum[i].cpu().detach().numpy()[0, :], label='output')
                    axs[i].legend()
                    axs[i].set_xlabel('time')
                    axs[i].set_title(f'Data sample {i + 1}')  # Set a title for each subplot

                wandb.log(
                    {
                        "train_loss": loss_train,
                        "loss_test_in": loss_test_in,
                        "loss_test_out": loss_test_out,
                        "param_errror_mape": param_error,
                        "prediction": wandb.Image(fig),
                        "lr": scheduler.get_last_lr()[0],
                    },
                )
            else:
                wandb.log(
                    {
                        "train_loss": loss_train,
                        "loss_test_in": loss_test_in,
                        "loss_test_out": loss_test_out,
                        "param_errror_mape": param_error,
                    },
                )

        else:

            wandb.log(
                {
                    "train_loss": loss_train,
                    "lr": scheduler.get_last_lr()[0],
                },
                step=epoch,
                commit=not step_show,
            )
        if dataset_name != 'pendulum':
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
        if dataset_name == 'pendulum':
            if loss_test_in < best_loss:
                best_loss = loss_test_in
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