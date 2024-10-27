import os
import torch
import hydra
import wandb

import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from fuels.utils import fix_seed, count_parameters, init_weights
from datasets import *
from forecasters import *
from fuels.losses import *
from metalearning import *

@hydra.main(config_path="../../config/cavia/", config_name="train.yaml")
def main(cfg: DictConfig) -> None:

    entity = cfg.wandb.entity
    project = cfg.wandb.project
    run_id = cfg.wandb.id
    run_name = cfg.wandb.name

    # data
    data_dir = cfg.data.dir
    dataset_name = cfg.data.dataset_name
    seed = cfg.data.seed
    path = os.path.join(data_dir, dataset_name)

    # optim
    batch_size_train = cfg.optim.batch_size_train
    batch_size_val = cfg.optim.batch_size_val
    epochs = cfg.optim.epochs
    inner_lr = cfg.optim.inner_lr
    outer_lr = cfg.optim.outer_lr
    inner_steps = cfg.optim.inner_steps
    test_inner_steps = cfg.optim.test_inner_steps
    init_type = cfg.optim.init_type
    ctx_dim = cfg.model.ctx_dim

    # model decoder
    state_c = cfg.model.state_c
    hidden_c = cfg.model.hidden_c

    # model forecaster
    method = cfg.model.method
    options = cfg.model.options
    epsilon = cfg.model.teacher_forcing_init
    epsilon_t = cfg.model.teacher_forcing_decay
    epsilon_freq = cfg.model.teacher_forcing_update
    
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
            ("cavia",)
            + (dataset_name,)
            + ("train",)
    )

    run_name = wandb.run.name
    wandb.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    # create model folder
    model_dir = Path(os.getenv("WANDB_DIR")) / 'cavia' / 'new_ver' / 'pretrain'/ dataset_name / "model"
    os.makedirs(str(model_dir), exist_ok=True)

    # set seed
    fix_seed(seed)

    # load data
    train, test, params = init_dataloaders(dataset_name, batch_size_train, batch_size_val, os.path.join(path, dataset_name))
    params = params.to(device)
    num_env = len(params)

    model = Forecaster(dataset_name, state_c, hidden_c, method, options, ctx_dim).to(device)
    init_weights(model, init_config=init_type)

    optimizer = optim.Adam(model.parameters(), outer_lr, betas=(0.9, 0.999))
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

    print("ntrain, ntest : ", ntrain, ntest)
    print("t_in : ", t_in)
    print("dataset_name : ", dataset_name)
    print("path : ", path)
    print("train.dataset[0]['states'].shape : ", train.dataset[0]['states'].shape)
    print("num_env : ", num_env)
    print(f"n_params forecaster: {count_parameters(model)}")
    print("params : ", params)

    # epsilon_t = 0
    for epoch in range(epochs):
        model.train()
        step_show = epoch % nupdate == 0
        step_show_last = epoch == epochs - 1
        loss_train = 0
        contexts = []

        if epoch % epsilon_freq == 0:
            epsilon_t = epsilon_t * epsilon

        for _, data in enumerate(train):
            model.derivative.model_nn.reset_context_params()
            states = data['states'].to(device)
            outputs = outer_step(model, data, inner_steps, inner_lr, epsilon_t, is_train = True)
            n_samples = len(data['states'])
            loss = criterion(outputs, states)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            optimizer.zero_grad()

            loss_train += loss.item() * n_samples
            if True in (step_show, step_show_last):
                contexts.append(model.derivative.model_nn.context_params.cpu().detach())

        loss_train /= ntrain
        scheduler.step(loss_train)

        if True in (step_show, step_show_last):
            loss_test_in = 0
            loss_test_out = 0
            model.eval()

            for i, data_test in enumerate(test):
                
                states = data_test['states'].to(device)
                n_samples = len(states)
                model.derivative.model_nn.context_params = contexts[i // (len(test) // num_env)].to(device)
                outputs = outer_step(model, data_test, test_inner_steps, None, epsilon_t = 0, is_train = False)
                loss_in = criterion_eval(outputs[..., :t_in], states[..., :t_in])
                loss_out = criterion_eval(outputs[..., t_in:], states[..., t_in:])
                loss_test_in += loss_in.item() * n_samples
                loss_test_out += loss_out.item() * n_samples

            loss_test_in /= ntest
            loss_test_out /= ntest

        if True in (step_show, step_show_last):

            wandb.log(
                {
                    "train_loss": loss_train,
                    "loss_test_in": loss_test_in,
                    "loss_test_out": loss_test_out,
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
        else:
            if loss_test_in < best_loss:
                best_loss = loss_test_in
                torch.save(
                    {
                        "cfg": cfg,
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "loss": best_loss,
                    },
                    f"{model_dir}/{run_name}.pt",
                )
    return best_loss

if __name__ == "__main__":
    main()