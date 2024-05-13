import torch
import torch.nn as nn

def inner_loop(
    model,
    batch,
    inner_steps,
    inner_lr,
    epsilon_t = 0,
):
    for step in range(inner_steps):
        outputs = inner_loop_step(
            model,
            batch,
            inner_lr,
            epsilon_t,
    )
    return outputs

def inner_loop_step(
    model,
    batch,
    inner_lr,
    epsilon_t = 0,
    device = 'cuda',
):
    """Performs a single inner loop step."""
    criterion = nn.MSELoss()
    targets = batch['states'].to(device)
    t = batch['t'][0].to(device)
    batch_size = len(targets)
    outputs = model(targets, t, epsilon_t)
    loss = criterion(outputs, targets).mean() * batch_size
    grad = torch.autograd.grad(
        loss,
        model.derivative.model_nn.context_params,
    )[0]
    # Perform single gradient descent step
    model.derivative.model_nn.context_params =  model.derivative.model_nn.context_params - inner_lr * grad
    return None

def outer_step(
    model,
    batch,
    inner_steps,
    inner_lr,
    epsilon_t = 0,
    is_train=False,
    device = 'cuda',
):
    
    if is_train:
        inner_loop(
            model,
            batch,
            inner_steps,
            inner_lr,
        )

    targets = batch['states'].to(device)
    t = batch['t'][0].to(device)

    with torch.set_grad_enabled(is_train):
        outputs = model(targets, t, epsilon_t)
    return outputs