import torch
import torch.nn as nn

class META(nn.Module):
    def __init__(self, model, model_target, inner_steps, ctx_dim, inner_lr, tau, device = 'cuda'):
        super().__init__()

        self.model = model
        self.model_target = model_target
        self.inner_steps = inner_steps
        self.ctx_dim = ctx_dim
        self.inner_lr = inner_lr
        self.tau = tau
        self.device = device

    def inner_loop(
        self,
        batch,
        ctx,
        epsilon = 0,
    ):
        for step in range(self.inner_steps):
            ctx = self.inner_loop_step(
                batch,
                ctx,
                epsilon,
            )
        return ctx

    def inner_loop_step(
        self,
        batch,
        ctx = None,
        epsilon = 0,
    ):
        """Performs a single inner loop step."""
        criterion = nn.MSELoss()

        targets = batch['states'].to(self.device)
        t = batch['t'][0].to(self.device)
        batch_size = len(targets)

        outputs = self.model_target(targets, t, ctx, epsilon)
        loss = criterion(outputs, targets).mean() * batch_size
        grad = torch.autograd.grad(
            loss,
            ctx,
        )[0]
        # Perform single gradient descent step
        return ctx - self.inner_lr * grad

    def outer_step(
        self,
        batch,
        epsilon = 0,
        is_train=False,
        ctx = None,
    ):

        targets = batch['states'].to(self.device)
        t = batch['t'][0].to(self.device)

        if is_train:
            ctx = nn.Parameter(torch.zeros(1, self.ctx_dim, requires_grad=True).to(self.device))
            ctx = ctx.repeat(len(targets), 1)

            self.update_target()
            ctx = self.inner_loop(
                batch,
                ctx,
                epsilon,
            )
        else:
            ctx = ctx.repeat(len(targets), 1)

        outputs = self.model(targets, t, ctx, epsilon)
        return outputs, ctx

    def update_target(self):
        for param_target, param in zip(self.model_target.parameters(), self.model.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)