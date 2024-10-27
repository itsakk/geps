import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DampedDrivenPDE(nn.Module):

    def __init__(self, is_complete = False, code = 2):
        super().__init__()
        self.params = nn.Parameter(torch.cat((torch.full((1, 1), 0.2), torch.full((1, 1), 0.1), torch.full((1, 1), 0.2), torch.full((1, 1), 0.5)), axis = 1))
        self.weight = nn.Parameter(torch.empty((4, code)))
        self.is_complete = is_complete

    def forward(self, t, y0, env, codes):
        self.estimated_params = self.params + F.linear(codes, self.weight) # codes = [num_env, code_dim], params = [1, 2], weight = [2, code_dim] => estimated_params = [num_env, 2]
        # self.estimated_params = self.params

        q = y0[:,0:1]
        p = y0[:,1:2]

        if self.is_complete:
            (omega0_square, alpha) = self.estimated_params[env.long(), 0:1], self.estimated_params[env.long(), 1:2]
            (f0, wf) = self.estimated_params[env.long(), 2:3], self.estimated_params[env.long(), 3:4]
            dqdt = p
            dpdt = - omega0_square * torch.sin(q) - alpha * p + f0 * torch.cos(wf * t)
        else:
            omega0_square = self.estimated_params[env.long(), 0:1]
            (f0, wf) = self.estimated_params[env.long(), 2:3], self.estimated_params[env.long(), 3:4]
            dqdt = p
            dpdt = - omega0_square * torch.sin(q) + f0 * torch.cos(wf * t)
        return torch.cat([dqdt, dpdt], dim = 1)

class LotkaVolteraPDE(nn.Module):

    def __init__(self, is_complete = False, code = 2):
        super().__init__()
        self.params = nn.Parameter(torch.full((1, 2), 0.1))
        #self.weight = nn.Parameter(torch.empty((2, code)))
        self.is_complete = is_complete

    def forward(self, t, y0, env, codes):
        self.estimated_params = self.params + F.linear(codes, self.weight)

        # params 
        alpha = gamma = 0.5
        delta = self.estimated_params[env.long(), 1:2]
        beta = self.estimated_params[env.long(), 0:1]

        x = y0[:, 0:1]
        y = y0[:, 1:2]

        if self.is_complete:
            dxdt = alpha * x - beta * x * y
            dydt = delta * x * y - gamma * y
        else:
            dxdt = alpha * x - beta * y
            dydt = delta * x - gamma * y
        return torch.cat([dxdt, dydt], dim = 1)

class BurgersParamPDE(nn.Module):
    
    def __init__(self, code = 2):
        super().__init__()
        self.params = nn.Parameter(torch.full((1, 1), 1e-2))
        self.weight = nn.Parameter(torch.empty((1, code)))
        self.dx = 128
        self.fdx = 2 * np.pi / self.dx
        self._laplacian = nn.Parameter(torch.tensor(
                    [
                        [ -1,  2, -1],
                    ],
                ).float().view(1, 1, 3) / (self.fdx ** 2), requires_grad=False)
        
    def _derive1(self, a):
        return(
                torch.roll(a, 1)
                -a
            )

    def _mean_f(self, a):
        return(
                torch.roll(a, 1)
                +a
            )/2

    def _laplacian1D(self, a):
        a = F.pad(a.unsqueeze(1), pad=(1, 1), mode='circular')
        return F.conv1d(a, self._laplacian).squeeze(1)

    def _fluxRight(self, a):
        return(
                + self._mean_f(1/2*a**2)
                - 1/6*self._derive1(self._derive1(self._mean_f(1/2*a**2)))
                + 1/30*self._derive1(self._derive1(self._derive1(self._derive1(self._mean_f(1/2*a**2)))))
        )

    def forward(self, t, y0, env, codes):
        U = y0[:, 0, :]
        self.estimated_params = self.params + F.linear(codes, self.weight)
        mu = self.estimated_params[env.long(), 0:1]

        U_ = U
        dUdt = -(self._fluxRight(U_)-self._fluxRight(torch.roll(U_,-1)))/self.fdx + torch.mul(mu,self._laplacian1D(U_))
        return dUdt.unsqueeze(1)

class GrayScottParamPDE(nn.Module):

    def __init__(self, is_complete, code, n_env):
        super().__init__()

        self._dx = 1
        self.is_complete = is_complete
        self.params = nn.Parameter(torch.full((n_env, 2), 0.05))
        self.weight = nn.Parameter(torch.empty((2, code)))
        self._laplacian = nn.Parameter(torch.tensor(
            [
                [ 0.25,  0.5, 0.25],
                [ 0.50, -3.0, 0.50],
                [ 0.25,  0.5, 0.25],
            ],
        ).float().view(1, 1, 3, 3) / (self._dx ** 2), requires_grad=False)
    
    def forward(self, t, y0, env, codes):
        self.estimated_params = self.params # + F.linear(codes, self.weight)
        params = self.estimated_params[env.long(), :]

        f, k = params[:, 0:1, None, None], params[:, 1:2, None, None]
        r_u, r_v = 0.2097, 0.105

        U = y0[:, :1]
        V = y0[:, 1:]
        
        U_ = F.pad(U, pad=(1, 1, 1, 1), mode='circular')
        Delta_u = F.conv2d(U_, self._laplacian)
        
        V_ = F.pad(V, pad=(1, 1, 1, 1), mode='circular')
        Delta_v = F.conv2d(V_, self._laplacian)

        if self.is_complete:
            dUdt = r_u * Delta_u - U * (V ** 2) + f * (1. - U)
            dVdt = r_v * Delta_v + U * (V ** 2) - (f + k) * V
        else:
            dUdt = r_u * Delta_u - U * (V ** 2) 
            dVdt = r_v * Delta_v + U * (V ** 2)
        return torch.cat([dUdt, dVdt], dim=1)
    
class Turb2dPDE(nn.Module):

    def __init__(self, code, n_env):
        super().__init__()
        self.params = nn.Parameter(torch.cat((torch.full((n_env, 1), 5e-3), torch.full((n_env, 1), 1)), axis = 1))
        self.weight = nn.Parameter(torch.empty((2, code)))
        self.N = 64
        self.dx = np.pi / self.N
        self._laplacian = nn.Parameter(torch.tensor(
            [
                [ 0.0,  1.0, 0.0],
                [ 1.0, -4.0, 1.0],
                [ 1.0,  1.0, 0.0],
            ],
        ).float().view(1, 1, 3, 3), requires_grad=False)
        self.domain = torch.Tensor([0.75, 1.0, 0.75, 1.0, 0.75, 1.0, 0.75, 1.0]).unsqueeze(-1).cuda() if n_env == 8 else torch.Tensor([1.5, 2.0, 1.5, 2.0]).unsqueeze(-1).cuda()

    def derive_x(self,a):
        return(
            torch.roll(a,-1,dims=2)
            -torch.roll(a,+1,dims=2)
        )

    def derive_y(self,a):
        return(
            torch.roll(a,-1,dims=1)
            -torch.roll(a,+1,dims=1)
            )
    
    def _laplacian2D(self, a, fdx):
        a = F.pad(a.unsqueeze(0), pad=(1, 1, 1, 1), mode='circular')
        fdx = fdx.unsqueeze(-1)
        kernel = self._laplacian / (fdx ** 2)
        a = F.conv2d(a, kernel, groups = a.shape[1]).squeeze(0)
        return a

    def ps(self, N, fdx, w):
        w_pad = F.pad(w,pad=(0,1,0,1),mode="circular")
        wf = torch.fft.fft2(1/N**2*w_pad)

        # with torch.no_grad():
        kxxs, kyys = [], []
        for dx in fdx[:, 0]:
            kxx, kyy = torch.meshgrid(2*np.pi*torch.fft.fftfreq(self.N+1, dx.item(), device="cuda"), 2*np.pi*torch.fft.fftfreq(self.N+1, dx.item(), device="cuda"))
            kxxs.append(kxx)
            kyys.append(kyy)
        kxxs = torch.stack(kxxs, dim = 0)
        kyys = torch.stack(kyys, dim = 0)

        uf = -wf/(1e-12 + kxxs**2 + kyys**2)
        uf[:, 0, 0] = 0
        return (torch.fft.ifft2(N**2*uf).real)[...,:self.N,:self.N]

    def f2(self, x, env, codes):
        self.estimated_params = self.params #+ F.linear(codes, self.weight)
        params = self.estimated_params[env.long(), :]
        mu, domain = params[:, 0:1], params[:, 1:2]
    
        domain = self.domain[env.long()]
        fdx = domain * self.dx
        psi = self.ps(self.N, fdx, -x)
        mu = mu[..., None]
        fdx = fdx.unsqueeze(-1)
        J1 =  (self.derive_y(psi)*self.derive_x(x) -self.derive_x(psi)*self.derive_y(x)) / (4 * fdx**2)

        J2 = (torch.roll(x,-1,dims=2)*(self.derive_y(torch.roll(psi,-1,dims=2))) \
            - torch.roll(x,+1,dims=2)*(self.derive_y(torch.roll(psi,+1,dims=2))) \
            - torch.roll(x,-1,dims=1)*(self.derive_x(torch.roll(psi,-1,dims=1))) \
            + torch.roll(x,+1,dims=1)*(self.derive_x(torch.roll(psi,+1,dims=1)))) / (4* fdx**2)
        
        J3 = (torch.roll(torch.roll(x,-1,dims=2),-1,dims=1)*(torch.roll(psi,-1,dims=1)-torch.roll(psi,-1,dims=2))\
            -torch.roll(torch.roll(x,+1,dims=2),+1,dims=1)*(torch.roll(psi,+1,dims=2)-torch.roll(psi,+1,dims=1))\
            -torch.roll(torch.roll(x,+1,dims=2),-1,dims=1)*(torch.roll(psi,-1,dims=1)-torch.roll(psi,+1,dims=2))\
            +torch.roll(torch.roll(x,-1,dims=2),+1,dims=1)*(torch.roll(psi,-1,dims=2)-torch.roll(psi,+1,dims=1))) / (4* fdx**2)
        return (-(J1+J2+J3)/3 + mu*self._laplacian2D(x, fdx))
        
    def forward(self, t, y0, env, codes):
        dudt = self.f2(y0[:, 0], env, codes).unsqueeze(1)
        return dudt
        
class Combined_equation(nn.Module):
    def __init__(self,
                 code,
                 n_env=None
                 ):
        super().__init__()
        self.params = nn.Parameter(torch.full((n_env, 4), 0.1))
        self.weight = nn.Parameter(torch.empty((4, code)))
        self.L=32
        
    def psdiff(self, x, order=1, period=None, axis=-1):
        if period is None:
            period = 2 * torch.pi

        # Compute FFT
        ft = torch.fft.fft(x, dim=axis)
        
        # Generate frequency array
        n = x.shape[axis]
        freq = torch.fft.fftfreq(n, d=1.0/n)
        
        # Reshape freq to match the dimensionality of ft
        shape = [1] * len(x.shape)
        shape[axis] = -1
        freq = freq.reshape(shape)
        
        # Compute the multiplier
        multiplier = (2j * torch.pi * freq / period) ** order
        
        # Apply the multiplier
        ft_diff = ft * multiplier.cuda()
        
        # Inverse FFT
        return torch.fft.ifft(ft_diff, dim=axis).real
    
    def forward(self, t, u, env, codes):
        self.estimated_params = self.params + F.linear(codes, self.weight)
        params = self.estimated_params[env.long(), :]
        alpha, beta, delta, gamma = params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4]
        ux = u * self.psdiff(u, period=self.L)
        uxx = self.psdiff(u, order=2, period=self.L)
        uxxx = self.psdiff(u, order=3, period=self.L)
        uxxxx = self.psdiff(u,order=4, period=self.L)
        dudt = - alpha*ux + beta*uxx - delta*uxxx - gamma*uxxxx
        return dudt

def get_numerical_solver(dataset_name, code_c, is_complete, n_env):
    if dataset_name == 'pendulum':
        model_phy = DampedDrivenPDE(is_complete=is_complete, code = code_c)
    elif dataset_name == 'lv':
        model_phy = LotkaVolteraPDE(is_complete=is_complete, code = code_c)
    elif "gs" in dataset_name:
        model_phy = GrayScottParamPDE(is_complete=is_complete, code = code_c, n_env = n_env)
    elif "burgers" in dataset_name:
        model_phy = BurgersParamPDE(code = code_c)
    elif dataset_name == 'kolmo':
        model_phy = Turb2dPDE(code = code_c, n_env = n_env)
    elif dataset_name == 'combined':
        model_phy = Combined_equation(code = code_c, n_env = n_env)
    else:
        model_phy = None
    return model_phy