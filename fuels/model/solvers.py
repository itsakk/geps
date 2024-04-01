import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DampedPendulumParamPDE(nn.Module):

    def __init__(self, is_complete = False, code = 2):
        super().__init__()
        self.params = nn.Parameter(torch.cat((torch.full((1, 1), 0.2), torch.full((1, 1), 0.1)), axis = 1))
        self.weight = nn.Parameter(torch.empty(( 2, code)))
        self.is_complete = is_complete

    def forward(self, t, y0, env, codes):
        self.estimated_params = self.params + F.linear(codes, self.weight) # codes = [num_env, code_dim], params = [1, 2], weight = [2, code_dim] => estimated_params = [num_env, 2]
            
        q = y0[:,0:1]
        p = y0[:,1:2]

        if self.is_complete:
            (omega0_square, alpha) = self.estimated_params[env.long(), 0:1], self.estimated_params[env.long(), 1:2]
            dqdt = p
            dpdt = - omega0_square * torch.sin(q) - alpha * p
        else:
            omega0_square = self.estimated_params[env.long(), 0:1]
            dqdt = p
            dpdt = - omega0_square * torch.sin(q)
        return torch.cat([dqdt, dpdt], dim = 1)

class LotkaVolteraPDE(nn.Module):

    def __init__(self, is_complete = False, code = 2):
        super().__init__()
        self.params = nn.Parameter(torch.full((1, 2), 0.1))
        self.weight = nn.Parameter(torch.empty((2, code)))
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
        self.params = nn.Parameter(torch.full((1, 1), 1e-3))
        self.weight = nn.Parameter(torch.empty((1, code)))
        self.dx = 128
        self.fdx = 2*np.pi/self.dx

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
        return (
                - 2 * a
                + torch.roll(a, +1) 
                + torch.roll(a, -1)
            ) / (self.fdx ** 2)

    def _laplacian1Dbis(self, a):
        return (
            + 2 * a
            + torch.roll(a,+1) 
            + torch.roll(a,-1)
            ) / (self.fdx ** 2)
    
    # def _fluxRight(self,a):
    #     return(
    #         self._mean_f(1/2*a**2)
    #         - 1/6*self._derive1(self._derive1(self._mean_f(1/2*a**2)))
    #         + 1/12 * 1/2 * self._derive1(self._derive1(self._derive1(a)))
    #     )

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

    def __init__(self, is_complete, code):
        super().__init__()

        self._dx = 1
        self.is_complete = is_complete
        self.params = nn.Parameter(torch.full((1, 2), 0.05))
        self.weight = nn.Parameter(torch.empty((2, code)))
        self._laplacian = nn.Parameter(torch.tensor(
            [
                [ 0.25,  0.5, 0.25],
                [ 0.50, -3.0, 0.50],
                [ 0.25,  0.5, 0.25],
            ],
        ).float().view(1, 1, 3, 3) / (self._dx ** 2), requires_grad=False)
    
    def forward(self, t, y0, env, codes):
        self.estimated_params = self.params + F.linear(codes, self.weight)
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

    def __init__(self, code):
        super().__init__()
        self.params = nn.Parameter(torch.cat((torch.full((1, 1), 5e-3), torch.full((1, 1), 1)), axis = 1))
        self.weight = nn.Parameter(torch.empty((2, code)))
        self.N = 256
        self.dx = np.pi / self.N

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
    
    def laplacian2D(self,a, fdx):
        return (
            - 4 * a
            + torch.roll(a,+1,dims=1) 
            + torch.roll(a,-1,dims=1)
            + torch.roll(a,+1,dims=2)
            + torch.roll(a,-1,dims=2)
            ) / (fdx ** 2)
    
    def ps(self, N, fdx, w):
        w_pad = F.pad(w,pad=(0,1,0,1),mode="circular")
        wf = torch.fft.fft2(1/N**2*w_pad)

        # with torch.no_grad():
        kxxs, kyys = [], []
        for dx in fdx[:, 0]:
            kxx, kyy = torch.meshgrid(2*np.pi*torch.fft.fftfreq(self.N+1, dx.item(), device="cuda"), 2*np.pi*torch.fft.fftfreq(self.N+1, dx.item(), device="cuda"))
            kxxs.append(kxx.unsqueeze(0))
            kyys.append(kyy.unsqueeze(0))
        kxxs = torch.stack(kxxs, dim = 0)
        kyys = torch.stack(kyys, dim = 0)

        uf = -wf/(1e-12 + kxxs**2 + kyys**2)
        uf[:, 0, 0] = 0
        return (torch.fft.ifft2(N**2*uf).real)[...,:self.N,:self.N]
    
    def f2(self, x, env, codes):
        self.estimated_params = self.params + F.linear(codes, self.weight)
        params = self.estimated_params[env.long(), :]
        mu, domain = params[:, 0:1], params[:, 1:2]
        fdx = domain * self.dx
        psi = self.ps(self.N, fdx, -x)

        J1 =  (self.derive_y(psi)*self.derive_x(x) -self.derive_x(psi)*self.derive_y(x)) / (4* fdx**2)

        J2 = (torch.roll(x,-1,dims=2)*(self.derive_y(torch.roll(psi,-1,dims=2))) \
            - torch.roll(x,+1,dims=2)*(self.derive_y(torch.roll(psi,+1,dims=2))) \
            - torch.roll(x,-1,dims=1)*(self.derive_x(torch.roll(psi,-1,dims=1))) \
            + torch.roll(x,+1,dims=1)*(self.derive_x(torch.roll(psi,+1,dims=1)))) / (4* fdx**2)
        
        J3 = (torch.roll(torch.roll(x,-1,dims=2),-1,dims=1)*(torch.roll(psi,-1,dims=1)-torch.roll(psi,-1,dims=2))\
            -torch.roll(torch.roll(x,+1,dims=2),+1,dims=1)*(torch.roll(psi,+1,dims=2)-torch.roll(psi,+1,dims=1))\
            -torch.roll(torch.roll(x,+1,dims=2),-1,dims=1)*(torch.roll(psi,-1,dims=1)-torch.roll(psi,+1,dims=2))\
            +torch.roll(torch.roll(x,-1,dims=2),+1,dims=1)*(torch.roll(psi,-1,dims=2)-torch.roll(psi,+1,dims=1))) / (4* fdx**2)

        return (-(J1+J2+J3)/3 + mu*self.laplacian2D(x, fdx))
        
    def forward(self, t, y0, env, codes):
        dudt = self.f2(y0[:, 0], env, codes)
        return dudt
        
def get_numerical_solver(dataset_name, code_c, is_complete):
    if dataset_name == 'pendulum':
        model_phy = DampedPendulumParamPDE(is_complete=is_complete, code = code_c)
    elif dataset_name == 'lv':
        model_phy = LotkaVolteraPDE(is_complete=is_complete, code = code_c)
    elif dataset_name == 'gs':
        model_phy = GrayScottParamPDE(is_complete=is_complete, code = code_c)
    elif dataset_name == 'burgers':
        model_phy = BurgersParamPDE(code = code_c)
    elif dataset_name == 'kolmo':
        model_phy = Turb2dPDE(code = code_c)
    return model_phy