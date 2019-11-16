import torch
import torch.nn as nn
from KFT.FLOWS.IAF.IAF import IAF_no_h,IAF
class PlanarTransform(nn.Module):
    def __init__(self, dim = 512,init_sigma=0.01):
        super().__init__()
        self.u = nn.Parameter(torch.randn(1, dim).normal_(0, init_sigma))
        self.w = nn.Parameter(torch.randn(1, dim).normal_(0, init_sigma))
        self.b = nn.Parameter(torch.randn(1).fill_(0))

    def forward(self, x, normalize_u=True):
        # allow for a single forward pass over all the transforms in the flows with a Sequential container
        if isinstance(x, tuple):
            z, sum_log_abs_det_jacobians = x
        else:
            z, sum_log_abs_det_jacobians = x, 0
        # print(self.u,self.b,self.w)
        # normalize u s.t. w @ u >= -1; sufficient condition for invertibility
        u_hat = self.u
        if normalize_u:
            wtu = (self.w @ self.u.t()).squeeze()
            m_wtu = - 1 + torch.log1p(wtu.exp())
            u_hat = self.u + (m_wtu - wtu) * self.w / (self.w @ self.w.t())

        # compute transform
        f_z = z + u_hat * torch.tanh(z @ self.w.t() + self.b)
        # compute log_abs_det_jacobian
        psi = (1 - torch.tanh(z @ self.w.t() + self.b)**2) @ self.w
        det = 1 + psi @ u_hat.t()
        log_abs_det_jacobian = torch.log(torch.abs(det) + 1e-6)#.squeeze()
        sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian

        return f_z, sum_log_abs_det_jacobians

class AffineTransform(nn.Module):

    def __init__(self,dim,learnable=False):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(dim)).requires_grad_(learnable)
        self.logsigma = nn.Parameter(torch.zeros(dim)).requires_grad_(learnable)

    def forward(self, x):
        z = self.mu + self.logsigma.exp() * x
        sum_log_abs_det_jacobians = self.logsigma.sum()
        return z, sum_log_abs_det_jacobians


def get_planar_flow(dim=512,K=2,affine=False):
    if affine:
        flow = nn.Sequential(AffineTransform(dim,True),*[PlanarTransform(dim) for _ in range(K)])
    else:
        flow = nn.Sequential(*[PlanarTransform(dim) for _ in range(K)])
    return flow

def get_IAF_h(dim=512,K=3,h_dim = 512,tanh_flag=False,C=100):
    flow = IAF(latent_size=dim,h_size=h_dim,depth=K,tanh_flag=tanh_flag,C=C)
    return flow


def get_IAF(dim=512,K=3,tanh_flag_h=False,C=100):
    flow = IAF_no_h(latent_size=dim,depth=K,tanh_flag_h=tanh_flag_h,C=C)
    return flow

if __name__ == '__main__':
    #TODO test flows and see what they do! Ok they work...
    # flow = get_planar_flow().cuda()
    flow = IAF_no_h(latent_size=512).cuda()
    z = torch.randn((100,512)).cuda()
    f_z,sum_log_abs_det_jacobians = flow(z)
    print(sum_log_abs_det_jacobians)
    #
    # log_q0 = torch.zeros_like(z) #uniform 0,1 distribution f(x) = 1, log(f(x))=0...
    # log_qk = log_q0 - sum_log_abs_det_jacobians
    # qk = log_qk.exp()
    # e = torch.mean(torch.sum(torch.log(qk)*qk,dim=0))
    #
    #
    # print(torch.sum(torch.log(qk)*qk,dim=0))
    # print(e)




