import torch as t
import torch.nn as nn
from collections import OrderedDict
from KFT.FLOWS.IAF.autoregressive_linear import AutoregressiveLinear
from KFT.FLOWS.IAF.highway import Highway

class IAF_no_h(nn.Module):
    def __init__(self, latent_size, depth,tanh_flag_h=False,C=100):
        super(IAF_no_h, self).__init__()
        self.C = C
        self.depth = depth
        self.z_size = latent_size
        self.tanh_op = nn.Tanh()
        self.flag = tanh_flag_h
        self.s_list = nn.ModuleList(
            [nn.Sequential(AutoregressiveLinear(self.z_size , self.z_size), nn.ELU(),nn.Sigmoid()) for i
             in range(depth)])
        self.m_list = nn.ModuleList(
            [nn.Sequential(AutoregressiveLinear(self.z_size , self.z_size), nn.ELU()) for i
             in range(depth)])

    def forward(self, z):
        """
        :param z: An float tensor with shape of [batch_size, z_size]
        :param h: An float tensor with shape of [batch_size, h_size]
        :return: An float tensor with shape of [batch_size, z_size] and log det value of the IAF mapping Jacobian
        """
        log_det = 0
        for i in range(self.depth):
            m = self.m_list[i](z)
            s = self.s_list[i](z)
            z = s * z + (1 - s) * m
            log_det = log_det - s.log().sum(1)
        if self.flag:
            z = self.tanh_op(z/self.C)*self.C
        return z, -log_det

class IAF(nn.Module):
    def __init__(self, latent_size, h_size,depth,tanh_flag=False,C=100):
        super(IAF, self).__init__()
        self.depth = depth
        self.z_size = latent_size
        self.h_size = h_size
        self.tanh_op = nn.Tanh()
        self.flag = tanh_flag
        self.h = Highway(self.h_size, 3, nn.ELU())
        self.C = C
        self.z_size = latent_size
        self.s_list = nn.ModuleList([nn.Sequential(AutoregressiveLinear(self.z_size+self.h_size, self.z_size),nn.ELU(),nn.Sigmoid()) for i in range(depth)])
        self.m_list = nn.ModuleList([nn.Sequential(AutoregressiveLinear(self.z_size+self.h_size, self.z_size),nn.ELU()) for i in range(depth)])

    def forward(self, z, h):
        """
        :param z: An float tensor with shape of [batch_size, z_size]
        :param h: An float tensor with shape of [batch_size, h_size]
        :return: An float tensor with shape of [batch_size, z_size] and log det value of the IAF mapping Jacobian
        """
        h = self.h(h)
        log_det = 0
        for i in range(self.depth):
            input = t.cat([z, h], 1)
            m = self.m_list[i](input)
            s = self.s_list[i](input)
            z = s*z+(1-s)*m
            log_det = log_det - s.log().sum(1)
        if self.flag:
            z = self.tanh_op(z/self.C)*self.C
        return z, -log_det

