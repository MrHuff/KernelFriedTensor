import math

import torch as t
import torch.nn as nn
from torch.nn.init import xavier_normal,xavier_normal_
from torch.nn.parameter import Parameter


class AutoregressiveLinear(nn.Module):
    def __init__(self, in_size, out_size, bias=True, ):
        super(AutoregressiveLinear, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.weight = Parameter(t.randn(*(self.in_size, self.out_size)),requires_grad=True)

        if bias:
            self.bias = Parameter(t.randn(self.out_size),requires_grad=True)
        else:
            self.register_buffer('bias', t.tensor(0))

        self.reset_parameters()

    def reset_parameters(self, ):
        stdv = 1. / math.sqrt(self.out_size)

        self.weight = xavier_normal_(self.weight)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if input.dim() == 2 and self.bias is not None:
            return t.addmm(self.bias, input, self.weight.tril(-1))

        output = input @ self.weight.tril(-1)
        if self.bias is not None:
            output += self.bias
        return output