import torch
from KFT.KFT_fp_16 import KFT, variational_KFT
from tqdm import tqdm
from torch.nn.modules.loss import _Loss
from apex import amp
import apex
import pandas as pd

class Log1PlusExp(torch.autograd.Function):
    """Implementation of x ↦ log(1 + exp(x))."""
    @staticmethod
    def forward(ctx, x):
        exp = x.exp()
        ctx.save_for_backward(x)
        y = exp.log1p()
        return x.where(torch.isinf(exp),y.half() if x.type()=='torch.cuda.HalfTensor' else y )

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        y = (-x).exp().half() if x.type()=='torch.cuda.HalfTensor' else (-x).exp()
        return grad_output / (1 + y)

class stableBCEwithlogits(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(stableBCEwithlogits, self).__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        self.f = Log1PlusExp.apply

    def forward(self, x, y):
        return torch.mean(self.f(x)-x*y)

def train_loop(model,train_config,dataloader_train, dataloader_val, dataloader_test):
    opt = torch.optim.Adam(model.parameters(), lr=train_config['lr'])
    if train_config['fp_16']:
        if train_config['fused']:
            del opt
            opt = apex.optimizers.FusedAdam(model.parameters(), lr=train_config['lr'])
            [model], [opt] = amp.initialize([model],[opt], opt_level='O1',num_losses=1)
        else:
            [model], [opt] = amp.initialize([model],[opt], opt_level='O1',num_losses=1)
    if train_config['task']=='reg':
        loss_func = torch.nn.MSELoss()
    else:
        loss_func = stableBCEwithlogits()

    for i in tqdm(range(train_config['epochs']+1)):
        for j,X,y in enumerate(dataloader_train):
            y_pred, reg = model(X)
            pred_loss = loss_func(y_pred,y)
            total_loss = pred_loss + reg
            opt.zero_grad()
            if train_config['fp_16']:
                with amp.scale_loss(total_loss, opt, loss_id=0) as loss_scaled:
                    loss_scaled.backward()
            else:
                total_loss.backward()
            opt.step()

            if j%train_config['train_loss_interval_print']:
                print(f'reg_term epoch {j}: {reg.data}')
                print(f'train_loss epoch {j}: {pred_loss.data}')

        with torch.no_grad():
            for i, X, y in enumerate(dataloader_val):
                y_pred, _ = model(X)
                val_loss = loss_func(y_pred,y)
            print(f'validation_loss epoch {i}: {val_loss.data}')

    val_loss_final = val_loss.data
    with torch.no_grad():
        for i, X, y in enumerate(dataloader_test):
            y_pred, _ = model(X)
            test_loss = loss_func(y_pred,y)
    test_loss_final = test_loss.data
    return val_loss_final,test_loss_final,model

class job_object():
    def __init__(self,side_info_dict,tensor_architecture,kernel_setup):
        """
        :param side_info_dict: Dict containing side info EX) {ii:side_info,...}
        :param tensor_architecture: Tensor architecture  EX) {0:{ii:[0,1],...}
        :param kernel_setup {1:{'kernel_type':'matern'...,}}
        """
        self.side_info = side_info_dict
        self.tensor_architecture = tensor_architecture
        self.kernel_setup = kernel_setup

    def construct_init_dict(self):

        pass
