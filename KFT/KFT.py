import torch
import pykeops.torch as pktorch
import pykeops
import tensorly
tensorly.set_backend('pytorch')
import gpytorch
from pykeops.torch import LazyTensor as keops
from tensorly.base import fold

def keops_mode_product(T,K,mode):
    """
    :param T: Pytorch tensor, just pass as usual
    :param K: Keops Lazytensor object, remember should be on form MxN
    :param mode: Mode of tensor
    :return:
    """
    t_new_shape = list(T.shape)
    t_new_shape[mode] = K.shape[0]
    T = K @ torch.reshape(torch.transpose(T, mode, 0), (T.shape[mode], -1))
    T = fold(unfolded_tensor=T,mode=mode,shape=t_new_shape)
    return T

class TT_component(torch.nn.Module):
    def __init__(self,r_1,n_list,r_2):
        super(TT_component, self).__init__()
        self.n_dict = {i+1:None for i in range(len(n_list))} #{mode:side_info}
        self.shape_list  = [n_i for n_i in n_list].insert(0,r_1).append(r_2)
        self.TT_core = torch.nn.Parameter(torch.randn_like(*self.shape_list),requires_grad=True)
        self.permutation_list = [i+1 for i in range(len(n_list))] + [0,-1]
    def forward(self,indices): #For tensors with no side info #just use gather
        if indices.shape>1:
            indices = indices.unbind(1)
        return self.TT_core.permute(self.permutation_list)[indices]


class TT_kernel_component(TT_component): #for tensors with full or "mixed" side info
    def __init__(self,r_1,n_list,r_2,side_information_dict,kernel_para_dict):
        super(TT_kernel_component, self).__init__(r_1,n_list,r_2)
        self.keys = []
        for key,value in side_information_dict.items(): #Should be on the form {mode: side_info}
            setattr(self, f'kernel_{key}', gpytorch.kernels.keops.RBFKernel())
            setattr(self, f'kernel_data_{key}',torch.nn.Parameter(value,requires_grad=False))
            self.keys.append(key)
            gwidth0 = self.get_median_ls(value)
            self.gamma_sq_init = torch.tensor(gwidth0).float() * kernel_para_dict['ls_factor']
            getattr(self,f'kernel_{key}').raw_lengthscale = torch.nn.Parameter(self.gamma_sq_init,requires_grad=False)

        for key in self.keys:
            k_data = getattr(self,f'kernel_data_{key}')
            tmp_kernel_func = getattr(self,f'kernel_{key}')
            self.n_dict[key] =  tmp_kernel_func(k_data,k_data) #lazily executed tensors, should have a bunch of lazy tensors...

    def get_median_ls(self,X):  # Super LS and init value sensitive wtf
        self.kernel_base = gpytorch.kernels.Kernel()
        if X.shape[0] > 20000:
            idx = torch.randperm(20000)
            X = X[idx, :]
        d = self.kernel_base.covar_dist(X, X)
        return torch.sqrt(torch.median(d[d > 0]))

    def forward(self,indices):
        """Do tensor ops"""
        T = self.TT_core
        for key,val in self.n_dict.items():
            if val is not None:
                T = keops_mode_product(T,val,key)
        if indices.shape>1:
            indices = indices.unbind(1)
        return T.permute(self.permutation_list)[indices]


if __name__ == '__main__':
    print('rip')