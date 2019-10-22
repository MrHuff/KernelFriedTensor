import torch
import tensorly
tensorly.set_backend('pytorch')
import gpytorch
from pykeops.torch import LazyTensor as keops
# from tensorly.base import fold
import math

PI  = math.pi

class keops_periodic():
    def __init__(self,p=None,ls=None,fixed_Y = None): #assuming p and ls to be torch parameters...
        self.p = p
        self.raw_lengthscale = ls
        if fixed_Y is not None:
            self.fixed_Y = keops(fixed_Y,axis=1)

    def __call__(self,X,Y=None):
        x = keops(X,axis=0)
        ls = keops(self.raw_lengthscale.data)
        p = keops(self.p.data)
        if Y is not None:
            y = keops(Y, axis=1)
            D = ((x - y).abs()).sum(-1)
        else:
            D = ((x - self.fixed_Y).abs()).sum(-1)
        K = ((-2 * (PI * D / p).sin() ** 2) / ls ** 2).exp()
        return K

# def lazy_ones(ones):
#     x = keops(ones.data,axis=0)
#     y = keops(ones.data,axis=1)
#     with torch.no_grad():
#         ones = (x*y).sum(dim=-1)
#     return ones



# def fold(unfolded_tensor, mode, shape):
#     full_shape = list(shape)
#     mode_dim = full_shape.pop(mode)
#     full_shape.insert(0, mode_dim)
#     return torch.transpose(torch.reshape(unfolded_tensor, full_shape).contiguous(), 0, mode).contiguous()

def keops_mode_product(T,K,mode):
    """
    :param T: Pytorch tensor, just pass as usual
    :param K: Keops Lazytensor object, remember should be on form MxN
    :param mode: Mode of tensor
    :return:
    """
    t_new_shape = list(T.shape)
    t_new_shape[mode] = K.shape[0]
    mode_dim = t_new_shape.pop(mode)
    t_new_shape.insert(0, mode_dim)
    T = K @ torch.reshape(torch.transpose(T, mode, 0), (T.shape[mode], -1)).contiguous()
    T = torch.transpose(torch.reshape(T, t_new_shape), 0, mode).contiguous()
    return T

class TT_component(torch.nn.Module):
    def __init__(self,r_1,n_list,r_2,cuda=None):
        super(TT_component, self).__init__()
        # for i, n in enumerate(n_list):
        #     if cuda is not None:
        #         setattr(self,f'ones_{i}',torch.nn.Parameter(torch.ones(*(n, 1)).contiguous().to(cuda),requires_grad=False))
        #     else:
        #         setattr(self,f'ones_{i}',torch.nn.Parameter(torch.ones(*(n, 1)).contiguous(),requires_grad=False))
        self.dummy_kernel = gpytorch.kernels.keops.RBFKernel()
        for p in self.dummy_kernel.parameters():
            p.requires_grad = False
        self.n_dict = {i + 1: None for i in range(len(n_list))}
        self.shape_list  = [r_1]+[n for n in n_list] + [r_2]
        self.permutation_list = [i + 1 for i in range(len(n_list))] + [0, -1]
        # self.reg_ones = {i + 1: lazy_ones(getattr(self,f'ones_{i}')) for i in range(len(n_list))}
        self.reg_ones = {i + 1: self.lazy_ones(n,cuda) for i,n in enumerate(n_list)}
        self.TT_core = torch.nn.Parameter(torch.randn(*self.shape_list),requires_grad=True)

    def lazy_ones(self,n, cuda):
        if cuda is not None:
            test = torch.zeros(*(n, 1), requires_grad=False).contiguous().to(cuda)
        else:
            test = torch.ones(*(n, 1), requires_grad=False).contiguous()

        # x = keops(test, axis=0)
        # y = keops(test, axis=1)
        # ones = (x + y).sum(dim=-1).exp()
        ones = self.dummy_kernel(test,test)
        return ones

    def forward(self,indices): #For tensors with no side info #just use gather
        if len(indices.shape)>1:
            indices = indices.unbind(1)
        return self.TT_core.permute(self.permutation_list)[indices]

    def get_aux_reg_term(self):
        T = self.TT_core**2
        for mode,ones in self.reg_ones.items():
            T = keops_mode_product(T,ones,mode)
        return T

class TT_kernel_component(TT_component): #for tensors with full or "mixed" side info
    def __init__(self,r_1,n_list,r_2,side_information_dict,kernel_para_dict,cuda=None):
        super(TT_kernel_component, self).__init__(r_1,n_list,r_2,cuda)
        self.keys = []
        for key,value in side_information_dict.items(): #Should be on the form {mode: side_info}
            self.assign_kernel(key,value,kernel_para_dict)

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

    def assign_kernel(self,key,value,kernel_para_dict):
        self.keys.append(key)
        gwidth0 = self.get_median_ls(value)
        self.gamma_sq_init = gwidth0 * kernel_para_dict['ls_factor']
        setattr(self, f'kernel_data_{key}', torch.nn.Parameter(value, requires_grad=False))
        if kernel_para_dict['kernel_type']=='rbf':
            setattr(self, f'kernel_{key}', gpytorch.kernels.keops.RBFKernel())
        elif kernel_para_dict['kernel_type']=='matern':
            setattr(self, f'kernel_{key}', gpytorch.kernels.keops.MaternKernel(nu=kernel_para_dict['nu']))
        elif kernel_para_dict['kernel_type']=='periodic':
            setattr(self, f'kernel_{key}', keops_periodic(p= torch.nn.Parameter(torch.tensor(kernel_para_dict['p']),requires_grad=False)))
        getattr(self, f'kernel_{key}').raw_lengthscale = torch.nn.Parameter(self.gamma_sq_init, requires_grad=False)

    def forward(self,indices):
        """Do tensor ops"""
        T = self.TT_core
        for key,val in self.n_dict.items():
            if val is not None:
                T = keops_mode_product(T,val,key)
        if len(indices.shape)>1:
            indices = indices.unbind(1)
        return T.permute(self.permutation_list)[indices], T  #return both to calculate regularization when doing frequentist

class KFT(torch.nn.Module):
    def __init__(self,initializaiton_data,cuda=None): #decomposition_data = {0:{'ii':[0,1],'lambda':0.01,r_1:1 n_list=[10,10],r_2:10,'has_side_info':True, side_info:{1:x_1,2:x_2},kernel_para:{'ls_factor':0.5, 'kernel_type':'RBF','nu':2.5} },1:{}}
        super(KFT, self).__init__()
        tmp_dict = {}
        tmp_dict_prime = {}
        lambdas = []
        self.ii = {}
        for i,v in initializaiton_data.items():
            self.ii[i] = v['ii']
            tmp_dict_prime[str(i)] = TT_component(r_1=v['r_1'],n_list=v['n_list'],r_2=v['r_2'],cuda=cuda)
            if v['has_side_info']:
                tmp_dict[str(i)] = TT_kernel_component(r_1=v['r_1'],
                                                       n_list=v['n_list'],
                                                       r_2=v['r_2'],
                                                       side_information_dict=v['side_info'],
                                                       kernel_para_dict=v['kernel_para'],cuda=cuda)
            else:
                tmp_dict[str(i)] = TT_component(r_1=v['r_1'],n_list=v['n_list'],r_2=v['r_2'],cuda=cuda)

            lambdas.append(v['lambda'])
        self.lambdas = torch.nn.Parameter(torch.tensor(lambdas),requires_grad=False)
        self.TT_cores = torch.nn.ModuleDict(tmp_dict)
        self.TT_cores_prime = torch.nn.ModuleDict(tmp_dict_prime)

    def collect_core_outputs(self,indices):
        pred_outputs = []
        reg_output=0
        for i,v in self.ii.items():
            ix = indices[:,v]
            tt = self.TT_cores[str(i)]
            tt_prime = self.TT_cores_prime[str(i)]
            prime_pred = tt_prime(ix)
            reg_prime = tt_prime.get_aux_reg_term()
            if tt.__class__.__name__=='TT_kernel_component':
                pred, reg = tt(ix)
            else:
                pred = tt(ix)
                reg = tt.TT_core
            pred_outputs.append(pred*prime_pred)
            # reg_term = (reg_prime *tt.TT_core *reg)
            reg_output += (torch.dot(reg_prime.view(-1), torch.ones_like(reg_prime).view(-1)))*self.lambdas[i]
            # reg_output += (torch.sum(reg_prime))*self.lambdas[i]

        return pred_outputs,reg_output

    def forward(self,indices):
        preds_list,regularization = self.collect_core_outputs(indices)
        # regularization = regs.dot(self.lambdas)
        preds = preds_list[0]
        for i in range(1,len(preds_list)):
            preds = torch.bmm(preds,preds_list[i])
        return preds.squeeze(),regularization
if __name__ == '__main__':
    test = TT_component(r_1=1,n_list = [100,100],r_2 = 10)
