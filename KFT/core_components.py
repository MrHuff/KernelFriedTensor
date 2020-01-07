import torch
import tensorly
tensorly.set_backend('pytorch')
import gpytorch
from tensorly.base import fold,unfold,partial_fold
from tensorly.tenalg import multi_mode_dot,mode_dot
from KFT.FLOWS.flows import IAF_no_h
import math
import timeit
import numpy as np
PI  = math.pi
torch.set_printoptions(profile="full")

def transpose_khatri_rao(x, y):
    """
    :param x: n x c
    :param y: n x d
    :return: n x (cd)
    """
    y = y.unsqueeze(-1).expand(-1,-1,x.shape[1]).permute(0,2,1)
    return torch.einsum('ij,ijk->ijk', x, y).flatten(1)

def edge_mode_product(T,also_T,mode_T, mode_also_T):
    """
    :param T: Tensor - core [n_1,n_2.n_3]
    :param also_T: Tensor - to be applied [n_3,m_2,m_3]
    :param mode: to which mode of T
    :return: mode = 2 -> [n_1,n_2,m_2,m_3]
    """
    final_shape = list(T.shape[0:(mode_T)]) + list(also_T.shape[mode_also_T+1:])
    also_T = unfold(also_T,mode_also_T)
    T = lazy_mode_product(T,also_T.t(),mode_T)
    return T.view(final_shape)

def lazy_mode_product(T, K, mode):
    """
    :param T: Pytorch tensor, just pass as usual
    :param K: Gpytorch lazy tensor, remember should be on form MxN
    :param mode: Mode of tensor
    :return:
    """
    new_shape = list(T.shape)
    new_shape[mode] = K.shape[0]
    T = unfold(T,mode)
    T = K@T
    T = fold(T,mode,new_shape)
    return T

def lazy_mode_hadamard(T,vec,mode):
    """
    :param T: Pytorch tensor, just pass as usual
    :param vec: A vector to be applied hadamard style, is equivalent of diagonalizing vec and doing matrix product.
    :param mode: Mode of tensor
    :return:
    """
    new_shape = list(T.shape)
    T = unfold(T,mode)
    T = vec*T
    T = fold(T,mode,new_shape)
    return T

class RFF(torch.nn.Module):
    def __init__(self, X, lengtscale,rand_seed=1,device=None):
        super(RFF, self).__init__()
        torch.random.manual_seed(rand_seed)
        self.n_input_feat = X.shape[1] # dimension of the original input
        self.n_feat = int(round((X.shape[0]**0.25)*math.log(X.shape[0])))#Michaels paper!
        print(f'I= {self.n_feat}')
        self.raw_lengthscale = torch.nn.Parameter(lengtscale,requires_grad=False)
        self.register_buffer('w' ,torch.randn(*(self.n_feat,self.n_input_feat)))
        self.register_buffer('b' ,torch.rand(*(self.n_feat, 1),device=device)*2.0*PI)

    def forward(self,X,dum_2=None):
        return torch.transpose(math.sqrt(2./float(self.n_feat))*torch.cos(torch.mm(self.w/self.raw_lengthscale, X.t()) + self.b),0,1)

class TT_component(torch.nn.Module):
    def __init__(self,r_1,n_list,r_2,cuda=None,config=None,init_scale=1.0,old_setup=False,reg_para=0,prime=False,sub_R=2):
        super(TT_component, self).__init__()
        self.register_buffer('reg_para',torch.tensor(reg_para))
        self.prime = prime
        self.r_1 = r_1
        self.r_2 = r_2
        self.n_list = n_list
        self.amp = None
        self.device = cuda
        self.old_setup = old_setup
        self.full_grad = config['full_grad']
        self.dual = config['dual']
        self.n_dict = {i + 1: None for i in range(len(n_list))}
        self.RFF_dict = {i + 1: False for i in range(len(n_list))}
        self.shape_list  = [r_1]+[n for n in n_list] + [r_2]
        self.permutation_list = [i + 1 for i in range(len(n_list))] + [0, -1]
        if self.prime:
            self.core_param = sub_factorization(self.shape_list,R=sub_R,init_scale=init_scale)
        else:
            self.core_param = torch.nn.Parameter(init_scale*torch.ones(*self.shape_list), requires_grad=True)

        self.init_scale = init_scale
        for i, n in enumerate(n_list):
            self.register_buffer(f'reg_ones_{i}',torch.ones((n,1)))

    def index_select(self,indices,T):
        return T.permute(self.permutation_list)[indices]

    def turn_off(self):
        for p in self.parameters():
            p.requires_grad = False

    def turn_on(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self,indices):
        if self.prime:
            p = self.core_param()
        else:
            p = self.core_param
        if self.dual and not self.old_setup:
            reg = self.get_aux_reg_term()
        else:
            reg = p**2
        if self.full_grad:
            return p, reg
        else:
            if len(indices.shape)>1:
                indices = indices.unbind(1)
            return p.permute(self.permutation_list)[indices], reg

    def forward_scale(self,indices):
        if self.full_grad:
            return self.core_param, self.core_param**2
        else:
            if len(indices.shape)>1:
                indices = indices.unbind(1)
            return self.core_param.permute(self.permutation_list)[indices], self.core_param**2

    def get_aux_reg_term(self):
        if self.old_setup:
            return 1.
        else:
            if self.prime:
                p = self.core_param()
            else:
                p = self.core_param
            T = p ** 2
            for mode,ones in enumerate(self.n_list):
                ones = getattr(self,f'reg_ones_{mode}')
                T = lazy_mode_product(T, ones.t(), mode+1)
                T = lazy_mode_product(T, ones, mode+1)
            return T

    def toggle_mean_var(self,toggle):
        self.core_param.requires_grad = toggle
        self.variance_parameters.requires_grad = not toggle

class TT_component_deep(TT_component):
    def __init__(self,r_1,n_list,r_2,cuda=None,config=None,init_scale=1.0,old_setup=False,reg_para=0):
        super(TT_component_deep, self).__init__(r_1,n_list,r_2,cuda,config,init_scale,old_setup,reg_para)
        self.L = config['L']
        self.non_lin = config['non_lin']
        self.init_index_logic()

    def init_index_logic(self):
        l = [self.r_1,self.r_2]
        i = np.argmax(l)
        r  = l[i]
        self.apply_index = 1 if i==0 else 2
        if self.r_1==self.r_2:
            self.apply_index = 2
        for i in range(self.L):
            setattr(self,f'deep_layer_{i}',torch.nn.Parameter(torch.eye(int(r)),requires_grad=True))

    def turn_off(self):
        for n,p in self.named_parameters():
            if 'deep_layer' not in n:
                p.requires_grad = False
        self.V_mode = False

    def nn_reg(self):
        p = 0
        for i in range(self.L):
            p+=torch.mean(getattr(self,f'deep_layer_{i}')**2)
        return p

    def nn_forward(self,X):
        for i in range(self.L-1):
            p = getattr(self,f'deep_layer_{i}')
            X = self.non_lin(lazy_mode_product(X,p,self.apply_index))
        p = getattr(self,f'deep_layer_{self.L-1}')
        X = lazy_mode_product(X, p, self.apply_index) #output layer lol
        return X

class TT_kernel_component(TT_component): #for tensors with full or "mixed" side info
    def __init__(self,r_1,n_list,r_2,side_information_dict,kernel_para_dict,cuda=None,config=None,init_scale=1.0,reg_para=0,old_setup=False):
        super(TT_kernel_component, self).__init__(r_1,n_list,r_2,cuda,config,init_scale,old_setup,reg_para)
        self.core_param = torch.nn.Parameter(init_scale*torch.randn(*self.shape_list), requires_grad=True)
        self.deep_kernel = config['deep_kernel']
        self.deep_mode = False
        self.kernel_eval_mode = False
        for key,value in side_information_dict.items(): #Should be on the form {mode: side_info}'
            if self.dual:
                self.assign_kernel(key,value,kernel_para_dict,config['deep_kernel'])
            else:
                self.n_dict[key] = value.to(self.device)

    def kernel_train_mode_on(self):
        self.turn_off()
        self.kernel_eval_mode = True
        if self.dual:
            for key,val in self.n_dict.items():
                if val is not None:
                    k = getattr(self,f'kernel_{key}')
                    k.raw_lengthscale.requires_grad = True
                    if k.__class__.__name__=='PeriodicKernel':
                        k.raw_period_length.requires_grad = True
            return 0

    def kernel_train_mode_off(self):
        self.turn_on()
        self.kernel_eval_mode = False
        if self.dual:
            for key,val in self.n_dict.items():
                if val is not None:
                    k = getattr(self,f'kernel_{key}')
                    k.raw_lengthscale.requires_grad = False
                    if k.__class__.__name__=='PeriodicKernel':
                        k.raw_period_length.requires_grad = False
                    with torch.no_grad():
                        value = getattr(self,f'kernel_data_{key}')
                        if  k.__class__.__name__=='RFF':
                            self.n_dict[key] = k(value)
                        else:
                            self.n_dict[key] = k(value).evaluate()
        return 0

    def deep_kernel_mode_on(self):
        self.deep_mode = True
        for key in self.n_dict.keys():
            f = getattr(self, f'transformation_{key}')
            for p in f.parameters():
                p.requires_grad = True
        return 0

    def deep_kernel_mode_off(self):
        self.deep_mode = False
        for key in self.n_dict.keys():
            f = getattr(self, f'transformation_{key}')
            for p in f.parameters():
                p.requires_grad = False
            for key,val in self.n_dict.items(): #Issue is probably here!
                k = getattr(self,f'kernel_{key}')
                f = getattr(self, f'transformation_{key}')
                input = getattr(self,f'kernel_data_{key}')
                with torch.no_grad():
                    X = f(input)
                    if  k.__class__.__name__=='RFF':
                        self.n_dict[key] = k(X)
                    else:
                        self.n_dict[key] = k(X).evaluate()
        return 0

    def get_median_ls(self,X,key):  # Super LS and init value sensitive wtf
        base = gpytorch.kernels.Kernel()
        if X.shape[0] > 5000:
            self.RFF_dict[key] = True
            idx = torch.randperm(5000)
            X = X[idx, :]
        d = base.covar_dist(X, X)
        return torch.sqrt(torch.median(d[d > 0])).unsqueeze(0)

    def assign_kernel(self,key,value,kernel_dict_input,deep_kernel=False):
        kernel_para_dict = kernel_dict_input[key]
        gwidth0 = self.get_median_ls(value,key)
        self.gamma_sq_init = gwidth0 * kernel_para_dict['ls_factor']
        ard_dims = None if not kernel_para_dict['ARD'] else value.shape[1]
        if self.RFF_dict[key]:
            setattr(self, f'kernel_{key}', RFF(value,lengtscale=self.gamma_sq_init))
            # value = value.to(self.device)
        else:
            if kernel_para_dict['kernel_type']=='rbf':
                setattr(self, f'kernel_{key}', gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims))
                getattr(self, f'kernel_{key}').raw_lengthscale = torch.nn.Parameter(
                    self.gamma_sq_init * torch.ones(*(1, 1 if ard_dims is None else ard_dims)),
                    requires_grad=False)

            elif kernel_para_dict['kernel_type']=='matern':
                setattr(self, f'kernel_{key}', gpytorch.kernels.MaternKernel(ard_num_dims=ard_dims,nu=kernel_para_dict['nu']))
                getattr(self, f'kernel_{key}').raw_lengthscale = torch.nn.Parameter(
                    self.gamma_sq_init * torch.ones(*(1, 1 if ard_dims is None else ard_dims)),
                    requires_grad=False)
            elif kernel_para_dict['kernel_type']=='periodic':
                setattr(self, f'kernel_{key}', gpytorch.kernels.PeriodicKernel())
                getattr(self, f'kernel_{key}').raw_lengthscale = torch.nn.Parameter(
                    self.gamma_sq_init * torch.ones(*(1, 1)),
                    requires_grad=False)
                getattr(self,f'kernel_{key}').raw_period_length = torch.nn.Parameter(kernel_para_dict['p']*torch.ones(*(1,1)),requires_grad=False)

        tmp_kernel_func = getattr(self,f'kernel_{key}')
        if tmp_kernel_func.__class__.__name__ in 'RFF':
            self.n_dict[key] =  tmp_kernel_func(value).to(self.device)
        else:
            self.n_dict[key] =  tmp_kernel_func(value).evaluate().to(self.device)
        self.register_buffer(f'kernel_data_{key}',value)
        if deep_kernel:
            setattr(self,f'transformation_{key}',IAF_no_h(latent_size=value.shape[1],depth=2,tanh_flag_h=True,C=10))

    def side_data_eval(self,key):
        X = getattr(self, f'kernel_data_{key}')
        tmp_kernel_func = getattr(self, f'kernel_{key}')
        if self.deep_mode:
            f = getattr(self, f'transformation_{key}')
            X = f(X)
        if tmp_kernel_func.__class__.__name__=='RFF':
            val = tmp_kernel_func(X)
        else:
            val = tmp_kernel_func(X).evaluate()
        return val
    def apply_kernels(self,T):
        for key, val in self.n_dict.items():
            if val is not None:
                if self.dual:
                    if self.kernel_eval_mode:
                        val = self.side_data_eval(key)
                    if not self.RFF_dict[key]:
                        T = lazy_mode_product(T, val, key)
                    else:
                        T = lazy_mode_product(T, val.t(), key)
                        T = lazy_mode_product(T, val, key)
                else:
                    T = lazy_mode_product(T, val, key)
        return T

    def forward(self,indices):
        """Do tensor ops"""
        T = self.apply_kernels(self.core_param)
        if self.dual:
            reg = T*self.core_param
        else:
            reg = self.core_param**2
        if self.full_grad:
            return T,reg
        else:
            if len(indices.shape)>1:
                indices = indices.unbind(1)
            return T.permute(self.permutation_list)[indices], reg  #return both to calculate regularization when doing frequentist

class sub_factorization(torch.nn.Module):
    def __init__(self,tensor_shape,R=2,init_scale=1):
        super(sub_factorization, self).__init__()
        self.tensor_shape = tensor_shape
        self.factor  = R**(len(tensor_shape)-1)
        self.init_scale = init_scale
        for i,n in enumerate(tensor_shape):
            if i==0:
                setattr(self,f'latent_component_{i}',torch.nn.Parameter(torch.ones(*(1,n,R) ),requires_grad=True))
            elif i==len(tensor_shape)-1:
                setattr(self,f'latent_component_{i}',torch.nn.Parameter(torch.ones(*(R,n,1) ),requires_grad=True))
            else:
                setattr(self,f'latent_component_{i}',torch.nn.Parameter(torch.ones(*(R,n,R) ),requires_grad=True))
    def forward(self):
        preds = getattr(self,f'latent_component_0')
        for i in range(1,len(self.tensor_shape)):
            m = getattr(self,f'latent_component_{i}')
            preds = edge_mode_product(preds, m, len(preds.shape) - 1, 0)  # General mode product!
        # print(preds.squeeze()/self.factor)
        return self.init_scale *preds.squeeze(0).squeeze(-1)/self.factor

