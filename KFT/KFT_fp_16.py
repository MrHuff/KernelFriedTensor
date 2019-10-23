import torch
import tensorly
tensorly.set_backend('pytorch')
import gpytorch
from tensorly.base import fold,unfold
import math
PI  = math.pi
torch.set_printoptions(profile="full")

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

class TT_component(torch.nn.Module):
    def __init__(self,r_1,n_list,r_2,cuda=None):
        super(TT_component, self).__init__()
        self.dummy_kernel = gpytorch.kernels.RBFKernel()
        for p in self.dummy_kernel.parameters():
            p.requires_grad = False
        self.n_dict = {i + 1: None for i in range(len(n_list))}
        self.shape_list  = [r_1]+[n for n in n_list] + [r_2]
        self.permutation_list = [i + 1 for i in range(len(n_list))] + [0, -1]
        self.reg_ones = {i + 1: self.lazy_ones(n,cuda) for i,n in enumerate(n_list)}
        self.TT_core = torch.nn.Parameter(torch.randn(*self.shape_list),requires_grad=True)

    def lazy_ones(self,n, cuda):
        if cuda is not None:
            test = torch.zeros(*(n, 1), requires_grad=False).to(cuda)
        else:
            test = torch.zeros(*(n, 1), requires_grad=False)
        ones = self.dummy_kernel(test,test)
        return ones

    def forward(self,indices): #For tensors with no side info #just use gather
        if len(indices.shape)>1:
            indices = indices.unbind(1)
        return self.TT_core.permute(self.permutation_list)[indices]

    def get_aux_reg_term(self):
        T = self.TT_core**2
        for mode,ones in self.reg_ones.items():
            T = lazy_mode_product(T, ones, mode)
        return T

class TT_kernel_component(TT_component): #for tensors with full or "mixed" side info
    def __init__(self,r_1,n_list,r_2,side_information_dict,kernel_para_dict,cuda=None):
        super(TT_kernel_component, self).__init__(r_1,n_list,r_2,cuda)
        self.keys = []
        for key,value in side_information_dict.items(): #Should be on the form {mode: side_info}'
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
        return torch.sqrt(torch.median(d[d > 0])).unsqueeze(0)

    def assign_kernel(self,key,value,kernel_para_dict):
        self.keys.append(key)
        gwidth0 = self.get_median_ls(value)
        self.gamma_sq_init = gwidth0 * kernel_para_dict['ls_factor']
        setattr(self, f'kernel_data_{key}', torch.nn.Parameter(value, requires_grad=False))
        if kernel_para_dict['kernel_type']=='rbf':
            setattr(self, f'kernel_{key}', gpytorch.kernels.RBFKernel())
        elif kernel_para_dict['kernel_type']=='matern':
            setattr(self, f'kernel_{key}', gpytorch.kernels.MaternKernel(nu=kernel_para_dict['nu']))
        elif kernel_para_dict['kernel_type']=='periodic':
            setattr(self, f'kernel_{key}', gpytorch.kernels.PeriodicKernel() )
            getattr(self,f'kernel_{key}').raw_period_length = torch.nn.Parameter(torch.tensor(kernel_para_dict['p']),requires_grad=False)
        getattr(self, f'kernel_{key}').raw_lengthscale = torch.nn.Parameter(self.gamma_sq_init, requires_grad=False)

    def forward(self,indices):
        """Do tensor ops"""
        T = self.TT_core
        for key,val in self.n_dict.items():
            if val is not None:
                T = lazy_mode_product(T, val, key)
        if len(indices.shape)>1:
            indices = indices.unbind(1)
        return T.permute(self.permutation_list)[indices], T*self.TT_core  #return both to calculate regularization when doing frequentist

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
                reg = tt.TT_core**2
            pred_outputs.append(pred*prime_pred)
            reg_output += torch.sum(reg*reg_prime)*self.lambdas[i]
        return pred_outputs,reg_output

    def forward(self,indices):
        preds_list,regularization = self.collect_core_outputs(indices)
        preds = preds_list[0]
        for i in range(1,len(preds_list)):
            preds = torch.bmm(preds,preds_list[i])
        return preds.squeeze(),regularization

class variational_TT_component(TT_component):
    def __init__(self,r_1,n_list,r_2,cuda=None):
        super(variational_TT_component, self).__init__(r_1,n_list,r_2,cuda)
        self.variance_parameters = torch.nn.Parameter(torch.randn(*self.shape_list),requires_grad=True)

    def calculate_KL(self,mean,sig):
        return 0.5*(sig.exp()+mean**2-sig-1)

    def forward(self,indices):
        if len(indices.shape)>1:
            indices = indices.unbind(1)
        mean = self.TT_core.permute(self.permutation_list)[indices]
        sig = self.variance_parameters.permute(self.permutation_list)[indices]
        z = mean + torch.randn_like(mean)*sig.exp()
        KL = self.calculate_KL(mean,sig)
        return z, KL

class variational_TT_kernel(TT_kernel_component):
    def __init__(self,r_1,n_list,r_2,side_information_dict,kernel_para_dict,bayes_dict,cuda=None):
        super(variational_TT_kernel, self).__init__(r_1,n_list,r_2,side_information_dict,kernel_para_dict,cuda)
        for key in self.keys:
            self.set_variational_parameters(key)

    def set_variational_parameters(self,key):
        prior_cov_mat =  torch.cholesky(self.n_dict[key].evaluate())
        prior_inv = torch.cholesky_inverse(prior_cov_mat)
        prior_log_det = torch.prod(torch.diag(prior_inv))**2
        setattr(self,f'priors_inv_{key}',prior_inv)
        setattr(self,f'prior_log_det_{key}',prior_log_det)
        setattr(self,f'sigma_{key}',torch.nn.Parameter(torch.randn_like(prior_inv),requires_grad=True))


# class variational_KFT(torch.nn.module):
#     def __init__(self,initializaiton_data_frequentist,initialization_data_bayeisan,cuda=None):
#         super(variational_KFT, self).__init__(initializaiton_data_frequentist,cuda)


if __name__ == '__main__':
    test = TT_component(r_1=1,n_list = [100,100],r_2 = 10)
