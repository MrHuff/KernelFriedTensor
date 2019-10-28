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

class RFF(torch.nn.Module):
    def __init__(self, X, lengtscale,rand_seed=1):
        super(RFF, self).__init__()
        torch.random.manual_seed(rand_seed)
        self.n_input_feat = X.shape[1] # dimension of the original input
        self.n_feat = int(round(math.sqrt(X.shape[0])*math.log(X.shape[0])))#Michaels paper!
        self.ls = lengtscale
        self.w = torch.randn(*(self.n_feat,self.n_input_feat))/self.ls
        self.b = torch.rand(*(self.n_feat, 1))*2.0*PI
    def forward(self,X,dum_2=None):
        return torch.transpose(math.sqrt(2./self.n_feat)*torch.cos(torch.mm(self.w, X.t()) + self.b),0,1)

class TT_component(torch.nn.Module):
    def __init__(self,r_1,n_list,r_2,cuda=None):
        super(TT_component, self).__init__()
        self.device = cuda
        self.dummy_kernel = gpytorch.kernels.RBFKernel()
        for p in self.dummy_kernel.parameters():
            p.requires_grad = False
        self.n_dict = {i + 1: None for i in range(len(n_list))}
        self.RFF_dict = {i + 1: True for i in range(len(n_list))}
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
        for key,value in side_information_dict.items(): #Should be on the form {mode: side_info}'
            self.assign_kernel(key,value,kernel_para_dict)

    def get_median_ls(self,X,key):  # Super LS and init value sensitive wtf
        base = gpytorch.kernels.Kernel()
        if X.shape[0] > 10000:
            self.RFF_dict[key] = True
            idx = torch.randperm(10000)
            X = X[idx, :]
        d = base.covar_dist(X, X)
        return torch.sqrt(torch.median(d[d > 0])).unsqueeze(0)

    def assign_kernel(self,key,value,kernel_para_dict):
        gwidth0 = self.get_median_ls(value,key)
        self.gamma_sq_init = gwidth0 * kernel_para_dict['ls_factor']
        if self.RFF_dict[key]:
            setattr(self, f'kernel_{key}', RFF(value,lengtscale=self.gamma_sq_init))
        else:
            if kernel_para_dict['kernel_type']=='rbf':
                setattr(self, f'kernel_{key}', gpytorch.kernels.RBFKernel())
            elif kernel_para_dict['kernel_type']=='matern':
                setattr(self, f'kernel_{key}', gpytorch.kernels.MaternKernel(nu=kernel_para_dict['nu']))
            elif kernel_para_dict['kernel_type']=='periodic':
                setattr(self, f'kernel_{key}', gpytorch.kernels.PeriodicKernel() )
                getattr(self,f'kernel_{key}').raw_period_length = torch.nn.Parameter(torch.tensor(kernel_para_dict['p']),requires_grad=False)
            getattr(self, f'kernel_{key}').raw_lengthscale = torch.nn.Parameter(self.gamma_sq_init, requires_grad=False)
        tmp_kernel_func = getattr(self,f'kernel_{key}')
        self.n_dict[key] =  tmp_kernel_func(value,value).to(self.device)
        if kernel_para_dict['deep']:
            setattr(self, f'kernel_data_{key}', torch.nn.Parameter(value, requires_grad=False))

    def forward(self,indices):
        """Do tensor ops"""
        T = self.TT_core
        for key,val in self.n_dict.items():
            if val is not None:
                if not self.RFF_dict[key]:
                    T = lazy_mode_product(T, val, key)
                else:
                    T = lazy_mode_product(T, val.t(), key)
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
        self.variance_parameters = torch.nn.Parameter(torch.zeros(*self.shape_list),requires_grad=True)

    def calculate_KL(self,mean,sig):
        return torch.mean(0.5*(sig.exp()+mean**2-sig-1))

    def forward(self,indices):
        if len(indices.shape)>1:
            indices = indices.unbind(1)
        mean = self.TT_core.permute(self.permutation_list)[indices]
        sig = self.variance_parameters.permute(self.permutation_list)[indices]
        z = mean + torch.randn_like(mean)*sig.exp()
        KL = self.calculate_KL(mean,sig)
        return z, KL

    def mean_forward(self,indices):
        if len(indices.shape)>1:
            indices = indices.unbind(1)
        return self.TT_core.permute(self.permutation_list)[indices]

class variational_kernel_TT(TT_kernel_component):
    def __init__(self, r_1, n_list, r_2, side_information_dict, kernel_para_dict, cuda=None):
        super(variational_kernel_TT, self).__init__(r_1, n_list, r_2, side_information_dict,
                                                               kernel_para_dict, cuda)
        self.noise_shape = [r_1]
        for key in self.keys:
            self.set_variational_parameters(key)
        self.noise_shape.append(r_2)

    def set_variational_parameters(self,key):
        if self.RFF_dict[key]:
            mat  = self.n_dict[key]
            R = mat.shape[1]
            setattr(self,f'sig_p_2_{key}',torch.nn.Parameter(torch.tensor(1e-5),requires_grad=False))
            sig_p_2 = torch.tensor(1e-5)
            eye = torch.eye(R)
            setattr(self, f'r_const_{key}',torch.nn.Parameter(torch.tensor(R).float(), requires_grad=False))
            setattr(self, f'Phi_T_{key}',mat.t()@mat)
            raw_cov = getattr(self,f'Phi_T_{key}')
            setattr(self,f'Phi_T_trace_{key}', torch.nn.Parameter(raw_cov.diag().sum(),requires_grad=False))
            RFF_dim_const = mat.shape[0]-R
            setattr(self,f'RFF_dim_const_{key}',RFF_dim_const)
            prior_log_det = -(torch.log(torch.det(raw_cov+eye*sig_p_2).abs())*(mat.shape[0])+ torch.log(sig_p_2)*(RFF_dim_const))
        else:
            R = int(round(math.log(self.n_dict[key].shape[0])))
            mat  = self.n_dict[key].evaluate()
            prior_log_det = -torch.log(torch.det(mat).abs()+1e-5)*(mat.shape[0])
        self.noise_shape.append(R)
        setattr(self,f'priors_inv_{key}',mat)
        setattr(self,f'prior_log_det_{key}',prior_log_det)
        setattr(self,f'n_const_{key}',torch.nn.Parameter(torch.tensor(self.shape_list[key]).float(),requires_grad=False))
        setattr(self,f'B_{key}',torch.nn.Parameter(torch.zeros(mat.shape[0],R),requires_grad=True))
        if self.RFF_dict[key]:
            setattr(self, f'D_{key}', torch.nn.Parameter(1e-4 * torch.tensor([1.]), requires_grad=True))
        else: #backup plan, do univariate meanfield...
            setattr(self, f'D_{key}', torch.nn.Parameter(1e-4 * torch.ones(mat.shape[0], 1), requires_grad=True))

    def get_trace_term_KL(self,key):
        if self.RFF_dict[key]:
            D = getattr(self,f'D_{key}')**2
            B = getattr(self, f'B_{key}')
            sig_p_2 = getattr(self,f'sig_p_2_{key}')
            cov = B.t()@B
            B_times_B_sum = torch.sum(B*B)
            trace_term = cov*getattr(self,f'Phi_T_{key}')+ sig_p_2*B_times_B_sum+D*getattr(self,f'Phi_T_trace_{key}')+D*sig_p_2*getattr(self,f'n_const_{key}')
        else:
            cov,B_times_B_sum,D = self.build_cov(key)
            trace_term  = cov*getattr(self,f'priors_inv_{key}')
        return trace_term,B_times_B_sum,D

    def get_log_term(self,key,B_times_B_sum,D):
        n = getattr(self, f'n_const_{key}')
        if self.RFF_dict[key]:
            dim_const = getattr(self,f'RFF_dim_const_{key}')
            inner_term = (n + n**2/(B_times_B_sum + D*n))*n + dim_const*torch.log(D)
        else:
            inner_term = (n + n**2 / (B_times_B_sum + D.sum()))*n
        return -inner_term

    def calculate_KL(self):
        tr_term = 1.
        T = self.TT_core
        log_term_1 = 0
        log_term_2 = 0
        for key in self.keys:
            pass

        log_term = log_term_1 - log_term_2
        middle_term = torch.sum(T*self.TT_core)
        return tr_term + middle_term + log_term

    def build_cov(self,key):
        D = getattr(self,f'D_{key}')**2
        B = getattr(self,f'B_{key}')
        return B@B.t()+torch.diagflat(D),B*B,D

    def forward(self, indices):
        noise = torch.randn_like(self.TT_core)
        noise_2 = torch.randn(*self.noise_shape).to(self.device)
        for key, val in self.n_dict.items(): #Sample from multivariate
            noise = lazy_mode_product(noise,torch.diagflat(getattr(self,f'D_{key}')), key)
            noise_2 = lazy_mode_product(noise_2,getattr(self,f'B_{key}'), key)
        # print(self.TT_core.norm(),noise.norm(),noise_2.norm())
        T = self.TT_core  + noise_2 + noise#Noisy gradients?! Double noise setup...
        for key, val in self.n_dict.items(): #Adding covariance nosie to the means seems to completely hinder optimization, might need to resort to analytical derivations?!
            if val is not None:
                if not self.RFF_dict[key]:
                    T = lazy_mode_product(T, val, key)
                else:
                    T = lazy_mode_product(T, val, key)
                    T = lazy_mode_product(T, val.t(), key)
        if len(indices.shape) > 1:
            indices = indices.unbind(1)

        return T.permute(self.permutation_list)[indices],  self.calculate_KL()

    def mean_forward(self,indices):
        """Do tensor ops"""
        T = self.TT_core
        for key,val in self.n_dict.items():
            if val is not None:
                T = lazy_mode_product(T, val, key)
        if len(indices.shape)>1:
            indices = indices.unbind(1)
        return T.permute(self.permutation_list)[indices]  #return both to calcul

class univariate_variational_kernel_TT(TT_kernel_component):
    def __init__(self,r_1,n_list,r_2,side_information_dict,kernel_para_dict,cuda=None):
        super(univariate_variational_kernel_TT, self).__init__(r_1, n_list, r_2, side_information_dict, kernel_para_dict, cuda)
        self.variance_parameters = torch.nn.Parameter(torch.zeros(*self.shape_list),requires_grad=True)

    def calculate_KL(self,mean,sig):
        return torch.mean(0.5*(sig.exp()+mean**2-sig-1))

    def forward(self,indices):
        noise = torch.randn_like(self.TT_core)*self.variance_parameters
        T = self.TT_core + noise  # Reparametrization, Prior fucked up change to isotropic/inverse ,i.e. (K^-1)
        for key, val in self.n_dict.items():
            if val is not None:
                T = lazy_mode_product(T, val, key)
        if len(indices.shape) > 1:
            indices = indices.unbind(1)
        mean = self.TT_core.permute(self.permutation_list)[indices]
        sig = self.variance_parameters.permute(self.permutation_list)[indices]
        KL = self.calculate_KL(mean,sig)
        return T.permute(self.permutation_list)[indices],KL #self.calculate_KL()

class variational_KFT(torch.nn.Module):
    def __init__(self,initializaiton_data_frequentist,KL_weight,cuda=None):
        super(variational_KFT, self).__init__()
        tmp_dict = {}
        tmp_dict_prime = {}
        lambdas = []
        self.ii = {}
        self.KL_weight = torch.nn.Parameter(torch.tensor(KL_weight),requires_grad=False)
        for i, v in initializaiton_data_frequentist.items():
            self.ii[i] = v['ii']
            tmp_dict_prime[str(i)] = TT_component(r_1=v['r_1'], n_list=v['n_list'], r_2=v['r_2'], cuda=cuda)
            if v['has_side_info']:
                tmp_dict[str(i)] = variational_kernel_TT(r_1=v['r_1'],
                                                                    n_list=v['n_list'],
                                                                    r_2=v['r_2'],
                                                                    side_information_dict=v['side_info'],
                                                                    kernel_para_dict=v['kernel_para'], cuda=cuda)
            else:
                tmp_dict[str(i)] = TT_component(r_1=v['r_1'], n_list=v['n_list'], r_2=v['r_2'], cuda=cuda)
            lambdas.append(v['lambda'])
        self.TT_cores = torch.nn.ModuleDict(tmp_dict)
        self.TT_cores_prime = torch.nn.ModuleDict(tmp_dict_prime)

    def collect_core_outputs_mean(self,indices):
        pred_outputs = []
        for i,v in self.ii.items():
            ix = indices[:,v]
            tt = self.TT_cores[str(i)]
            tt_prime = self.TT_cores_prime[str(i)]
            # prime_pred = tt_prime.mean_forward(ix)
            prime_pred = tt_prime(ix)
            pred = tt.mean_forward(ix)
            pred_outputs.append(pred*prime_pred)
        return pred_outputs

    def collect_core_outputs(self,indices):
        pred_outputs = []
        total_KL=0
        for i,v in self.ii.items():
            ix = indices[:,v]
            tt = self.TT_cores[str(i)]
            tt_prime = self.TT_cores_prime[str(i)]
            # prime_pred,KL_prime = tt_prime(ix)
            prime_pred = tt_prime(ix)
            pred, KL = tt(ix)
            pred_outputs.append(pred*prime_pred)
            total_KL += KL#+KL_prime
        return pred_outputs,total_KL*self.KL_weight

    def forward(self,indices):
        preds_list,regularization = self.collect_core_outputs(indices)
        preds = preds_list[0]
        for i in range(1,len(preds_list)):
            preds = torch.bmm(preds,preds_list[i])
        return preds.squeeze(),regularization

    def mean_forward(self,indices):
        preds_list = self.collect_core_outputs_mean(indices)
        preds = preds_list[0]
        for i in range(1,len(preds_list)):
            preds = torch.bmm(preds,preds_list[i])
        return preds.squeeze()

if __name__ == '__main__':
    test = TT_component(r_1=1,n_list = [100,100],r_2 = 10)
