import torch
import tensorly
tensorly.set_backend('pytorch')
import gpytorch
from tensorly.base import fold,unfold,partial_fold
from tensorly.tenalg import multi_mode_dot,mode_dot
import math
PI  = math.pi
torch.set_printoptions(profile="full")
from gpytorch.lazy import LazyTensor,LazyEvaluatedKernelTensor

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
        self.n_feat = int(round(2*math.log(X.shape[0])))#Michaels paper!
        self.raw_lengthscale = torch.nn.Parameter(lengtscale,requires_grad=False)
        self.w = torch.randn(*(self.n_feat,self.n_input_feat),device=device)
        self.b = torch.rand(*(self.n_feat, 1),device=device)*2.0*PI
    def forward(self,X,dum_2=None):
        return torch.transpose(math.sqrt(2./self.n_feat)*torch.cos(torch.mm(self.w/self.raw_lengthscale, X.t()) + self.b),0,1)

class TT_component(torch.nn.Module):
    def __init__(self,r_1,n_list,r_2,cuda=None,config=None):
        super(TT_component, self).__init__()
        self.V_mode = True
        self.device = cuda
        self.full_grad = config['full_grad']
        self.n_dict = {i + 1: None for i in range(len(n_list))}
        self.RFF_dict = {i + 1: False for i in range(len(n_list))}
        self.shape_list  = [r_1]+[n for n in n_list] + [r_2]
        self.permutation_list = [i + 1 for i in range(len(n_list))] + [0, -1]
        self.reg_ones = {i + 1: self.lazy_ones(n,cuda) for i,n in enumerate(n_list)}
        self.TT_core = torch.nn.Parameter(torch.randn(*self.shape_list),requires_grad=True)

    def turn_off(self):
        for parameters in self.parameters():
            parameters.requires_grad = False
        self.V_mode=False

    def turn_on(self):
        for parameters in self.parameters():
            parameters.requires_grad = True

        self.V_mode=True

    def lazy_ones(self,n, cuda):
        if cuda is not None:
            o = torch.ones(*(n, 1), requires_grad=False).to(cuda)
        else:
            o = torch.ones(*(n, 1), requires_grad=False)
        return o

    def forward(self,indices):
        if self.full_grad:
            return self.TT_core,self.get_aux_reg_term()
        else:
            if len(indices.shape)>1:
                indices = indices.unbind(1)
            return self.TT_core.permute(self.permutation_list)[indices],self.get_aux_reg_term()

    def get_aux_reg_term(self):
        T = self.TT_core**2
        for mode,ones in self.reg_ones.items():
            T = lazy_mode_product(T, ones.t(), mode)
            T = lazy_mode_product(T, ones, mode)
        return T

class TT_kernel_component(TT_component): #for tensors with full or "mixed" side info
    def __init__(self,r_1,n_list,r_2,side_information_dict,kernel_para_dict,cuda=None,config=None):
        super(TT_kernel_component, self).__init__(r_1,n_list,r_2,cuda,config)
        for key,value in side_information_dict.items(): #Should be on the form {mode: side_info}'
            self.assign_kernel(key,value,kernel_para_dict)

    def kernel_train_mode_on(self):
        self.turn_off()
        for key,val in self.n_dict.items():
            if val is not None:
                k = getattr(self,f'kernel_{key}')
                k.raw_lengthscale.requires_grad = True
                if k.__class__.__name__=='PeriodicKernel':
                    k.raw_period_length.requires_grad = True

    def kernel_train_mode_off(self):
        self.turn_on()
        for key,val in self.n_dict.items():
            if val is not None:
                k = getattr(self,f'kernel_{key}')
                k.raw_lengthscale.requires_grad = False
                if k.__class__.__name__=='PeriodicKernel':
                    k.raw_period_length.requires_grad = False
                with torch.no_grad():
                    value = getattr(self,f'kernel_data_{key}')
                    self.n_dict[key] = k(value)

    def get_median_ls(self,X,key):  # Super LS and init value sensitive wtf
        base = gpytorch.kernels.Kernel()
        if X.shape[0] > 5000:
            self.RFF_dict[key] = True
            idx = torch.randperm(5000)
            X = X[idx, :]
        d = base.covar_dist(X, X)
        return torch.sqrt(torch.median(d[d > 0])).unsqueeze(0).to(self.device)

    def assign_kernel(self,key,value,kernel_dict_input):
        kernel_para_dict = kernel_dict_input[key]
        gwidth0 = self.get_median_ls(value,key)
        self.gamma_sq_init = gwidth0 * kernel_para_dict['ls_factor']
        ard_dims = None if not kernel_para_dict['ARD'] else value.shape[1]
        if self.RFF_dict[key]:
            setattr(self, f'kernel_{key}', RFF(value,lengtscale=self.gamma_sq_init,device=self.device))
            value = value.to(self.device)
        else:
            if kernel_para_dict['kernel_type']=='rbf':
                setattr(self, f'kernel_{key}', gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims))
            elif kernel_para_dict['kernel_type']=='matern':
                setattr(self, f'kernel_{key}', gpytorch.kernels.MaternKernel(ard_num_dims=ard_dims,nu=kernel_para_dict['nu']))
            elif kernel_para_dict['kernel_type']=='periodic':
                setattr(self, f'kernel_{key}', gpytorch.kernels.PeriodicKernel())
                getattr(self,f'kernel_{key}').raw_period_length = torch.nn.Parameter(torch.tensor(kernel_para_dict['p']),requires_grad=False)
            getattr(self, f'kernel_{key}').raw_lengthscale = torch.nn.Parameter(self.gamma_sq_init*torch.ones(*(1,1 if ard_dims is None else ard_dims),device=self.device), requires_grad=False)
        tmp_kernel_func = getattr(self,f'kernel_{key}')
        self.n_dict[key] =  tmp_kernel_func(value).to(self.device)
        self.register_buffer(f'kernel_data_{key}',value)

    def forward(self,indices):
        """Do tensor ops"""
        T = self.TT_core
        for key,val in self.n_dict.items():
            if val is not None:
                if not self.V_mode:
                    tmp_kernel_func = getattr(self, f'kernel_{key}')
                    val = tmp_kernel_func(getattr(self, f'kernel_data_{key}'))
                if not self.RFF_dict[key]:
                    T = lazy_mode_product(T, val, key)
                else:
                    T = lazy_mode_product(T, val.t(), key)
                    T = lazy_mode_product(T, val, key)
        if self.full_grad:
            return T,T*self.TT_core
        else:
            if len(indices.shape)>1:
                indices = indices.unbind(1)
            return T.permute(self.permutation_list)[indices], T*self.TT_core  #return both to calculate regularization when doing frequentist

class KFT(torch.nn.Module):
    def __init__(self,initializaiton_data,lambda_reg=1e-6,cuda=None,config=None): #decomposition_data = {0:{'ii':[0,1],'lambda':0.01,r_1:1 n_list=[10,10],r_2:10,'has_side_info':True, side_info:{1:x_1,2:x_2},kernel_para:{'ls_factor':0.5, 'kernel_type':'RBF','nu':2.5} },1:{}}
        super(KFT, self).__init__()
        tmp_dict = {}
        tmp_dict_prime = {}
        self.full_grad = config['full_grad']
        self.ii = {}
        for i,v in initializaiton_data.items():
            self.ii[i] = v['ii']
            tmp_dict_prime[str(i)] = TT_component(r_1=v['r_1'],n_list=v['n_list'],r_2=v['r_2'],cuda=cuda,config=config)
            if v['has_side_info']:
                tmp_dict[str(i)] = TT_kernel_component(r_1=v['r_1'],
                                                       n_list=v['n_list'],
                                                       r_2=v['r_2'],
                                                       side_information_dict=v['side_info'],
                                                       kernel_para_dict=v['kernel_para'],cuda=cuda,config=config)
            else:
                tmp_dict[str(i)] = TT_component(r_1=v['r_1'],n_list=v['n_list'],r_2=v['r_2'],cuda=cuda,config=config)
        self.register_buffer('lambda_reg',torch.tensor(lambda_reg).float())
        self.TT_cores = torch.nn.ModuleDict(tmp_dict)
        self.TT_cores_prime = torch.nn.ModuleDict(tmp_dict_prime)

    def turn_on_kernel_mode(self):
        for i,v in self.ii.items():
            self.TT_cores[str(i)].kernel_train_mode_on()
            self.TT_cores_prime[str(i)].turn_off()

    def turn_off_kernel_mode(self):
        for i,v in self.ii.items():
            self.TT_cores[str(i)].kernel_train_mode_off()
            self.TT_cores_prime[str(i)].turn_on()

    def collect_core_outputs(self,indices):
        pred_outputs = []
        reg_output=0
        for i,v in self.ii.items():
            ix = indices[:,v]
            tt = self.TT_cores[str(i)]
            tt_prime = self.TT_cores_prime[str(i)]
            prime_pred,reg_prime = tt_prime(ix)
            pred, reg = tt(ix)
            pred_outputs.append(pred*prime_pred)
            reg_output += torch.sum(reg*reg_prime)
        return pred_outputs,reg_output*self.lambda_reg

    def forward(self,indices):

        preds_list,regularization = self.collect_core_outputs(indices)
        preds = preds_list[0]

        for i in range(1,len(preds_list)):
            if self.full_grad:
                preds = edge_mode_product(preds,preds_list[i],len(preds.shape)-1,0) #General mode product!
            else:
                preds = torch.bmm(preds,preds_list[i])
        if self.full_grad:
            return preds.squeeze()[torch.unbind(indices,dim=1)],regularization
        else:
            return preds.squeeze(),regularization

class variational_TT_component(TT_component):
    def __init__(self,r_1,n_list,r_2,cuda=None):
        super(variational_TT_component, self).__init__(r_1,n_list,r_2,cuda)
        self.variance_parameters = torch.nn.Parameter(torch.zeros(*self.shape_list),requires_grad=True)

    def calculate_KL(self,mean,sig):
        KL = torch.mean(0.5*(sig.exp()+mean**2-sig-1))
        return KL

    def forward(self,indices):
        if len(indices.shape)>1:
            indices = indices.unbind(1)
        mean = self.TT_core.permute(self.permutation_list)[indices]
        sig = self.variance_parameters.permute(self.permutation_list)[indices]
        z = mean + torch.randn_like(mean)*sig.exp()
        return z, self.calculate_KL(mean,sig)

    def mean_forward(self,indices):
        if len(indices.shape)>1:
            indices = indices.unbind(1)
        return self.TT_core.permute(self.permutation_list)[indices]

class variational_kernel_TT(TT_kernel_component):
    def __init__(self, r_1, n_list, r_2, side_information_dict, kernel_para_dict, cuda=None):
        super(variational_kernel_TT, self).__init__(r_1, n_list, r_2, side_information_dict,
                                                               kernel_para_dict, cuda)
        self.noise_shape = [r_1]
        for key,val in  self.n_dict.items():
            self.set_variational_parameters(key,val)
        self.noise_shape.append(r_2)

    def kernel_train_mode_off(self):
        self.turn_on()
        for key, val in self.n_dict.items():
            if val is not None:
                k = getattr(self, f'kernel_{key}')
                k.raw_lengthscale.requires_grad = False
                if k.__class__.__name__ == 'PeriodicKernel':
                    k.raw_period_length.requires_grad = False
                with torch.no_grad():
                    value = getattr(self,f'kernel_data_{key}')
                    self.n_dict[key] = k(value)
        self.recalculate_priors()

    def recalculate_priors(self):
        for key, val in self.n_dict.items():
            if val is not None:
                if self.RFF_dict[key]:
                    mat = val
                    self.register_buffer(f'Phi_T_{key}',mat.t()@mat)
                    sig_p_2 = getattr(self, f'sig_p_2_{key}')
                    raw_cov = getattr(self,f'Phi_T_{key}')
                    self.register_buffer(f'Phi_T_trace_{key}',raw_cov.diag().sum())
                    eye = getattr(self,f'eye_{key}')
                    RFF_dim_const = getattr(self,f'RFF_dim_const_{key}')
                    if len(self.shape_list) > 3:
                        prior_log_det = -(gpytorch.logdet(raw_cov + eye * sig_p_2) * mat.shape[0] + torch.log(
                            sig_p_2) * (RFF_dim_const))
                    else:
                        prior_log_det = -gpytorch.logdet(raw_cov + eye * sig_p_2) + torch.log(sig_p_2) * (
                            RFF_dim_const)

                else:
                    mat = val.evaluate()
                    mat = mat + torch.eye(mat.shape[0],device=self.device)*1e-5
                    setattr(self, f'priors_inv_{key}', mat)
                    if len(self.shape_list) > 3:
                        prior_log_det = -gpytorch.logdet(mat) * mat.shape[0]
                    else:
                        prior_log_det = -gpytorch.logdet(mat)

                setattr(self, f'prior_log_det_{key}', prior_log_det)

    def set_variational_parameters(self,key,val):
        if val is None:
            self.RFF_dict[key] = True
        if self.RFF_dict[key]:
            if val is None:
                R = int(round(math.log(self.shape_list[key])))
                self.register_buffer(f'sig_p_2_{key}',torch.tensor(1,device=self.device))
                eye = torch.eye(R).to(self.device)
                self.register_buffer(f'eye_{key}',eye)
                self.register_buffer(f'r_const_{key}',torch.tensor(R).float())
                self.register_buffer(f'Phi_T_{key}',torch.ones_like(eye))
                self.register_buffer(f'Phi_T_trace_{key}',eye.sum())
                setattr(self,f'RFF_dim_const_{key}',0)
                prior_log_det = torch.tensor(0)
            else:
                mat  = val
                R = mat.shape[1]
                self.register_buffer(f'sig_p_2_{key}',torch.tensor(1e-5,device=self.device))
                eye = torch.eye(R).to(self.device)
                self.register_buffer(f'eye_{key}',eye)
                self.register_buffer(f'r_const_{key}',torch.tensor(R).float())
                self.register_buffer(f'Phi_T_{key}',mat.t()@mat)
                sig_p_2 = getattr(self, f'sig_p_2_{key}')
                raw_cov = getattr(self,f'Phi_T_{key}')
                self.register_buffer(f'Phi_T_trace_{key}',raw_cov.diag().sum())
                RFF_dim_const = mat.shape[0]-R
                setattr(self,f'RFF_dim_const_{key}',RFF_dim_const)
                if len(self.shape_list)>3:
                    prior_log_det = -(torch.log(torch.det(raw_cov+eye*sig_p_2).abs())*(mat.shape[0])+ torch.log(sig_p_2)*(RFF_dim_const))
                else:
                    prior_log_det = -(torch.log(torch.det(raw_cov+eye*sig_p_2).abs()) + torch.log(sig_p_2)*(RFF_dim_const))
            setattr(self, f'D_{key}', torch.nn.Parameter(1e-4 * torch.tensor([1.]), requires_grad=True))
        else:
            R = int(round(math.log(self.n_dict[key].shape[0])))
            mat  = val.evaluate()
            if len(self.shape_list) > 3:
                prior_log_det = -torch.log(torch.det(mat).abs()+1e-5)*(mat.shape[0])
            else:
                prior_log_det = -torch.log(torch.det(mat).abs()+1e-5)
            setattr(self, f'D_{key}', torch.nn.Parameter(1e-4 * torch.ones(mat.shape[0], 1), requires_grad=True))
            setattr(self, f'priors_inv_{key}', mat)
        self.noise_shape.append(R)
        setattr(self,f'prior_log_det_{key}',prior_log_det)
        self.register_buffer(f'n_const_{key}',torch.tensor(self.shape_list[key]).float())
        setattr(self,f'B_{key}',torch.nn.Parameter(torch.zeros(self.shape_list[key],R),requires_grad=True))

    def get_trace_term_KL(self,key):
        if self.RFF_dict[key]:
            D = getattr(self,f'D_{key}')**2
            B = getattr(self, f'B_{key}')
            sig_p_2 = getattr(self,f'sig_p_2_{key}')
            cov = B.t()@B
            B_times_B_sum = torch.sum(B*B)
            trace_term = torch.sum(cov*getattr(self,f'Phi_T_{key}')) + sig_p_2*B_times_B_sum + D*getattr(self,f'Phi_T_trace_{key}') + D*sig_p_2*getattr(self,f'n_const_{key}')
        else:
            cov,B_times_B_sum,D,B = self.build_cov(key)
            trace_term  = torch.sum(cov*getattr(self,f'priors_inv_{key}'))
        return trace_term.squeeze(),B_times_B_sum,D,cov,B

    def get_log_term(self,key,cov,D):
        n = getattr(self, f'n_const_{key}')
        if self.RFF_dict[key]:
            dim_const = getattr(self, f'RFF_dim_const_{key}')
            input = cov+getattr(self,f'eye_{key}')*D
            if len(self.shape_list) > 3:
                det = gpytorch.logdet(input)*n + dim_const * torch.log(D)
            else:
                det = gpytorch.logdet(input) + dim_const * torch.log(D)
        else:
            if len(self.shape_list) > 3:
                det = gpytorch.logdet(cov)*n
            else:
                det = gpytorch.logdet(cov)
        return det.squeeze()

    def calculate_KL(self):
        tr_term = 1.
        T = self.TT_core
        log_term_1 = 0
        log_term_2 = 0
        for key in self.n_dict.keys():
            trace,B_times_B_sum,D,cov,B = self.get_trace_term_KL(key)
            tr_term = tr_term*trace
            log_term_1 = log_term_1 + getattr(self,f'prior_log_det_{key}')
            log_term_2 = log_term_2 + self.get_log_term(key,cov,D)
            if self.RFF_dict[key]:
                T = lazy_mode_product(T,B.t(),key)
                T = lazy_mode_product(T,B,key)
                T = T + D*self.TT_core
            else:
                T = lazy_mode_product(T,cov,key)
        log_term = log_term_1 - log_term_2
        middle_term = torch.sum(T*self.TT_core)
        return tr_term + middle_term + log_term

    def build_cov(self,key):
        D = getattr(self,f'D_{key}')**2
        B = getattr(self,f'B_{key}')
        return B@B.t()+torch.diagflat(D),B*B,D,B

    def forward(self, indices):
        noise = torch.randn_like(self.TT_core)
        noise_2 = torch.randn(*self.noise_shape).to(self.device)
        for key, val in self.n_dict.items(): #Sample from multivariate
            noise = lazy_mode_hadamard(noise,getattr(self,f'D_{key}'), key)
            noise_2 = lazy_mode_product(noise_2,getattr(self,f'B_{key}'), key)
        T = self.TT_core  + noise_2 + noise

        for key, val in self.n_dict.items():
            if val is not None:
                if not self.V_mode:
                    tmp_kernel_func = getattr(self, f'kernel_{key}')
                    val = tmp_kernel_func(getattr(self, f'kernel_data_{key}'))
                    self.n_dict[key] = val
                    self.recalculate_priors()
                if not self.RFF_dict[key]:
                    T = lazy_mode_product(T, val, key)
                else:
                    T = lazy_mode_product(T, val.t(), key)
                    T = lazy_mode_product(T, val, key)
        if len(indices.shape) > 1:
            indices = indices.unbind(1)
        KL = self.calculate_KL()
        return T.permute(self.permutation_list)[indices], KL

    def mean_forward(self,indices):
        """Do tensor ops"""
        T = self.TT_core
        for key,val in self.n_dict.items():
            if val is not None:
                T = lazy_mode_product(T, val, key)
        if len(indices.shape)>1:
            indices = indices.unbind(1)
        return T.permute(self.permutation_list)[indices]  #return both to calcul

    def turn_on_kernel_mode(self):
        for i,v in self.ii.items():
            self.TT_cores[str(i)].kernel_train_mode_on()
            self.TT_cores_prime[str(i)].turn_off()

    def turn_off_kernel_mode(self):
        for i,v in self.ii.items():
            self.TT_cores[str(i)].kernel_train_mode_off()
            self.TT_cores_prime[str(i)].turn_on()

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
            tmp_dict_prime[str(i)] = variational_TT_component(r_1=v['r_1'], n_list=v['n_list'], r_2=v['r_2'], cuda=cuda)
            if v['has_side_info']:
                tmp_dict[str(i)] = variational_kernel_TT(r_1=v['r_1'],
                                                                    n_list=v['n_list'],
                                                                    r_2=v['r_2'],
                                                                    side_information_dict=v['side_info'],
                                                                    kernel_para_dict=v['kernel_para'], cuda=cuda)
            else:
                tmp_dict[str(i)] = variational_TT_component(r_1=v['r_1'], n_list=v['n_list'], r_2=v['r_2'], cuda=cuda)
            lambdas.append(v['lambda'])
        self.TT_cores = torch.nn.ModuleDict(tmp_dict)
        self.TT_cores_prime = torch.nn.ModuleDict(tmp_dict_prime)

    def collect_core_outputs_mean(self,indices):
        pred_outputs = []
        for i,v in self.ii.items():
            ix = indices[:,v]
            tt = self.TT_cores[str(i)]
            tt_prime = self.TT_cores_prime[str(i)]
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
            prime_pred,KL_prime = tt_prime(ix)
            pred, KL = tt(ix)

            pred_outputs.append(pred*prime_pred)
            total_KL += KL + KL_prime
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

    def turn_on_kernel_mode(self):
        for i,v in self.ii.items():
            self.TT_cores[str(i)].kernel_train_mode_on()
            self.TT_cores_prime[str(i)].turn_off()

    def turn_off_kernel_mode(self):
        for i,v in self.ii.items():
            self.TT_cores[str(i)].kernel_train_mode_off()
            self.TT_cores_prime[str(i)].turn_on()

if __name__ == '__main__':
    test = TT_component(r_1=1,n_list = [100,100],r_2 = 10)
