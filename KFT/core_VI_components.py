import torch
import tensorly
tensorly.set_backend('pytorch')
import gpytorch
import math
import timeit
from KFT.core_components import TT_component,TT_kernel_component,lazy_mode_product,lazy_mode_hadamard,row_outer_prod,edge_mode_product
PI  = math.pi
torch.set_printoptions(profile="full")

class variational_TT_component(TT_component):
    def __init__(self,r_1,n_list,r_2,cuda=None,config=None,init_scale=1.0,old_setup=False,prime=False,sub_R=1,mu_prior=0,sigma_prior=-1):
        super(variational_TT_component, self).__init__(r_1,n_list,r_2,cuda,config,init_scale,old_setup,prime=prime,sub_R=sub_R)
        self.variance_parameters = torch.nn.Parameter(-2*torch.ones(*self.shape_list),requires_grad=True)
        self.register_buffer('mu_prior',torch.tensor(mu_prior))
        self.register_buffer('sigma_prior',torch.tensor(sigma_prior))

    def calculate_KL(self,mean,sig):
        KL = 0.5*((mean-self.mu_prior)**2 + sig.exp()/self.sigma_prior.exp()-1-(sig-self.sigma_prior)).sum(dim=1).mean().squeeze()
        return KL

    def sample(self,indices):
        if self.full_grad:
            mean = self.core_param
            sig = self.variance_parameters
            T = mean + (0.5*sig).exp()*torch.randn_like(mean)
            return T
        else:
            if len(indices.shape)>1:
                indices = indices.unbind(1)
            mean = self.core_param.permute(self.permutation_list)[indices]
            sig = self.variance_parameters.permute(self.permutation_list)[indices]
            z = mean + torch.randn_like(mean)*(0.5*sig).exp()
            return z

    def forward_reparametrization(self,indices):
        if self.full_grad:
            mean = self.core_param
            sig = self.variance_parameters
            T = mean + (0.5*sig).exp()*torch.randn_like(mean)
            return T,  self.calculate_KL(mean,sig)
        else:
            if len(indices.shape)>1:
                indices = indices.unbind(1)
            mean = self.core_param.permute(self.permutation_list)[indices]
            sig = self.variance_parameters.permute(self.permutation_list)[indices]
            z = mean + torch.randn_like(mean)*(sig*0.5).exp()
            return z, self.calculate_KL(mean,sig)

    def mean_forward(self,indices):
        if self.full_grad:
            return self.core_param
        else:
            if len(indices.shape)>1:
                indices = indices.unbind(1)
            return self.core_param.permute(self.permutation_list)[indices]

    def forward(self,indices):
        if self.full_grad:
            KL = self.calculate_KL(self.core_param,self.variance_parameters)
            mean = self.core_param
            sig = self.variance_parameters
        else:
            if len(indices.shape) > 1:
                indices = indices.unbind(1)
            mean = self.core_param.permute(self.permutation_list)[indices]
            sig = self.variance_parameters.permute(self.permutation_list)[indices]
            KL = self.calculate_KL(mean,sig)
        return mean,sig.exp(),KL

class univariate_variational_kernel_TT(TT_kernel_component):
    def __init__(self, r_1, n_list, r_2, side_information_dict, kernel_para_dict, cuda=None,config=None,init_scale=1.0,mu_prior=0,sigma_prior=-1):
        super(univariate_variational_kernel_TT, self).__init__(r_1, n_list, r_2, side_information_dict,
                                                                 kernel_para_dict, cuda, config, init_scale)
        self.variance_parameters = torch.nn.Parameter(-2*torch.ones(*self.shape_list),requires_grad=True)
        self.register_buffer('mu_prior', torch.tensor(mu_prior))
        self.register_buffer('sigma_prior', torch.tensor(sigma_prior))

    def calculate_KL(self,mean,sig):
        KL = 0.5*((mean-self.mu_prior)**2 + sig.exp()/self.sigma_prior.exp()-1-(sig-self.sigma_prior)).sum(dim=1).mean().squeeze()
        return KL

    def forward_reparametrization(self, indices):
        """Do tensor ops"""
        mean = self.core_param
        sig = self.variance_parameters
        T = mean + (0.5*self.variance_parameters).exp()*torch.randn_like(mean)
        T = self.apply_kernels(T)
        if self.full_grad:
            return T, self.calculate_KL(mean,sig)
        else:
            if len(indices.shape) > 1:
                indices = indices.unbind(1)
            return T.permute(self.permutation_list)[
                       indices],self.calculate_KL(mean,sig)  # return both to calculate regularization when doing freque

    def apply_square_kernel(self,T):
        for key, val in self.n_dict.items():
            if val is not None:
                if self.dual:
                    if self.kernel_eval_mode:
                        X = getattr(self, f'kernel_data_{key}')
                        if self.deep_mode:
                            f = getattr(self, f'transformation_{key}')
                            X = f(X)
                        tmp_kernel_func = getattr(self, f'kernel_{key}')
                        if tmp_kernel_func.__class__.__name__=='RFF':
                            val = tmp_kernel_func(X)
                        else:
                            val = tmp_kernel_func(X).evaluate()
                    if not self.RFF_dict[key]:
                        T = lazy_mode_product(T, val*val, key)
                    else:
                        RFF_squared = row_outer_prod(val,val)
                        T = lazy_mode_product(T,RFF_squared.t(), key)
                        T = lazy_mode_product(T, RFF_squared, key)
                else:
                    T = lazy_mode_product(T, val**2, key)
        return T

    def sample(self,indices):
        mean = self.core_param
        T = mean + (0.5 * self.variance_parameters).exp() * torch.randn_like(mean)
        T = self.apply_kernels(T)
        if self.full_grad:
            return T
        else:
            if len(indices.shape) > 1:
                indices = indices.unbind(1)
            return T.permute(self.permutation_list)[
                       indices]

    def forward(self,indices):
        T = self.apply_kernels(self.core_param)
        T_additional = self.apply_square_kernel(self.variance_parameters.exp())
        if self.full_grad or not self.dual:
            KL = self.calculate_KL(self.core_param, self.variance_parameters)
        else:
            mean = self.core_param.permute(self.permutation_list)[indices]
            sig = self.variance_parameters.permute(self.permutation_list)[indices]
            KL = self.calculate_KL(mean, sig)
        if self.full_grad:
            return T,T_additional, KL
        else:
            if len(indices.shape) > 1:
                indices = indices.unbind(1)
            return T.permute(self.permutation_list)[indices],T_additional.permute(self.permutation_list)[indices], KL

    def mean_forward(self,indices):
        """Do tensor ops"""
        T = self.core_param
        for key, val in self.n_dict.items():
            if val is not None:
                if self.kernel_eval_mode:
                    tmp_kernel_func = getattr(self, f'kernel_{key}')
                    val = tmp_kernel_func(getattr(self, f'kernel_data_{key}'))
                if not self.RFF_dict[key]:
                    T = lazy_mode_product(T, val, key)
                else:
                    T = lazy_mode_product(T, val.t(), key)
                    T = lazy_mode_product(T, val, key)
        if self.full_grad:
            return T
        else:
            if len(indices.shape) > 1:
                indices = indices.unbind(1)
            return T.permute(self.permutation_list)[indices]

class multivariate_variational_kernel_TT(TT_kernel_component):
    def __init__(self, r_1, n_list, r_2, side_information_dict, kernel_para_dict, cuda=None,config=None,init_scale=1.0, mu_prior=1):
        super(multivariate_variational_kernel_TT, self).__init__(r_1, n_list, r_2, side_information_dict,
                                                                 kernel_para_dict, cuda, config, init_scale)
        self.register_buffer('mu_prior', torch.tensor(mu_prior))
        self.register_buffer('ones',torch.ones_like(self.core_param))
        self.noise_shape = [r_1]
        for key,val in  self.n_dict.items():
            self.set_variational_parameters(key,val)
        self.noise_shape.append(r_2)

    def kernel_train_mode_off(self):
        self.turn_on()
        self.kernel_eval_mode = False
        for key, val in self.n_dict.items():
            if val is not None:
                k = getattr(self, f'kernel_{key}')
                k.raw_lengthscale.requires_grad = False
                if k.__class__.__name__ == 'PeriodicKernel':
                    k.raw_period_length.requires_grad = False
                with torch.no_grad():
                    value = getattr(self,f'kernel_data_{key}')
                    if k.__class__.__name__=='RFF':
                        self.n_dict[key] = k(value)
                    else:
                        self.n_dict[key] = k(value).evaluate()
        self.recalculate_priors()

    def recalculate_priors(self):
        for key, val in self.n_dict.items():
            if val is not None:
                mat = val
                if self.RFF_dict[key]:
                    self.register_buffer(f'Phi_T_{key}',mat.t()@mat)
                    sig_p_2 = getattr(self, f'sig_p_2_{key}')
                    raw_cov = getattr(self,f'Phi_T_{key}')
                    self.register_buffer(f'Phi_T_trace_{key}',raw_cov.diag().mean())
                    eye = getattr(self,f'eye_{key}')
                    RFF_dim_const = getattr(self,f'RFF_dim_const_{key}')
                    if len(self.shape_list) > 3:
                        prior_log_det = -(gpytorch.logdet(raw_cov.float() + (eye * sig_p_2).float()) * mat.shape[0] + torch.log(
                            sig_p_2) * (RFF_dim_const))
                    else:
                        prior_log_det = -gpytorch.logdet(raw_cov.float() + (eye * sig_p_2).float()) + torch.log(sig_p_2) * (
                            RFF_dim_const)
                else:
                    mat = val + getattr(self,f'reg_diag_cholesky_{key}')
                    self.register_buffer( f'priors_inv_{key}', mat)
                    if len(self.shape_list) > 3:
                        prior_log_det = -gpytorch.log_det(mat) * mat.shape[0]
                    else:
                        prior_log_det = -gpytorch.log_det(mat)
                self.register_buffer(f'prior_log_det_{key}', prior_log_det)

    def set_variational_parameters(self,key,val):
        if val is None:
            self.RFF_dict[key] = True
        if self.RFF_dict[key]:
            if val is None:
                R = int(round(math.log(self.shape_list[key])))
                self.register_buffer(f'sig_p_2_{key}',torch.tensor(1.,device=self.device))
                eye = torch.eye(R).to(self.device)
                self.register_buffer(f'eye_{key}',eye)
                self.register_buffer(f'r_const_{key}',torch.tensor(R).float())
                self.register_buffer(f'Phi_T_{key}',torch.ones_like(eye))
                self.register_buffer(f'Phi_T_trace_{key}',eye.mean())
                self.register_buffer(f'RFF_dim_const_{key}',torch.tensor(0))
                prior_log_det = torch.tensor(0)
            else:
                mat  = val
                R = mat.shape[1]
                self.register_buffer(f'sig_p_2_{key}',torch.tensor(1e-2))
                eye = torch.eye(R,device=self.device)
                self.register_buffer(f'eye_{key}',eye)
                self.register_buffer(f'r_const_{key}',torch.tensor(R).float())
                self.register_buffer(f'Phi_T_{key}',mat.t()@mat)
                sig_p_2 = getattr(self, f'sig_p_2_{key}')
                raw_cov = getattr(self,f'Phi_T_{key}')
                self.register_buffer(f'Phi_T_trace_{key}',raw_cov.diag().mean())
                self.register_buffer(f'Phi_T_trace_{key}',raw_cov.diag().mean())
                RFF_dim_const = mat.shape[0]-R
                self.register_buffer(f'RFF_dim_const_{key}',torch.tensor(RFF_dim_const))
                if len(self.shape_list)>3:
                    prior_log_det = -(gpytorch.log_det(raw_cov+eye*sig_p_2))*(mat.shape[0])+ torch.log(sig_p_2)*(RFF_dim_const)
                else:
                    prior_log_det = -(gpytorch.log_det(raw_cov+eye*sig_p_2)) + torch.log(sig_p_2)*(RFF_dim_const)
            setattr(self, f'D_{key}', torch.nn.Parameter(self.init_scale * torch.tensor([1.]), requires_grad=True))
            setattr(self, f'B_{key}', torch.nn.Parameter(1e-5*torch.randn(self.shape_list[key], R), requires_grad=True))
        else:
            R = int(round(20.*math.log(self.n_dict[key].shape[0])))
            self.register_buffer(f'reg_diag_cholesky_{key}',torch.eye(val.shape[0],device=self.device)*1e-3)
            mat  = val + getattr(self,f'reg_diag_cholesky_{key}')
            if len(self.shape_list) > 3:
                prior_log_det = -gpytorch.log_det(mat)*(mat.shape[0])
            else:
                prior_log_det = -gpytorch.log_det(mat)
            setattr(self, f'D_{key}', torch.nn.Parameter(1e-3*torch.ones(mat.shape[0], 1), requires_grad=True))
            self.register_buffer(f'priors_inv_{key}', mat)
            setattr(self, f'B_{key}', torch.nn.Parameter(1e-5*torch.randn(self.shape_list[key], R), requires_grad=True))

        self.noise_shape.append(R)
        self.register_buffer(f'prior_log_det_{key}',prior_log_det)
        self.register_buffer(f'n_const_{key}',torch.tensor(self.shape_list[key]).float())

    def fast_log_det(self,L):
        return torch.log(torch.prod(L.diag())**2+1e-5)

    def get_trace_term_KL(self,key):
        if self.RFF_dict[key]:
            D = getattr(self,f'D_{key}')**2
            B = getattr(self, f'B_{key}')
            sig_p_2 = getattr(self,f'sig_p_2_{key}')
            cov = B.t()@B
            B_times_B_sum = torch.sum(B*B)
            trace_term = torch.sum(cov*getattr(self,f'Phi_T_{key}')) + sig_p_2*B_times_B_sum + D*getattr(self,f'Phi_T_trace_{key}') + D*sig_p_2*getattr(self,f'n_const_{key}')
        else:
            cov,D,B = self.build_cov(key)
            trace_term  = torch.sum(cov*getattr(self,f'priors_inv_{key}'))
        return trace_term.squeeze(),D,cov,B

    def get_log_term(self,key,cov,D,B):
        n = getattr(self, f'n_const_{key}')
        if self.RFF_dict[key]:
            dim_const = getattr(self, f'RFF_dim_const_{key}')
            input = cov+getattr(self,f'eye_{key}')*D
            if len(self.shape_list) > 3:
                det = torch.logdet(input)*n + dim_const * torch.log(D)
            else:
                det = torch.logdet(input) + dim_const * torch.log(D)
        else:
            if len(self.shape_list) > 3:
                det = self.fast_log_det(B)*n
            else:
                det = self.fast_log_det(B)
        return det.squeeze()

    def calculate_KL(self):
        tr_term = 1.
        T = self.ones*self.mu_prior - self.core_param
        log_term_1 = 0
        log_term_2 = 0
        for key in self.n_dict.keys():
            trace,D,cov,B = self.get_trace_term_KL(key)
            tr_term = tr_term*trace
            fix_det = getattr(self,f'prior_log_det_{key}')
            log_term_1 = log_term_1 + fix_det
            log_term_2 = log_term_2 + self.get_log_term(key,cov,D,B)
            if self.RFF_dict[key]:
                ref = lazy_mode_product(T,B.t(),key)
                ref = lazy_mode_product(ref,B,key)
                ref = ref + D*T
            else:
                ref = lazy_mode_product(T,cov,key)
        log_term = log_term_1 - log_term_2
        middle_term = torch.sum(ref * T)
        return tr_term + middle_term + log_term

    def get_L(self,key):
        D = getattr(self, f'D_{key}')
        B = getattr(self, f'B_{key}')
        L = torch.tril(B @ B.t()) + D
        return L

    def build_cov(self,key):
        D = getattr(self,f'D_{key}')
        B = getattr(self,f'B_{key}')
        L = torch.tril(B@B.t())+D
        cov = L@L.t()
        return cov,D,L

    def forward_reparametrization(self, indices):
        noise = torch.randn_like(self.core_param)
        noise_2 = torch.randn(*self.noise_shape).to(self.device)
        for key, val in self.n_dict.items(): #Sample from multivariate
            noise = lazy_mode_hadamard(noise,getattr(self,f'D_{key}'), key)
            noise_2 = lazy_mode_product(noise_2, getattr(self,f'B_{key}'), key)
        T = self.core_param + noise_2 + noise
        T = self.apply_kernels(T)
        if self.kernel_eval_mode:
            self.recalculate_priors()
        KL = self.calculate_KL()
        if self.full_grad:
            return T,KL
        else:
            if len(indices.shape) > 1:
                indices = indices.unbind(1)
            return T.permute(self.permutation_list)[indices], KL

    def sample(self,indices):
        noise = torch.randn_like(self.core_param)
        noise_2 = torch.randn(*self.noise_shape).to(self.device)
        for key, val in self.n_dict.items():  # Sample from multivariate
            noise = lazy_mode_hadamard(noise, getattr(self, f'D_{key}'), key)
            noise_2 = lazy_mode_product(noise_2, getattr(self, f'B_{key}'), key)
        T = self.core_param + noise_2 + noise
        T = self.apply_kernels(T)
        if self.full_grad:
            return T
        else:
            if len(indices.shape) > 1:
                indices = indices.unbind(1)
            return T.permute(self.permutation_list)[indices]

    def forward(self,indices):
        T = self.apply_kernels(self.core_param)
        T_cross_sigma = self.apply_cross_kernel() #Fundamental error, think about kronecker product and flattening sigma_p instead
        if self.kernel_eval_mode:
            self.recalculate_priors()
        KL = self.calculate_KL()
        if self.full_grad:
            return T,T_cross_sigma, KL
        else:
            if len(indices.shape) > 1:
                indices = indices.unbind(1)
            return T.permute(self.permutation_list)[indices],T_cross_sigma.permute(self.permutation_list)[indices], KL

    def apply_cross_kernel(self):
        T = self.ones
        for key, val in self.n_dict.items():
            if val is not None:
                if self.kernel_eval_mode:
                    X = getattr(self, f'kernel_data_{key}')
                    if self.deep_mode:
                        f = getattr(self, f'transformation_{key}')
                        X = f(X)
                    tmp_kernel_func = getattr(self, f'kernel_{key}')
                    if tmp_kernel_func.__class__.__name__=='RFF':
                        val = tmp_kernel_func(X)
                    else:
                        val = tmp_kernel_func(X).evaluate()
                if not self.RFF_dict[key]:
                    B = self.get_L(key)
                    vec_apply = torch.sum((val@B)**2,dim=1,keepdim=True)
                    T = lazy_mode_hadamard(T,vec_apply,key)
                else:
                    D = getattr(self, f'D_{key}') ** 2
                    B = getattr(self, f'B_{key}')
                    vec_apply = val.t()@B
                    vec_apply = torch.sum((val@vec_apply)**2,dim=1,keepdim=True)+D
                    T = lazy_mode_hadamard(T,vec_apply,key)
        return T

    def mean_forward(self,indices):
        """Do tensor ops"""
        T = self.core_param
        for key, val in self.n_dict.items():
            if val is not None:
                if self.kernel_eval_mode:
                    tmp_kernel_func = getattr(self, f'kernel_{key}')
                    val = tmp_kernel_func(getattr(self, f'kernel_data_{key}'))
                    self.n_dict[key] = val
                    self.recalculate_priors()
                if not self.RFF_dict[key]:
                    T = lazy_mode_product(T, val, key)
                else:
                    T = lazy_mode_product(T, val.t(), key)
                    T = lazy_mode_product(T, val, key)
        if self.full_grad:
            return T
        else:
            if len(indices.shape) > 1:
                indices = indices.unbind(1)
            return T.permute(self.permutation_list)[indices]

