import torch
import tensorly
tensorly.set_backend('pytorch')
import gpytorch
import math
import timeit
from KFT.core_components import TT_component,TT_kernel_component,lazy_mode_product,lazy_mode_hadamard,transpose_khatri_rao,KFTR_temporal_regulizer
PI  = math.pi
torch.set_printoptions(profile="full")


class KFTR_temporal_regulizer_VI(KFTR_temporal_regulizer):
    def __init__(self, r_1, n_list, r_2, time_idx, base_ref_int, lag_set_tensor, lambda_W, lambda_T_x):
        super(KFTR_temporal_regulizer_VI, self).__init__(r_1, n_list, r_2, time_idx, base_ref_int, lag_set_tensor, lambda_W, lambda_T_x)

    def calc_autoregressive_predictions(self,time_component):
        x_data = time_component.index_select(self.time_idx+1,self.idx_extractor.flatten())
        x_data = torch.stack(x_data.chunk(x_data.shape[self.time_idx+1]//self.lag_size,dim=self.time_idx+1),dim=self.time_idx+1)
        x_data = (x_data*self.W).sum(dim=self.time_idx+2)
        return x_data

    def calc_autoregressive_predictions_square(self,time_component):
        x_data = time_component.index_select(self.time_idx+1,self.idx_extractor.flatten())
        x_data = torch.stack(x_data.chunk(x_data.shape[self.time_idx+1]//self.lag_size,dim=self.time_idx+1),dim=self.time_idx+1)
        x_data = (x_data*self.W**2).sum(dim=self.time_idx+2)
        return x_data

    def calc_error(self,
                   x_square_ref,
                   x_ref,
                   predictions_square,
                   predictions_x_ref,
                   predictions_x_ref_square_W
                   ):
        error = x_square_ref - 2.* x_ref * predictions_x_ref + predictions_square + predictions_x_ref**2-predictions_x_ref_square_W
        return error

    def calculate_KFTR_VI(self,x_square_term,x_term):
        x_square_ref  =  x_square_term.index_select(self.time_idx+1,self.indices_iterate.squeeze())
        x_ref  =  x_term.index_select(self.time_idx+1,self.indices_iterate.squeeze())
        predictions_square = self.calc_autoregressive_predictions_square(x_square_term)
        predictions_x_ref = self.calc_autoregressive_predictions(x_term)
        predictions_x_ref_square_W = self.calc_autoregressive_predictions_square(x_term)
        err = self.calc_error( x_square_ref,
                   x_ref,
                   predictions_square,
                   predictions_x_ref,
                   predictions_x_ref_square_W
                         )
        return err.mean()*self.lambda_T_x
    def get_reg(self):
        return self.lambda_W * torch.mean(self.W ** 2)

class variational_TT_component(TT_component):
    def __init__(self, r_1, n_list, r_2, cuda=None, config=None, init_scale=1.0, old_setup=False, double_factor=False, sub_R=1, mu_prior=0, sigma_prior=-1):
        super(variational_TT_component, self).__init__(r_1, n_list, r_2, cuda, config, init_scale, old_setup,
                                                       double_factor=double_factor, sub_R=sub_R)
        self.variance_parameters = torch.nn.Parameter(math.log(init_scale)*torch.ones(*self.shape_list),requires_grad=True)
        self.train_group = ['mean','var']
        self.register_buffer('mu_prior',torch.tensor(mu_prior))
        self.register_buffer('sigma_prior',torch.tensor(sigma_prior))

    def calculate_KL(self,mean,sig):
        KL = 0.5*( ((mean-self.mu_prior)**2+ sig.exp())/self.sigma_prior.exp()-1-(sig-self.sigma_prior)).mean().squeeze()
        return KL

    def calculate_KL_likelihood(self,mean,sig):
        KL = 0.5*( ((mean-self.mu_prior)**2+ sig.exp())/self.sigma_prior.exp()-1-(sig-self.sigma_prior)).flatten(1).sum(dim=1).squeeze()
        return KL

    def get_KL(self,indices):
        if self.full_grad or not self.dual:
            KL = self.calculate_KL_likelihood(self.core_param, self.variance_parameters)
        else:
            mean = self.core_param.permute(self.permutation_list)[indices]
            sig = self.variance_parameters.permute(self.permutation_list)[indices]
            KL = self.calculate_KL_likelihood(mean, sig)
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

    def toggle_mean_var(self, train_mean): #Fix turn on and turn off mechanics being incorrect!
        self.core_param.requires_grad = train_mean
        self.variance_parameters.requires_grad = not train_mean
        if train_mean:
            self.train_group=['mean']
        else:
            self.train_group=['var']


    def turn_off(self):
        for el in self.train_group:
            if el=='mean':
                self.core_param.requires_grad=False
            elif el=='var':
                self.variance_parameters.requires_grad=False

    def turn_on(self):
        for el in self.train_group:
            if el=='mean':
                self.core_param.requires_grad=True
            elif el=='var':
                self.variance_parameters.requires_grad=True

class univariate_variational_kernel_TT(TT_kernel_component):
    def __init__(self, r_1, n_list, r_2, side_information_dict, kernel_para_dict, cuda=None,config=None,init_scale=1.0,mu_prior=0,sigma_prior=-1):
        super(univariate_variational_kernel_TT, self).__init__(r_1, n_list, r_2, side_information_dict,
                                                                 kernel_para_dict, cuda, config, init_scale)
        self.variance_parameters = torch.nn.Parameter(math.log(init_scale)*torch.ones(*self.shape_list),requires_grad=True)
        self.cached_tensor_2 = None
        self.train_group = ['mean','var']
        self.register_buffer('mu_prior', torch.tensor(mu_prior))
        self.register_buffer('sigma_prior', torch.tensor(sigma_prior))

    def calculate_KL(self,mean,sig):
        KL = 0.5*( ((mean-self.mu_prior)**2+ sig.exp())/self.sigma_prior.exp()-1-(sig-self.sigma_prior)).mean().squeeze()
        return KL
    def calculate_KL_likelihood(self,mean,sig):
        KL = 0.5*( ((mean-self.mu_prior)**2+ sig.exp())/self.sigma_prior.exp()-1-(sig-self.sigma_prior)).flatten(1).sum(dim=1).squeeze()
        return KL

    def get_KL(self,indices):
        if self.full_grad or not self.dual:
            KL = self.calculate_KL_likelihood(self.core_param, self.variance_parameters)
        else:
            mean = self.core_param.permute(self.permutation_list)[indices]
            sig = self.variance_parameters.permute(self.permutation_list)[indices]
            KL = self.calculate_KL_likelihood(mean, sig)
        return KL

    def apply_square_kernel(self,T):
        for key, val in self.n_dict.items():
            if val is not None:
                if self.dual:
                    if self.kernel_eval_mode:
                        val = self.side_data_eval(key)
                    if not self.RFF_dict[key]:
                        T = lazy_mode_product(T, val*val, key)
                    else:
                        RFF_squared = transpose_khatri_rao(val, val)
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

    def cache_results(self):
        with torch.no_grad():
            self.cached_tensor = self.apply_kernels(self.core_param)
            self.cached_tensor_2 = self.apply_square_kernel(self.variance_parameters.exp())
            if self.full_grad or not self.dual:
                self.cached_reg = self.calculate_KL(self.core_param, self.variance_parameters)

    def forward(self,indices):
        if not self.cache_mode:
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
        else:

            if self.full_grad or not self.dual:
                pass
            else:
                with torch.no_grad():
                    mean = self.core_param.permute(self.permutation_list)[indices]
                    sig = self.variance_parameters.permute(self.permutation_list)[indices]
                    self.cached_reg = self.calculate_KL(mean, sig)
            if self.full_grad:
                return self.cached_tensor,self.cached_tensor_2,self.cached_reg
            else:
                if len(indices.shape) > 1:
                    indices = indices.unbind(1)
                return self.cached_tensor.permute(self.permutation_list)[indices], self.cached_tensor_2.permute(self.permutation_list)[
                    indices], self.cached_reg

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

    def toggle_mean_var(self, train_mean): #Fix turn on and turn off mechanics being incorrect!
        self.core_param.requires_grad = train_mean
        self.variance_parameters.requires_grad = not train_mean
        if train_mean:
            self.train_group=['mean']
        else:
            self.train_group=['var']

    def turn_off(self):
        for el in self.train_group:
            if el=='mean':
                self.core_param.requires_grad=False
            elif el=='var':
                self.variance_parameters.requires_grad=False

    def turn_on(self):
        for el in self.train_group:
            if el=='mean':
                self.core_param.requires_grad=True
            elif el=='var':
                self.variance_parameters.requires_grad=True

    def kernel_train_mode_on(self):
        self.kernel_eval_mode = True
        if self.dual:
            for key,val in self.n_dict.items():
                if val is not None:
                    k = getattr(self,f'kernel_{key}')
                    k.raw_lengthscale.requires_grad = True
                    if k.__class__.__name__=='PeriodicKernel':
                        k.raw_period_length.requires_grad=True
            return 0

    def kernel_train_mode_off(self):
        self.kernel_eval_mode = False
        if self.dual:
            for key,val in self.n_dict.items():
                if val is not None:
                    k = getattr(self,f'kernel_{key}')
                    k.raw_lengthscale.requires_grad = False
                    if k.__class__.__name__=='PeriodicKernel':
                        k.raw_period_length.requires_grad=False
                    self.set_side_info(key)
        return 0

class multivariate_variational_kernel_TT(TT_kernel_component):
    def __init__(self, r_1, n_list, r_2, side_information_dict, kernel_para_dict, cuda=None,config=None,init_scale=1.0, mu_prior=1):
        super(multivariate_variational_kernel_TT, self).__init__(r_1, n_list, r_2, side_information_dict,
                                                                 kernel_para_dict, cuda, config, init_scale)
        self.cached_tensor_2=None
        self.register_buffer('mu_prior', torch.tensor(mu_prior))
        self.register_buffer('ones',torch.ones_like(self.core_param))
        self.noise_shape = [r_1]
        for key,val in  self.n_dict.items():
            self.set_variational_parameters(key,val)
        self.noise_shape.append(r_2)
        self.train_group = ['mean','var']

    def turn_off(self):
        for el in self.train_group:
            if el=='mean':
                self.core_param.requires_grad=False
            elif el=='var':
                for key in self.n_dict.keys():
                    getattr(self, f'D_{key}').requires_grad =False
                    getattr(self, f'B_{key}').requires_grad = False

    def turn_on(self):
        for el in self.train_group:
            if el=='mean':
                self.core_param.requires_grad=True
            elif el=='var':
                for key in self.n_dict.keys():
                    getattr(self, f'D_{key}').requires_grad = True
                    getattr(self, f'B_{key}').requires_grad = True

    def toggle_mean_var(self, train_mean):
        self.core_param.requires_grad = train_mean
        for key in self.n_dict.keys():
            getattr(self,f'D_{key}').requires_grad = not train_mean
            getattr(self, f'B_{key}').requires_grad = not train_mean
        if train_mean:
            self.train_group=['mean']
        else:
            self.train_group=['var']

    def kernel_train_mode_off(self):
        self.kernel_eval_mode = False
        if self.dual:
            for key, val in self.n_dict.items():
                if val is not None:
                    k = getattr(self, f'kernel_{key}')
                    k.raw_lengthscale.requires_grad = False
                    if k.__class__.__name__ == 'PeriodicKernel':
                        k.raw_period_length.requires_grad = False
                    self.set_side_info(key)
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
                    eye = getattr(self,f'prior_eye_{key}')
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
                        prior_log_det = -gpytorch.logdet(mat) * mat.shape[0]
                    else:
                        prior_log_det = -gpytorch.logdet(mat)
                self.register_buffer(f'prior_log_det_{key}', prior_log_det)

    def set_variational_parameters(self,key,val):
        if val is None:
            self.RFF_dict[key] = True
        if self.RFF_dict[key]:
            if val is None:
                R = int(round(math.log(self.shape_list[key])))
                self.register_buffer(f'sig_p_2_{key}', torch.tensor(1., device=self.device))
                eye = torch.eye(R).to(self.device)
                self.register_buffer(f'eye_{key}', eye)
                self.register_buffer(f'r_const_{key}', torch.tensor(R).float())
                self.register_buffer(f'Phi_T_{key}', torch.ones_like(eye))
                self.register_buffer(f'Phi_T_trace_{key}', eye.sum())
                self.register_buffer(f'RFF_dim_const_{key}', torch.tensor(0))
                prior_log_det = torch.tensor(0)
            else:
                mat  = val
                R = int(round(math.log(self.n_dict[key].shape[0])))
                self.register_buffer(f'sig_p_2_{key}',torch.tensor(1e-2))
                eye = torch.eye(R,device=self.device)
                self.register_buffer(f'eye_{key}',eye)
                self.register_buffer(f'r_const_{key}',torch.tensor(R).float())
                self.register_buffer(f'Phi_T_{key}',mat.t()@mat)
                sig_p_2 = getattr(self, f'sig_p_2_{key}')
                raw_cov = getattr(self,f'Phi_T_{key}')
                self.register_buffer(f'Phi_T_trace_{key}',raw_cov.diag().sum())
                RFF_dim_const = mat.shape[0]-R
                self.register_buffer(f'RFF_dim_const_{key}',torch.tensor(RFF_dim_const))
                self.register_buffer(f'prior_eye_{key}',torch.eye(mat.shape[1],device=self.device))
                prior_eye = getattr(self,f'prior_eye_{key}')
                if len(self.shape_list)>3:
                    prior_log_det = -(gpytorch.logdet(raw_cov+prior_eye*sig_p_2))*(mat.shape[0])+ torch.log(sig_p_2)*(RFF_dim_const)
                else:
                    prior_log_det = -(gpytorch.logdet(raw_cov+prior_eye*sig_p_2)) + torch.log(sig_p_2)*(RFF_dim_const)
            setattr(self, f'D_{key}', torch.nn.Parameter(self.init_scale * torch.tensor([1.]), requires_grad=True))
            setattr(self, f'B_{key}', torch.nn.Parameter(1e-5*torch.randn(self.shape_list[key], R), requires_grad=True))
        else:
            R = int(round(20.*math.log(self.n_dict[key].shape[0])))
            self.register_buffer(f'reg_diag_cholesky_{key}',torch.eye(val.shape[0],device=self.device)*1e-3)
            mat  = val + getattr(self,f'reg_diag_cholesky_{key}')
            if len(self.shape_list) > 3:
                prior_log_det = -gpytorch.logdet(mat)*(mat.shape[0])
            else:
                prior_log_det = -gpytorch.logdet(mat)
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
            if self.n_dict[key] is not None:
                C = transpose_khatri_rao(B,self.n_dict[key])
            else:
                C = B
            trace_term = torch.sum(C.t()@C) + sig_p_2*B_times_B_sum + D*getattr(self,f'Phi_T_trace_{key}') + D*sig_p_2*getattr(self,f'n_const_{key}')
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

    def get_KL(self,indices=[]):
        return self.calculate_KL()

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
        L = torch.triu(B @ B.t()) + D
        return L

    def build_cov(self,key):
        D = getattr(self,f'D_{key}')
        B = getattr(self,f'B_{key}')
        L = torch.tril(B@B.t())+D
        cov = L@L.t()
        return cov,D,L

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
        if not self.cache_mode:
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
        else:
            if self.full_grad:
                return self.cached_tensor, self.cached_tensor_2, self.cached_reg
            else:
                if len(indices.shape) > 1:
                    indices = indices.unbind(1)
                return self.cached_tensor.permute(self.permutation_list)[indices], self.cached_tensor_2.permute(self.permutation_list)[
                    indices], self.cached_reg

    def cache_results(self):
        with torch.no_grad():
            self.cached_tensor = self.apply_kernels(self.core_param)
            self.cached_tensor_2 = self.apply_cross_kernel()
            self.cached_reg = self.calculate_KL()

    def apply_cross_kernel(self):
        T = self.ones
        for key, val in self.n_dict.items():
            if val is not None:
                if self.kernel_eval_mode:
                   val = self.side_data_eval(key)
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

