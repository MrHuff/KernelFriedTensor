from KFT.KFT_VI import *
from KFT.core_VI_components import *


class KFT_forecast_VI(variational_KFT):
    def __init__(self,initialization_data,shape_permutation,lags,base_ref_int,cuda=None,config=None,old_setup=False,lambdas=None):
        super(KFT_forecast_VI,self).__init__(initialization_data,shape_permutation,cuda,config,old_setup,lambdas)
        self.extract_temporal_dimension(initialization_data)
        v = initialization_data[self.tt_core_temporal_idx]
        self.KFTR = KFTR_temporal_regulizer_VI(
            r_1=v['r_1'],
            n_list=v['n_list'],
            r_2=v['r_2'],
            time_idx= self.KFTR_time_idx,
            base_ref_int=base_ref_int,
            lag_set_tensor=lags,
            lambda_W = lambdas['lambda_W'],
            lambda_T_x = lambdas['lambda_T_x']
        )

    def extract_temporal_dimension(self,initialization_data):
        self.temporal_tag = self.config['temporal_tag']
        for i,v in initialization_data.items():
            if self.temporal_tag in self.ii[i]:
                self.tt_core_temporal_idx = i
                self.KFTR_time_idx = self.ii[i].index(self.temporal_tag)

    def get_time_component(self):
        if not self.old_setup:
            tt = self.TT_cores[str(self.tt_core_temporal_idx)]
            tt_prime = self.TT_cores_prime[str(self.tt_core_temporal_idx)]
            if self.has_dual_kernel_component(self.tt_core_temporal_idx):
                if tt.__class__.__name__ == 'univariate_variational_kernel_TT':
                    square_term_variance = tt.apply_square_kernel(tt.variance_parameters.exp())
                else:
                    square_term_variance = tt.apply_cross_kernel()
                basic_term = tt.apply_kernels(tt.core_param)
            else:
                square_term_variance = tt.variance_parameters.exp()
                basic_term = tt.core_param
            x_square_term = (tt_prime.core_param ** 2 + tt_prime.variance_parameters.exp()) * (
                        basic_term ** 2 + square_term_variance)
            x_term = basic_term * tt_prime.variance_parameters.exp()

        else:
            tt = self.TT_cores[str(self.tt_core_temporal_idx)]
            if self.has_dual_kernel_component(self.tt_core_temporal_idx):
                if tt.__class__.__name__ == 'univariate_variational_kernel_TT':
                    square_term_variance = tt.apply_square_kernel(tt.variance_parameters.exp())
                else:
                    square_term_variance = tt.apply_cross_kernel()
                basic_term = tt.apply_kernels(tt.core_param)
            else:
                square_term_variance = tt.variance_parameters.exp()
                basic_term = tt.core_param
            x_square_term = basic_term ** 2 + square_term_variance
            x_term = basic_term
        return x_square_term,x_term

    def activate_W_mode(self):
        self.turn_off_all()
        self.current_update_pointers = [i for i in self.ii.keys()]
        self.KFTR.W.requires_grad=True

    def deactivate_W_mode(self):
        self.KFTR.W.requires_grad=False

    def forward(self,indices):
        indices = indices[:,self.shape_permutation]
        middle,last_term, total_KL = self.collect_core_outputs(indices)
        if (self.tt_core_temporal_idx in self.current_update_pointers) and (not self.TT_cores[str(self.tt_core_temporal_idx)].kernel_eval_mode):
            x_square_term,x_term = self.get_time_component()
            T_reg = self.KFTR.calculate_KFTR_VI(x_square_term,x_term) + self.KFTR.get_reg()
        else:
            T_reg = 0.
        print('KL',total_KL)
        print('T_reg',T_reg)
        return middle,last_term,total_KL+T_reg

class KFT_forecast_VI_LS(varitional_KFT_scale):
    def __init__(self, initialization_data,lambdas,shape_permutation,lags,base_ref_int,cuda=None, config=None, old_setup=False): #decomposition_data = {0:{'ii':[0,1],'lambda':0.01,r_1:1 n_list=[10,10],r_2:10,'has_side_info':True, side_info:{1:x_1,2:x_2},kernel_para:{'ls_factor':0.5, 'kernel_type':'RBF','nu':2.5} },1:{}}
        super(KFT_forecast_VI_LS, self).__init__(initialization_data=initialization_data,
                                                 lambdas=lambdas,
                                                 shape_permutation=shape_permutation,
                                                 cuda=cuda,
                                                 config=config,
                                                 old_setup=old_setup)
        self.extract_temporal_dimension(initialization_data)
        v = initialization_data[self.tt_core_temporal_idx]
        self.KFTR = KFTR_temporal_regulizer_VI(
            r_1=v['r_1'],
            n_list=v['prime_list'],
            r_2=v['r_2'],
            time_idx= self.KFTR_time_idx,
            base_ref_int=base_ref_int,
            lag_set_tensor=lags,
            lambda_W = lambdas['lambda_W'],
            lambda_T_x = lambdas['lambda_T_x']
        )
        self.KFTR_s = KFTR_temporal_regulizer_VI(
            r_1=v['r_1_latent'],
            n_list=v['prime_list'],
            r_2=v['r_2_latent'],
            time_idx= self.KFTR_time_idx,
            base_ref_int=base_ref_int,
            lag_set_tensor=lags,
            lambda_W = lambdas['lambda_W'],
            lambda_T_x = lambdas['lambda_T_x']
        )
        self.KFTR_b = KFTR_temporal_regulizer_VI(
            r_1=v['r_1_latent'],
            n_list=v['prime_list'],
            r_2=v['r_2_latent'],
            time_idx= self.KFTR_time_idx,
            base_ref_int=base_ref_int,
            lag_set_tensor=lags,
            lambda_W = lambdas['lambda_W'],
            lambda_T_x = lambdas['lambda_T_x']
        )
        self.deactivate_W_mode()

    def get_time_component(self):
        if not self.old_setup:
            tt = self.TT_cores[str(self.tt_core_temporal_idx)]
            tts = self.TT_cores_s[str(self.tt_core_temporal_idx)]
            ttb = self.TT_cores_b[str(self.tt_core_temporal_idx)]
            if self.has_dual_kernel_component(self.tt_core_temporal_idx):
                if tt.__class__.__name__ == 'univariate_variational_kernel_TT':
                    square_term_variance = tt.apply_square_kernel(tt.variance_parameters.exp())
                else:
                    square_term_variance = tt.apply_cross_kernel()
                basic_term = tt.apply_kernels(tt.core_param)
            else:
                square_term_variance = tt.variance_parameters.exp()
                basic_term = tt.core_param
            square_term_variance_s = tts.variance_parameters.exp()
            basic_term_s = tts.core_param
            square_term_variance_b = ttb.variance_parameters.exp()
            basic_term_b = ttb.core_param
        else:
            tt = self.TT_cores[str(self.tt_core_temporal_idx)]
            if self.has_dual_kernel_component(self.tt_core_temporal_idx):
                if tt.__class__.__name__ == 'univariate_variational_kernel_TT':
                    square_term_variance = tt.apply_square_kernel(tt.variance_parameters.exp())
                else:
                    square_term_variance = tt.apply_cross_kernel()
                basic_term = tt.apply_kernels(tt.core_param)
            else:
                square_term_variance = tt.variance_parameters.exp()
                basic_term = tt.core_param

            square_term_variance_s = 0.
            basic_term_s = 0.
            square_term_variance_b = 0.
            basic_term_b = 0.
        x_term,x_square_term = self.get_x_terms(basic_term,square_term_variance)
        x_term_s,x_square_term_s = self.get_x_terms(basic_term_s,square_term_variance_s)
        x_term_b,x_square_term_b = self.get_x_terms(basic_term_b,square_term_variance_b)
        return x_term,x_square_term, x_term_s,x_square_term_s,x_term_b,x_square_term_b

    def get_x_terms(self,basic_term,square_term_variance):
        x_square_term = basic_term ** 2 + square_term_variance
        x_term = basic_term
        return x_term, x_square_term


    def activate_W_mode(self):
        self.turn_off_all()
        self.current_update_pointers = [i for i in self.ii.keys()]
        self.KFTR.W.requires_grad=True
        self.KFTR_s.W.requires_grad=True
        self.KFTR_b.W.requires_grad=True

    def deactivate_W_mode(self):
        self.KFTR.W.requires_grad=False
        self.KFTR_s.W.requires_grad=False
        self.KFTR_b.W.requires_grad=False

    def forward(self,indices):
        indices = indices[:,self.shape_permutation]
        middle,last_term, total_KL = self.collect_core_outputs(indices)
        if (self.tt_core_temporal_idx in self.current_update_pointers) and (not self.TT_cores[str(self.tt_core_temporal_idx)].kernel_eval_mode):
            x_term,x_square_term, x_term_s,x_square_term_s,x_term_b,x_square_term_b  = self.get_time_component()
            T_reg = self.KFTR.calculate_KFTR_VI(x_square_term,x_term) + self.KFTR.get_reg()
            T_reg_s = self.KFTR_s.calculate_KFTR_VI(x_square_term_s,x_term_s) + self.KFTR_s.get_reg()
            T_reg_b = self.KFTR_b.calculate_KFTR_VI(x_square_term_b,x_term_b) + self.KFTR_b.get_reg()
            T_reg_tot = T_reg+T_reg_s+T_reg_b
        else:
            T_reg_tot = 0.
        return middle,last_term,total_KL+T_reg_tot

    def extract_temporal_dimension(self,initialization_data):
        self.temporal_tag = self.config['temporal_tag']
        for i,v in initialization_data.items():
            if self.temporal_tag in self.ii[i]:
                self.tt_core_temporal_idx = i
                self.KFTR_time_idx = self.ii[i].index(self.temporal_tag)