from KFT.KFT import *

class KFT_forecast(KFT):
    def __init__(self, initialization_data,lambdas,shape_permutation,lags,base_ref_int,cuda=None, config=None, old_setup=False): #decomposition_data = {0:{'ii':[0,1],'lambda':0.01,r_1:1 n_list=[10,10],r_2:10,'has_side_info':True, side_info:{1:x_1,2:x_2},kernel_para:{'ls_factor':0.5, 'kernel_type':'RBF','nu':2.5} },1:{}}
        super(KFT_forecast, self).__init__(initialization_data,lambdas,shape_permutation, cuda, config, old_setup)
        self.extract_temporal_dimension(initialization_data)
        v = initialization_data[self.tt_core_temporal_idx]
        self.KFTR = KFTR_temporal_regulizer(
            r_1=v['r_1'],
            n_list=v['n_list'],
            r_2=v['r_2'],
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
            if self.has_dual_kernel_component(self.tt_core_temporal_idx):
                temporal_comp = self.TT_cores_prime[str(self.tt_core_temporal_idx)].core_param *  tt.get_temporal_compoment()
            else:
                temporal_comp = self.TT_cores_prime[str(self.tt_core_temporal_idx)].core_param *  tt.core_param
        else:
            tt = self.TT_cores[str(self.tt_core_temporal_idx)]
            if self.has_dual_kernel_component(self.tt_core_temporal_idx):
                temporal_comp = tt.get_temporal_compoment()
            else:
                temporal_comp = tt.core_param

        return temporal_comp

    def activate_W_mode(self):
        self.turn_off_all()
        self.current_update_pointers = [i for i in self.ii.keys()]
        self.KFTR.W.requires_grad=True
    def deactivate_W_mode(self):
        self.KFTR.W.requires_grad=False

    def forward(self,indices):
        indices = indices[:,self.shape_permutation]
        preds_list,regularization = self.collect_core_outputs(indices)
        if (self.tt_core_temporal_idx in self.current_update_pointers) and (not self.TT_cores[str(self.tt_core_temporal_idx)].kernel_eval_mode):
            temporal_reg = self.get_time_component()
            T_reg = self.KFTR.calculate_KFTR(temporal_reg) + self.KFTR.get_reg()
        else:
            T_reg = 0.
        if self.full_grad:
            preds = self.edge_mode_collate(preds_list=preds_list)
            return preds[torch.unbind(indices,dim=1)],regularization+T_reg
        else:
            preds = self.bmm_collate(preds_list=preds_list)
            return preds,regularization+T_reg

    def extract_temporal_dimension(self,initialization_data):
        self.temporal_tag = self.config['temporal_tag']
        for i,v in initialization_data.items():
            if self.temporal_tag in self.ii[i]:
                self.tt_core_temporal_idx = i
                self.KFTR_time_idx = self.ii[i].index(self.temporal_tag)

class KFT_forecast_LS(KFT_scale):
    def __init__(self, initialization_data,lambdas,shape_permutation,lags,base_ref_int,cuda=None, config=None, old_setup=False): #decomposition_data = {0:{'ii':[0,1],'lambda':0.01,r_1:1 n_list=[10,10],r_2:10,'has_side_info':True, side_info:{1:x_1,2:x_2},kernel_para:{'ls_factor':0.5, 'kernel_type':'RBF','nu':2.5} },1:{}}
        super(KFT_forecast_LS, self).__init__(initialization_data,lambdas,shape_permutation, cuda, config, old_setup)
        self.extract_temporal_dimension(initialization_data)
        v = initialization_data[self.tt_core_temporal_idx]
        self.KFTR = KFTR_temporal_regulizer(
            r_1=v['r_1'],
            n_list=v['prime_list'],
            r_2=v['r_2'],
            time_idx= self.KFTR_time_idx,
            base_ref_int=base_ref_int,
            lag_set_tensor=lags,
            lambda_W = lambdas['lambda_W'],
            lambda_T_x = lambdas['lambda_T_x']
        )
        self.KFTR_s = KFTR_temporal_regulizer(
            r_1=v['r_1_latent'],
            n_list=v['prime_list'],
            r_2=v['r_2_latent'],
            time_idx= self.KFTR_time_idx,
            base_ref_int=base_ref_int,
            lag_set_tensor=lags,
            lambda_W = lambdas['lambda_W'],
            lambda_T_x = lambdas['lambda_T_x']
        )
        self.KFTR_b = KFTR_temporal_regulizer(
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
                temporal_comp = tt.get_temporal_compoment()
            else:
                temporal_comp =  tt.core_param
            temporal_comp_s = tts.core_param
            temporal_comp_b = ttb.core_param
        else:
            tt = self.TT_cores[str(self.tt_core_temporal_idx)]
            if self.has_dual_kernel_component(self.tt_core_temporal_idx):
                temporal_comp = tt.get_temporal_compoment()
            else:
                temporal_comp = tt.core_param
            temporal_comp_s = 0.
            temporal_comp_b = 0.
        return temporal_comp,temporal_comp_s,temporal_comp_b

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
        preds,regularization = self.collect_core_outputs(indices)
        if (self.tt_core_temporal_idx in self.current_update_pointers) and (not self.TT_cores[str(self.tt_core_temporal_idx)].kernel_eval_mode):
            temporal_reg,temporal_reg_s,temporal_reg_b = self.get_time_component()
            T_reg = self.KFTR.calculate_KFTR(temporal_reg) + self.KFTR.get_reg()
            T_reg_s = self.KFTR_s.calculate_KFTR(temporal_reg_s) + self.KFTR_s.get_reg()
            T_reg_b = self.KFTR_b.calculate_KFTR(temporal_reg_b) + self.KFTR_b.get_reg()
            T_reg_tot = T_reg+T_reg_s+T_reg_b
        else:
            T_reg_tot = 0.
        if self.full_grad:
            return preds[torch.unbind(indices,dim=1)],regularization+T_reg_tot
        else:
            return preds,regularization+T_reg_tot

    def extract_temporal_dimension(self,initialization_data):
        self.temporal_tag = self.config['temporal_tag']
        for i,v in initialization_data.items():
            if self.temporal_tag in self.ii[i]:
                self.tt_core_temporal_idx = i
                self.KFTR_time_idx = self.ii[i].index(self.temporal_tag)

