from KFT.KFT import *
from KFT.core_VI_components import univariate_variational_kernel_TT,multivariate_variational_kernel_TT,variational_TT_component

class variational_KFT(KFT):
    def __init__(self,initialization_data,shape_permutation,cuda=None,config=None,old_setup=False,lambdas=None):
        super(variational_KFT, self).__init__(initialization_data=initialization_data,shape_permutation=shape_permutation, cuda=cuda, config=config, old_setup=old_setup,lambdas=lambdas)
        tmp_dict = {}
        tmp_dict_prime = {}
        self.kernel_class_name = ['multivariate_variational_kernel_TT','univariate_variational_kernel_TT']
        self.full_grad = config['full_grad']
        self.ii = {}
        for i, v in initialization_data.items():
            self.ii[i] = v['ii']
            if not old_setup:
                tmp_dict_prime[str(i)] = variational_TT_component(r_1=v['r_1'],
                                                                  n_list=v['prime_list'],
                                                                  r_2=v['r_2'],
                                                                  cuda=cuda,
                                                                  config=config,
                                                                  init_scale=v['init_scale'],
                                                                  double_factor=v['double_factor'],
                                                                  sub_R=config['sub_R'],
                                                                  mu_prior=v['mu_prior_prime'],
                                                                  sigma_prior=v['sigma_prior_prime'])
            if v['has_side_info']:
                if v['multivariate'] and config['dual']:
                    tmp_dict[str(i)] = multivariate_variational_kernel_TT(r_1=v['r_1'],
                                                                          n_list=v['n_list'] if config['dual'] else v[
                                                                              'prime_list'],
                                                                          r_2=v['r_2'],
                                                                          side_information_dict=v['side_info'],
                                                                          kernel_para_dict=v['kernel_para'],
                                                                          cuda=cuda,
                                                                          config=config,
                                                                          init_scale=v['init_scale'],
                                                                          mu_prior=v['mu_prior'],
                                                                          )
                else:
                    tmp_dict[str(i)] = univariate_variational_kernel_TT(r_1=v['r_1'],
                                                                        n_list=v['n_list'] if config['dual'] else v[
                                                                            'prime_list'],
                                                                        r_2=v['r_2'],
                                                                        side_information_dict=v['side_info'],
                                                                        kernel_para_dict=v['kernel_para'],
                                                                        cuda=cuda,
                                                                        config=config,
                                                                        init_scale=v['init_scale'],
                                                                        mu_prior=v['mu_prior'],
                                                                        sigma_prior=v['sigma_prior']
                                                                        )
            else:
                tmp_dict[str(i)] = variational_TT_component(r_1=v['r_1'],
                                                            n_list= v['n_list'],
                                                            r_2=v['r_2'],
                                                            cuda=cuda,
                                                            config=config,
                                                            init_scale=v['init_scale'],
                                                            mu_prior=v['mu_prior'],
                                                            sigma_prior=v['sigma_prior']
                                                            )

        self.TT_cores = torch.nn.ModuleDict(tmp_dict)
        self.TT_cores_prime = torch.nn.ModuleDict(tmp_dict_prime)

    def get_norms(self):
        with torch.no_grad():
            for i, v in self.ii.items():
                print(torch.mean(self.TT_cores[str(i)].core_param.abs()))
                if self.TT_cores[str(i)].__class__.__name__=='univariate_variational_kernel_TT':
                    print(torch.mean(self.TT_cores[str(i)].variance_parameters))
                if not self.old_setup:
                    print(torch.mean(self.TT_cores_prime[str(i)].core_param.abs()))
                    print(torch.mean(self.TT_cores_prime[str(i)].variance_parameters))

    def toggle(self,toggle):
        for i,v in self.ii.items():
            self.TT_cores[str(i)].toggle_mean_var(toggle)
            if not self.old_setup:
                self.TT_cores_prime[str(i)].toggle_mean_var(toggle)
        return 0

    def collect_core_outputs_mean(self,indices):
        pred_outputs = []
        for i,v in self.ii.items():
            ix = indices[:,v]
            tt = self.TT_cores[str(i)]
            tt_prime = self.TT_cores_prime[str(i)]
            prime_pred = tt_prime.mean_forward(ix)
            pred = tt.mean_forward(ix)
            pred_outputs.append(pred*prime_pred)
        return pred_outputs

    def collect_core_outputs_mean_old(self,indices):
        pred_outputs = []
        for i,v in self.ii.items():
            ix = indices[:,v]
            tt = self.TT_cores[str(i)]
            pred = tt.mean_forward(ix)
            pred_outputs.append(pred)
        return pred_outputs

    def collect_core_outputs_old(self,indices):
        first_term = []
        second_term = []
        total_KL = 0
        for i, v in self.ii.items():
            ix = indices[:, v]
            tt = self.TT_cores[str(i)]
            base, extra, KL = tt(ix)
            first_term.append(base)
            second_term.append(extra)
            total_KL += KL
        if self.full_grad:
            group_func = self.edge_mode_collate
        else:
            group_func = self.bmm_collate
        middle = group_func(first_term)
        gf = group_func(second_term)
        last_term = middle ** 2 + gf
        if self.full_grad:
            middle = middle[torch.unbind(indices, dim=1)]
            last_term = last_term[torch.unbind(indices, dim=1)]
        return middle, last_term, total_KL

    def collect_core_outputs(self,indices):
        first_term = []
        second_term_a = []
        second_term_b = []
        second_term_c = []
        total_KL = 0
        for i, v in self.ii.items():
            ix = indices[:, v]
            tt = self.TT_cores[str(i)]
            tt_prime = self.TT_cores_prime[str(i)]
            M_prime, sigma_prime, KL_prime = tt_prime(ix)
            base, extra, KL = tt(ix)
            first_term.append(M_prime*base) #base preds
            second_term_a.append(extra*sigma_prime)
            second_term_b.append((M_prime**2)*extra)
            second_term_c.append(base**2*sigma_prime)
            if (i in self.current_update_pointers):
                total_KL = torch.relu(KL) + KL_prime
        if self.full_grad:
            group_func = self.edge_mode_collate
        else:
            group_func = self.bmm_collate
        middle = group_func(first_term)
        gf_a = group_func(second_term_a)
        gf_b = group_func(second_term_b)
        gf_c = group_func(second_term_c)
        last_term = middle**2 + gf_a+gf_b+gf_c
        if self.full_grad:
            middle = middle[torch.unbind(indices, dim=1)]
            last_term = last_term[torch.unbind(indices, dim=1)]
        return middle,last_term, total_KL

    def collect_core_outputs_sample_old(self, indices):
        pred_outputs = []
        total_KL = 0
        for i,v in self.ii.items():
            ix = indices[:,v]
            tt = self.TT_cores[str(i)]
            pred= tt.sample(ix)
            pred_outputs.append(pred)
            KL = tt.get_KL(ix)
            total_KL += KL

        return pred_outputs,total_KL

    def collect_core_outputs_sample(self, indices):
        pred_outputs = []
        total_KL = 0

        for i,v in self.ii.items():
            ix = indices[:,v]
            tt = self.TT_cores[str(i)]
            tt_prime = self.TT_cores_prime[str(i)]
            prime_pred = tt_prime.sample(ix)
            pred= tt.sample(ix)
            pred_outputs.append(pred*prime_pred)
            KL = tt.get_KL(ix)
            KL_prime =tt_prime.get_KL(ix)
            total_KL += KL + KL_prime

        return pred_outputs,total_KL

    def sample(self, indices):
        indices = indices[:,self.shape_permutation]
        if self.old_setup:
            preds_list,KL = self.collect_core_outputs_sample_old(indices)
        else:
            preds_list,KL = self.collect_core_outputs_sample(indices)
        if self.full_grad:
            preds = self.edge_mode_collate(preds_list)
            return preds[torch.unbind(indices, dim=1)]
        else:
            preds = self.bmm_collate(preds_list)
            return preds,KL

    def forward(self, indices):
        indices = indices[:,self.shape_permutation]
        if self.old_setup:
            middle,third, regularization = self.collect_core_outputs_old(indices)
        else:
            middle,third, regularization = self.collect_core_outputs(indices)
        return middle,third,regularization

    def mean_forward(self,indices):
        indices = indices[:,self.shape_permutation]
        if self.old:
            preds_list = self.collect_core_outputs_mean_old(indices)
        else:
            preds_list = self.collect_core_outputs_mean(indices)
        if self.full_grad:
            preds = self.edge_mode_collate(preds_list)
            return preds[torch.unbind(indices, dim=1)]
        else:
            preds = self.bmm_collate(preds_list)
            return preds


class varitional_KFT_scale(KFT_scale):
    def __init__(self,initialization_data,shape_permutation,cuda=None,config=None,old_setup=False,lambdas=None):
        super(varitional_KFT_scale, self).__init__(initialization_data=initialization_data,shape_permutation=shape_permutation, cuda=cuda, config=config, old_setup=old_setup,lambdas=lambdas)
        tmp_dict = {}
        tmp_dict_s = {}
        tmp_dict_b = {}
        self.kernel_class_name = ['multivariate_variational_kernel_TT', 'univariate_variational_kernel_TT']
        self.full_grad = config['full_grad']
        self.ii = {}
        for i, v in initialization_data.items():
            self.ii[i] = v['ii']
            tmp_dict_s[str(i)] = variational_TT_component(r_1=v['r_1_latent'],
                                                              n_list=v['prime_list'],
                                                              r_2=v['r_2_latent'],
                                                              cuda=cuda,
                                                              config=config,
                                                              mu_prior=v['mu_prior_s'],
                                                              sigma_prior=v['sigma_prior_s'],
                                                          )
            tmp_dict_b[str(i)] = variational_TT_component(r_1=v['r_1_latent'],
                                                          n_list=v['prime_list'],
                                                          r_2=v['r_2_latent'],
                                                          cuda=cuda,
                                                          config=config,
                                                          mu_prior=v['mu_prior_b'],
                                                          sigma_prior=v['sigma_prior_b']

                                                          )
            if v['has_side_info']:
                if v['multivariate'] and config['dual']:
                    tmp_dict[str(i)] = multivariate_variational_kernel_TT(r_1=v['r_1'],
                                                                          n_list= v['n_list'],
                                                                          r_2=v['r_2'],
                                                                          side_information_dict=v['side_info'],
                                                                          kernel_para_dict=v['kernel_para'],
                                                                          cuda=cuda,
                                                                          config=config,
                                                                          init_scale=1.0,
                                                                          mu_prior=v['mu_prior'],

                                                                          )
                else:
                    tmp_dict[str(i)] = univariate_variational_kernel_TT(r_1=v['r_1'],
                                                                        n_list= v['n_list'],
                                                                        r_2=v['r_2'],
                                                                        side_information_dict=v['side_info'],
                                                                        kernel_para_dict=v['kernel_para'],
                                                                        cuda=cuda,
                                                                        config=config,
                                                                        init_scale=1.0,
                                                                        mu_prior=v['mu_prior'],
                                                                        sigma_prior=v['sigma_prior']
                                                                        )
            else:
                tmp_dict[str(i)] = variational_TT_component(r_1=v['r_1'],
                                                            n_list= v['n_list'],
                                                            r_2=v['r_2'],
                                                            cuda=cuda,
                                                            config=config,
                                                            init_scale=1.0,
                                                            mu_prior=v['mu_prior'],
                                                            sigma_prior=v['sigma_prior']
                                                            )
        self.TT_cores = torch.nn.ModuleDict(tmp_dict)
        self.TT_cores_s = torch.nn.ModuleDict(tmp_dict_s)
        self.TT_cores_b = torch.nn.ModuleDict(tmp_dict_b)

    def collect_core_sample(self,indices):
        scale = []
        bias = []
        core = []
        total_KL = 0
        for i, v in self.ii.items():
            ix = indices[:, v]
            tt = self.TT_cores[str(i)]
            tt_s = self.TT_cores_s[str(i)]
            tt_b = self.TT_cores_b[str(i)]
            V_s = tt_s.sample(ix)
            V_b = tt_b.sample(ix)
            base = tt.sample(ix)
            KL = tt.get_KL(ix)
            KL_s =tt_s.get_KL(ix)
            KL_b =tt_b.get_KL(ix)
            scale.append(V_s)
            bias.append(V_b)
            core.append(base)
            total_KL+= KL_s+KL_b+torch.relu(KL)

        if self.full_grad:
            group_func = self.edge_mode_collate
        else:
            group_func = self.bmm_collate
        scale_forward = group_func(scale)
        bias_forward = group_func(bias)
        core_forward = group_func(core)

        if self.full_grad:
            T = scale_forward*core_forward+bias_forward
            return T[torch.unbind(indices, dim=1)]
        else:
            return scale_forward*core_forward+bias_forward,total_KL

    def sample(self,indices):
        indices = indices[:,self.shape_permutation]
        T,KL = self.collect_core_sample(indices)
        return T,KL

    def collect_core_outputs(self, indices):
        scale = []
        scale_var = []
        bias = []
        bias_var = []
        core = []
        core_var = []
        total_KL = 0
        for i, v in self.ii.items():
            ix = indices[:, v]
            tt = self.TT_cores[str(i)]
            tt_s = self.TT_cores_s[str(i)]
            tt_b = self.TT_cores_b[str(i)]
            V_s, var_s, KL_s = tt_s(ix)
            V_b, var_b, KL_b = tt_b(ix)
            base, extra, KL = tt(ix)
            scale.append(V_s)
            scale_var.append(var_s)
            bias.append(V_b)
            bias_var.append(var_b)
            core.append(base)
            core_var.append(extra)
            total_KL+= KL_s+KL_b+torch.relu(KL)

        if self.full_grad:
            group_func = self.edge_mode_collate
        else:
            group_func = self.bmm_collate
        scale_forward = group_func(scale)
        scale_forward_var = group_func(scale_var)
        bias_forward = group_func(bias)
        bias_forward_var = group_func(bias_var)
        core_forward = group_func(core)
        core_forward_var = group_func(core_var)
        middle = scale_forward*core_forward+bias_forward
        third_term = (scale_forward**2+scale_forward_var)*(core_forward**2+core_forward_var)+2*scale_forward*core_forward*bias_forward+bias_forward**2+bias_forward_var

        if self.full_grad:
            return middle[torch.unbind(indices, dim=1)], third_term[torch.unbind(indices, dim=1)], total_KL
        else:
            return middle, third_term, total_KL

    def forward(self,indices):
        indices = indices[:,self.shape_permutation]
        middle_term,third_term,reg = self.collect_core_outputs(indices)
        return middle_term,third_term,reg

    def toggle(self, train_means):
        for i, v in self.ii.items():
            self.TT_cores[str(i)].toggle_mean_var(train_means)
            if not self.old_setup:
                self.TT_cores_s[str(i)].toggle_mean_var(train_means)
                self.TT_cores_b[str(i)].toggle_mean_var(train_means)
        return 0

    def get_norms(self):
        with torch.no_grad():
            for i, v in self.ii.items():
                print(torch.mean(self.TT_cores[str(i)].core_param.abs()))
                if self.TT_cores[str(i)].__class__.__name__ == 'univariate_variational_kernel_TT':
                    print(torch.mean(self.TT_cores[str(i)].variance_parameters))
                if not self.old_setup:
                    print(torch.mean(self.TT_cores_s[str(i)].core_param.abs()))
                    print(torch.mean(self.TT_cores_s[str(i)].variance_parameters))
                    print(torch.mean(self.TT_cores_b[str(i)].core_param.abs()))
                    print(torch.mean(self.TT_cores_b[str(i)].variance_parameters))