import torch
import tensorly
tensorly.set_backend('pytorch')
import math
import timeit
from KFT.core_components import TT_component,TT_kernel_component,edge_mode_product,TT_component_deep
from KFT.core_VI_components import univariate_variational_kernel_TT,multivariate_variational_kernel_TT,variational_TT_component
PI  = math.pi
torch.set_printoptions(profile="full")

class KFT(torch.nn.Module):
    def __init__(self, initialization_data,lambdas, cuda=None, config=None, old_setup=False): #decomposition_data = {0:{'ii':[0,1],'lambda':0.01,r_1:1 n_list=[10,10],r_2:10,'has_side_info':True, side_info:{1:x_1,2:x_2},kernel_para:{'ls_factor':0.5, 'kernel_type':'RBF','nu':2.5} },1:{}}
        super(KFT, self).__init__()
        self.kernel_class_name = ['TT_kernel_component']
        self.cuda = cuda
        self.amp = None
        self.config = config
        self.old_setup = old_setup
        tmp_dict = {}
        tmp_dict_prime = {}
        self.full_grad = config['full_grad']
        self.ii = {}
        for i,v in initialization_data.items():
            self.ii[i] = v['ii']
            if not self.old_setup:
                if config['deep']:
                    tmp_dict_prime[str(i)] = TT_component_deep(r_1=v['r_1'],
                                                               n_list=v['n_list'],
                                                               r_2=v['r_2'],
                                                               cuda=cuda,
                                                               config=config,
                                                               init_scale=v['init_scale'],
                                                               reg_para=lambdas[f'reg_para_prime_{i}'])
                else:
                    tmp_dict_prime[str(i)] = TT_component(r_1=v['r_1'],
                                                          n_list=v['n_list'],
                                                          r_2=v['r_2'],
                                                          cuda=cuda,
                                                          config=config,
                                                          init_scale=v['init_scale'],
                                                          reg_para=lambdas[f'reg_para_prime_{i}'],
                                                          prime=v['prime'],sub_R=config['sub_R'])
                    #TODO: "Mixed prime factorizations..."
            if v['has_side_info']:
                tmp_dict[str(i)] = TT_kernel_component(r_1=v['r_1'],
                                                       n_list=v['n_list'] if config['dual'] else v['primal_list'],
                                                       r_2=v['r_2'],
                                                       side_information_dict=v['side_info'],
                                                       kernel_para_dict=v['kernel_para'],
                                                       cuda=cuda,
                                                       config=config,
                                                       init_scale=v['init_scale'],
                                                       reg_para=lambdas[f'reg_para_{i}'])
            else:
                tmp_dict[str(i)] = TT_component(r_1=v['r_1'],
                                                n_list= v['n_list'] if config['dual'] else v['primal_list'],
                                                r_2=v['r_2'],
                                                cuda=cuda,
                                                config=config,
                                                init_scale=v['init_scale'],
                                                old_setup=old_setup,
                                                reg_para=lambdas[f'reg_para_{i}'])
        self.TT_cores = torch.nn.ModuleDict(tmp_dict)
        self.TT_cores_prime = torch.nn.ModuleDict(tmp_dict_prime)

    def turn_on_all(self):
        for i, v in self.ii.items():
            self.TT_cores[str(i)].turn_on()
            self.TT_cores_prime[str(i)].turn_on()

    def turn_on_V(self):
        for i, v in self.ii.items():
            if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
                self.TT_cores[str(i)].kernel_train_mode_off()
                if self.TT_cores[str(i)].deep_kernel:
                    self.TT_cores[str(i)].deep_kernel_mode_off()
            else:
                self.TT_cores[str(i)].turn_on()
            if not self.old_setup:
                self.TT_cores_prime[str(i)].turn_off()
        return 0

    def turn_on_prime(self):
        if not self.old_setup:
            for i, v in self.ii.items():
                if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
                    self.TT_cores[str(i)].kernel_train_mode_off()
                    self.TT_cores[str(i)].turn_off()
                    if self.TT_cores[str(i)].deep_kernel:
                        self.TT_cores[str(i)].deep_kernel_mode_off()
                else:
                    self.TT_cores[str(i)].turn_off()
                self.TT_cores_prime[str(i)].turn_on()
        return 0
    #TODO: fix deep kernel by properly activating and turning of parameters
    def turn_on_deep_kernel(self):
        for i, v in self.ii.items():
            if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
                self.TT_cores[str(i)].kernel_train_mode_off()
                self.TT_cores[str(i)].turn_off()
                if self.TT_cores[str(i)].deep_kernel:
                    self.TT_cores[str(i)].deep_kernel_mode_on()
            else:
                self.TT_cores[str(i)].turn_off()
            if not self.old_setup:
                self.TT_cores_prime[str(i)].turn_off()
        return 0

    def has_kernel_component(self):
        for i, v in self.ii.items():
            if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
                for v in self.TT_cores[str(i)].n_dict.values():
                    if v is not None:
                        if self.TT_cores[str(i)].deep_kernel:
                            return True,True
                        else:
                            return True,False
        return False,False

    def turn_on_kernel_mode(self):
        for i,v in self.ii.items():
            if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
                self.TT_cores[str(i)].kernel_train_mode_on()
                if self.TT_cores[str(i)].deep_kernel:
                    self.TT_cores[str(i)].deep_kernel_mode_off()
            else:
                self.TT_cores[str(i)].turn_off()
            if not self.old_setup:
                self.TT_cores_prime[str(i)].turn_off()
        return 0

    def turn_off_kernel_mode(self):
        for i,v in self.ii.items():
            if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
                self.TT_cores[str(i)].kernel_train_mode_off()
            else:
                self.TT_cores[str(i)].turn_on()
            if not self.old_setup:
                self.TT_cores_prime[str(i)].turn_off()
        return 0

    def amp_patch(self,amp):
        for i,v in self.ii.items():
            self.TT_cores[str(i)].amp=amp
            self.TT_cores_prime[str(i)].amp=amp

    def collect_core_outputs(self,indices):
        pred_outputs = []
        reg_output=0
        for i,v in self.ii.items():
            ix = indices[:,v]
            tt = self.TT_cores[str(i)]
            if not self.old_setup:
                tt_prime = self.TT_cores_prime[str(i)]
                prime_pred,reg_prime = tt_prime(ix)
                pred, reg = tt(ix)
                if self.config['deep'] and self.config['dual']:
                    pred_outputs.append(tt_prime.nn_forward(pred)*prime_pred)
                else:
                    pred_outputs.append(pred * prime_pred)
            else:
                pred, reg = tt(ix)
                pred_outputs.append(pred)
            if self.config['dual']:
                if not self.old_setup:
                    reg_output += tt.reg_para * torch.mean(reg.float()*reg_prime.float()).abs() #numerical issue with fp 16 how fix, sum of square terms, serves as fp 16 fix
                else:
                    reg_output += torch.mean(reg.float()).abs() * tt.reg_para#numerical issue with fp 16 how fix, sum of square terms, serves as fp 16 fix
            else:
                if not self.old_setup:
                    if self.config['deep']:
                        reg_output +=tt.reg_para * torch.mean(reg)+ tt_prime.reg_para*torch.mean(reg_prime)+tt_prime.reg_para* tt_prime.nn_reg() #numerical issue with fp 16 how fix, sum of square terms, serves as fp 16 fix
                    else:
                        reg_output += tt.reg_para*torch.mean(reg)+tt_prime.reg_para*torch.mean(reg_prime) #numerical issue with fp 16 how fix, sum of square terms, serves as fp 16 fix
                else:
                    reg_output += tt.reg_para*torch.mean(reg) #numerical issue with fp 16 how fix, sum of square terms, serves as fp 16 fix
        return pred_outputs,reg_output

    def bmm_collate(self, preds_list):
        preds = preds_list[0]
        for i in range(1, len(preds_list)):
            preds = torch.bmm(preds, preds_list[i])
        return preds.squeeze()

    def edge_mode_collate(self, preds_list):
        preds = preds_list[0]
        for i in range(1, len(preds_list)):
            preds = edge_mode_product(preds, preds_list[i], len(preds.shape) - 1, 0)  # General mode product!
        return preds.squeeze()

    def forward(self,indices):
        preds_list,regularization = self.collect_core_outputs(indices)
        if self.full_grad:
            preds = self.edge_mode_collate(preds_list=preds_list)
            return preds[torch.unbind(indices,dim=1)],regularization
        else:
            preds = self.bmm_collate(preds_list=preds_list)
            return preds,regularization

    def full_grad_debug(self,indices):
        pred_outputs = []
        reg_output = 0
        for i, v in self.ii.items():
            ix = indices[:, v]
            tt = self.TT_cores[str(i)]
            tt_prime = self.TT_cores_prime[str(i)]
            tt.full_grad=True
            tt_prime.full_grad=True
            prime_pred, reg_prime = tt_prime(ix)
            pred, reg = tt(ix)
            pred_outputs.append(pred * prime_pred)
            if self.config['dual']:
                reg_output += torch.sum(
                    reg.float() * reg_prime.float())  # numerical issue with fp 16 how fix, sum of square terms, serves as fp 16 fix
            else:
                reg_output += torch.mean(reg.float()) + torch.mean(reg_prime.float())  # numerical issue with fp 16 how fix,

        return pred_outputs, reg_output * self.lambda_reg



class KFT_scale(torch.nn.Module):
    def __init__(self, initialization_data,lambdas,  cuda=None, config=None, old_setup=False): #decomposition_data = {0:{'ii':[0,1],'lambda':0.01,r_1:1 n_list=[10,10],r_2:10,'has_side_info':True, side_info:{1:x_1,2:x_2},kernel_para:{'ls_factor':0.5, 'kernel_type':'RBF','nu':2.5} },1:{}}
        super(KFT_scale, self).__init__()
        self.kernel_class_name = ['TT_kernel_component']
        self.cuda = cuda
        self.config = config
        self.amp = None
        self.old_setup = old_setup
        tmp_dict = {}
        tmp_dict_s = {}
        tmp_dict_b = {}
        self.full_grad = config['full_grad']
        self.ii = {}
        for i,v in initialization_data.items():
            self.ii[i] = v['ii']

            tmp_dict_b[str(i)] = TT_component(r_1=v['r_1_latent'],
                                              n_list=v['n_list'],
                                              r_2=v['r_2_latent'],
                                              cuda=cuda,
                                              config=config,
                                              init_scale=v['init_scale'],
                                              reg_para=lambdas[f'reg_para_b_{i}'])
            tmp_dict_s[str(i)] = TT_component(r_1=v['r_1_latent'],
                                              n_list=v['n_list'],
                                              r_2=v['r_2_latent'],
                                              cuda=cuda,
                                              config=config,
                                              init_scale=v['init_scale'],
                                              reg_para=lambdas[f'reg_para_s_{i}'])
            if v['has_side_info']:
                tmp_dict[str(i)] = TT_kernel_component(r_1=v['r_1'],
                                                       n_list=v['n_list'] if config['dual'] else v['primal_list'],
                                                       r_2=v['r_2'],
                                                       side_information_dict=v['side_info'],
                                                       kernel_para_dict=v['kernel_para'],
                                                       cuda=cuda,
                                                       config=config,
                                                       init_scale=v['init_scale'],
                                                       reg_para=lambdas[f'reg_para_{i}'])
            else:
                tmp_dict[str(i)] = TT_component(r_1=v['r_1'],n_list=v['n_list'] if config['dual'] else v['primal_list'],r_2=v['r_2'],cuda=cuda,config=config,init_scale=v['init_scale'],old_setup=old_setup)
        self.TT_cores = torch.nn.ModuleDict(tmp_dict)
        self.TT_cores_s = torch.nn.ModuleDict(tmp_dict_s)
        self.TT_cores_b = torch.nn.ModuleDict(tmp_dict_b)

    def turn_on_V(self):
        for i, v in self.ii.items():
            if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
                self.TT_cores[str(i)].kernel_train_mode_off()
                if self.TT_cores[str(i)].deep_kernel:
                    self.TT_cores[str(i)].deep_kernel_mode_off()
            else:
                self.TT_cores[str(i)].turn_on()
            self.TT_cores_s[str(i)].turn_off()
            self.TT_cores_b[str(i)].turn_off()
        return 0

    def amp_patch(self,amp):
        for i,v in self.ii.items():
            self.TT_cores[str(i)].amp=amp
            self.TT_cores_s[str(i)].amp=amp
            self.TT_cores_b[str(i)].amp=amp

    def turn_on_prime(self):
        for i, v in self.ii.items():
            if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
                self.TT_cores[str(i)].kernel_train_mode_off()
                self.TT_cores[str(i)].turn_off()
                if self.TT_cores[str(i)].deep_kernel:
                    self.TT_cores[str(i)].deep_kernel_mode_off()
            else:
                self.TT_cores[str(i)].turn_off()
            self.TT_cores_s[str(i)].turn_on()
            self.TT_cores_b[str(i)].turn_on()

        return 0
    #TODO: fix deep kernel by properly activating and turning of parameters
    def turn_on_deep_kernel(self):
        for i, v in self.ii.items():
            if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
                self.TT_cores[str(i)].kernel_train_mode_off()
                self.TT_cores[str(i)].turn_off()
                if self.TT_cores[str(i)].deep_kernel:
                    self.TT_cores[str(i)].deep_kernel_mode_on()
            else:
                self.TT_cores[str(i)].turn_off()
            self.TT_cores_s[str(i)].turn_off()
            self.TT_cores_b[str(i)].turn_off()
        return 0

    def has_kernel_component(self):
        for i, v in self.ii.items():
            if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
                for v in self.TT_cores[str(i)].n_dict.values():
                    if v is not None:
                        if self.TT_cores[str(i)].deep_kernel:
                            return True,True
                        else:
                            return True,False
        return False,False

    def turn_on_kernel_mode(self):
        for i,v in self.ii.items():
            if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
                self.TT_cores[str(i)].kernel_train_mode_on()
                if self.TT_cores[str(i)].deep_kernel:
                    self.TT_cores[str(i)].deep_kernel_mode_off()
            else:
                self.TT_cores[str(i)].turn_off()
            self.TT_cores_s[str(i)].turn_off()
            self.TT_cores_b[str(i)].turn_off()
        return 0

    def turn_off_kernel_mode(self):
        for i,v in self.ii.items():
            if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
                self.TT_cores[str(i)].kernel_train_mode_off()
            else:
                self.TT_cores[str(i)].turn_on()
            self.TT_cores_s[str(i)].turn_on()
            self.TT_cores_b[str(i)].turn_on()
        return 0

    def bmm_collate(self, preds_list):
        preds = preds_list[0]
        for i in range(1, len(preds_list)):
            preds = torch.bmm(preds, preds_list[i])
        return preds.squeeze()

    def edge_mode_collate(self, preds_list):
        preds = preds_list[0]
        for i in range(1, len(preds_list)):
            preds = edge_mode_product(preds, preds_list[i], len(preds.shape) - 1, 0)  # General mode product!
        return preds.squeeze()

    def collect_core_outputs(self,indices):
        scale = []
        regression = []
        bias = []
        reg_output=0
        for i,v in self.ii.items():
            ix = indices[:,v]
            tt = self.TT_cores[str(i)]
            tt_s = self.TT_cores_s[str(i)]
            tt_b = self.TT_cores_b[str(i)]
            prime_s,reg_s = tt_s.forward_scale(ix)
            prime_b,reg_b = tt_b.forward_scale(ix)
            pred, reg = tt(ix)
            reg_output += tt.reg_para*torch.mean(reg).abs()+tt_s.reg_para*torch.mean(reg_s)+tt_b.reg_para*torch.mean(reg_b) #numerical issue with fp 16 how fix, sum of square terms, serves as fp 16 fix
            scale.append(prime_s)
            bias.append(prime_b)
            regression.append(pred)

        if self.full_grad:
            group_func = self.edge_mode_collate
        else:
            group_func = self.bmm_collate
        s = group_func(scale)
        r = group_func(regression)
        b = group_func(bias)

        pred = s*r+b
        if self.full_grad:
            return pred[torch.unbind(indices, dim=1)], reg_output
        else:
            return pred,reg_output

    def forward(self,indices):
        pred, reg = self.collect_core_outputs(indices)
        return pred,reg

class variational_KFT(KFT):
    def __init__(self,initialization_data,KL_weight,cuda=None,config=None,old_setup=False):
        super(variational_KFT, self).__init__(initialization_data, lambda_reg=KL_weight, cuda=cuda, config=config, old_setup=old_setup)
        tmp_dict = {}
        tmp_dict_prime = {}
        self.kernel_class_name = ['multivariate_variational_kernel_TT','univariate_variational_kernel_TT']
        self.full_grad = config['full_grad']
        self.ii = {}
        self.KL_weight = torch.nn.Parameter(torch.tensor(KL_weight),requires_grad=False)
        for i, v in initialization_data.items():
            self.ii[i] = v['ii']
            tmp_dict_prime[str(i)] = variational_TT_component(r_1=v['r_1'],
                                                              n_list=v['n_list'],
                                                              r_2=v['r_2'],
                                                              cuda=cuda,
                                                              config=config,
                                                              init_scale=v['init_scale'])
            if v['has_side_info']:
                if v['multivariate'] and config['dual']:
                    tmp_dict[str(i)] = multivariate_variational_kernel_TT(r_1=v['r_1'],
                                                                          n_list=v['n_list'] if config['dual'] else v[
                                                                              'primal_list'],
                                                                          r_2=v['r_2'],
                                                                          side_information_dict=v['side_info'],
                                                                          kernel_para_dict=v['kernel_para'],
                                                                          cuda=cuda,
                                                                          config=config,
                                                                          init_scale=v['init_scale'])
                else:
                    tmp_dict[str(i)] = univariate_variational_kernel_TT(r_1=v['r_1'],
                                                                        n_list=v['n_list'] if config['dual'] else v[
                                                                            'primal_list'],
                                                                        r_2=v['r_2'],
                                                                        side_information_dict=v['side_info'],
                                                                        kernel_para_dict=v['kernel_para'],
                                                                        cuda=cuda,
                                                                        config=config,
                                                                        init_scale=v['init_scale'])
            else:
                tmp_dict[str(i)] = variational_TT_component(r_1=v['r_1'],
                                                            n_list=v['n_list'] if config['dual'] else v['primal_list'],
                                                            r_2=v['r_2'],
                                                            cuda=cuda,
                                                            config=config,
                                                            init_scale=v['init_scale'])

        self.TT_cores = torch.nn.ModuleDict(tmp_dict)
        self.TT_cores_prime = torch.nn.ModuleDict(tmp_dict_prime)

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

    def collect_core_outputs(self,indices):
        first_term = []
        second_term = []
        third_term = []
        total_KL = 0
        for i, v in self.ii.items():
            ix = indices[:, v]
            tt = self.TT_cores[str(i)]
            tt_prime = self.TT_cores_prime[str(i)]
            V_prime, var_prime, KL_prime = tt_prime(ix)
            base, extra, KL = tt(ix)
            first_term.append(V_prime*base)
            second_term.append((extra+base*base)*var_prime)
            third_term.append((V_prime*V_prime)*extra)
            total_KL += KL.abs() + KL_prime
        if self.full_grad:
            group_func = self.edge_mode_collate
        else:
            group_func = self.bmm_collate
        middle = group_func(first_term)
        last_term = middle**2
        for i,preds_list in enumerate([second_term,third_term]):
            tmp = group_func(preds_list)
            last_term +=  tmp
        if self.full_grad:
            middle = middle[torch.unbind(indices, dim=1)]
            last_term = last_term[torch.unbind(indices, dim=1)]
        return middle,last_term, total_KL * self.KL_weight

    def collect_core_outputs_reparametrization(self, indices):
        pred_outputs = []
        total_KL=0
        for i,v in self.ii.items():
            ix = indices[:,v]
            tt = self.TT_cores[str(i)]
            tt_prime = self.TT_cores_prime[str(i)]
            prime_pred,KL_prime = tt_prime(ix)
            pred, KL = tt(ix)
            pred_outputs.append(pred*prime_pred)
            total_KL += KL.abs() + KL_prime.abs()
        return pred_outputs,total_KL*self.KL_weight

    def forward_reparametrization(self, indices):
        preds_list, regularization = self.collect_core_outputs_reparametrization(indices)
        if self.full_grad:
            preds = self.edge_mode_collate(preds_list)
            return preds[torch.unbind(indices, dim=1)], regularization
        else:
            preds = self.bmm_collate(preds_list)
            return preds, regularization

    def forward(self, indices):
        middle,third, regularization = self.collect_core_outputs(indices)
        return middle,third,regularization

    def mean_forward(self,indices):
        preds_list = self.collect_core_outputs_mean(indices)
        if self.full_grad:
            preds = self.edge_mode_collate(preds_list)
            return preds[torch.unbind(indices, dim=1)]
        else:
            preds = self.bmm_collate(preds_list)
            return preds


class varitional_KFT_scale(KFT_scale):
    def __init__(self,initialization_data,KL_weight,cuda=None,config=None,old_setup=False):
        super(varitional_KFT_scale, self).__init__(initialization_data,KL_weight,cuda,config,old_setup)
        tmp_dict = {}
        tmp_dict_s = {}
        tmp_dict_b = {}
        self.kernel_class_name = ['multivariate_variational_kernel_TT', 'univariate_variational_kernel_TT']
        self.full_grad = config['full_grad']
        self.ii = {}
        self.KL_weight = torch.nn.Parameter(torch.tensor(KL_weight), requires_grad=False)
        for i, v in initialization_data.items():
            self.ii[i] = v['ii']
            tmp_dict_s[str(i)] = variational_TT_component(r_1=v['r_1_latent'],
                                                              n_list=v['n_list'],
                                                              r_2=v['r_2_latent'],
                                                              cuda=cuda,
                                                              config=config)
            tmp_dict_b[str(i)] = variational_TT_component(r_1=v['r_1_latent'],
                                                          n_list=v['n_list'],
                                                          r_2=v['r_2_latent'],
                                                          cuda=cuda,
                                                          config=config)
            if v['has_side_info']:
                if v['multivariate'] and config['dual']:
                    tmp_dict[str(i)] = multivariate_variational_kernel_TT(r_1=v['r_1'],
                                                                          n_list=v['n_list'] if config['dual'] else v['primal_list'],
                                                                          r_2=v['r_2'],
                                                                          side_information_dict=v['side_info'],
                                                                          kernel_para_dict=v['kernel_para'],
                                                                          cuda=cuda,
                                                                          config=config,
                                                                          init_scale=1.0)
                else:
                    tmp_dict[str(i)] = univariate_variational_kernel_TT(r_1=v['r_1'],
                                                                        n_list=v['n_list'] if config['dual'] else v['primal_list'],
                                                                        r_2=v['r_2'],
                                                                        side_information_dict=v['side_info'],
                                                                        kernel_para_dict=v['kernel_para'],
                                                                        cuda=cuda,
                                                                        config=config,
                                                                        init_scale=1.0)
            else:
                tmp_dict[str(i)] = variational_TT_component(r_1=v['r_1'],
                                                            n_list=v['n_list'] if config['dual'] else v['primal_list'],
                                                            r_2=v['r_2'],
                                                            cuda=cuda,
                                                            config=config,
                                                            init_scale=1.0)
        self.TT_cores = torch.nn.ModuleDict(tmp_dict)
        self.TT_cores_s = torch.nn.ModuleDict(tmp_dict_s)
        self.TT_cores_b = torch.nn.ModuleDict(tmp_dict_b)

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
            total_KL+= KL_s+KL_b+KL

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
            return middle[torch.unbind(indices, dim=1)], third_term[torch.unbind(indices, dim=1)], total_KL * self.KL_weight
        else:
            return middle, third_term, total_KL * self.KL_weight

    def forward(self,indices):
        middle_term,third_term,reg = self.collect_core_outputs(indices)
        return middle_term,third_term,reg