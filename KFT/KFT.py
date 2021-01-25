import torch
import math
import time
from KFT.core_components import TT_component,TT_kernel_component,edge_mode_product,KFTR_temporal_regulizer
from KFT.core_VI_components import univariate_variational_kernel_TT,multivariate_variational_kernel_TT,variational_TT_component
PI  = math.pi
torch.set_printoptions(profile="full")

class KFT(torch.nn.Module):
    def __init__(self, initialization_data,lambdas,shape_permutation, cuda=None, config=None, old_setup=False): #decomposition_data = {0:{'ii':[0,1],'lambda':0.01,r_1:1 n_list=[10,10],r_2:10,'has_side_info':True, side_info:{1:x_1,2:x_2},kernel_para:{'ls_factor':0.5, 'kernel_type':'RBF','nu':2.5} },1:{}}
        super(KFT, self).__init__()
        self.kernel_class_name = ['TT_kernel_component']
        self.cuda = cuda
        self.config = config
        self.old_setup = old_setup
        self.shape_permutation = torch.tensor(shape_permutation).long()
        tmp_dict = {}
        tmp_dict_prime = {}
        self.full_grad = config['full_grad']
        self.ii = {}
        self.current_update_pointers = []
        for i,v in initialization_data.items():
            self.ii[i] = v['ii']
            if not self.old_setup:
                tmp_dict_prime[str(i)] = TT_component(r_1=v['r_1'],
                                                          n_list=v['n_list'],
                                                          r_2=v['r_2'],
                                                          cuda=cuda,
                                                          config=config,
                                                          init_scale=v['init_scale'],
                                                          reg_para=lambdas[f'reg_para_prime_{i}'],
                                                          double_factor=v['double_factor'], sub_R=config['sub_R'])
                    #TODO: "Mixed prime factorizations..."
            if v['has_side_info']:
                tmp_dict[str(i)] = TT_kernel_component(r_1=v['r_1'],
                                                       n_list= v['n_list'],
                                                       r_2=v['r_2'],
                                                       side_information_dict=v['side_info'],
                                                       kernel_para_dict=v['kernel_para'],
                                                       cuda=cuda,
                                                       config=config,
                                                       init_scale=v['init_scale'],
                                                       reg_para=lambdas[f'reg_para_{i}'])
            else:
                tmp_dict[str(i)] = TT_component(r_1=v['r_1'],
                                                n_list=  v['n_list'],
                                                r_2=v['r_2'],
                                                cuda=cuda,
                                                config=config,
                                                init_scale=v['init_scale'],
                                                old_setup=old_setup,
                                                reg_para=lambdas[f'reg_para_{i}'])
        self.TT_cores = torch.nn.ModuleDict(tmp_dict)
        self.TT_cores_prime = torch.nn.ModuleDict(tmp_dict_prime)

    def turn_on_all(self):
        self.current_update_pointers = []
        for i, v in self.ii.items():
            self.current_update_pointers.append(i)
            if not self.old_setup:
                self.TT_cores_prime[str(i)].turn_on()
                self.TT_cores_prime[str(i)].cache_mode = False
            self.TT_cores[str(i)].turn_on()
            if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
                self.TT_cores[str(i)].kernel_train_mode_on()
            self.TT_cores[str(i)].cache_mode = False

    def turn_off_all(self):
        self.current_update_pointers = []
        for i, v in self.ii.items():
            if not self.old_setup:
                self.TT_cores_prime[str(i)].turn_off()
                self.TT_cores_prime[str(i)].cache_results()
                self.TT_cores_prime[str(i)].cache_mode = True
            self.TT_cores[str(i)].turn_off()
            if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
                self.TT_cores[str(i)].kernel_train_mode_off()
            self.TT_cores[str(i)].cache_results()
            self.TT_cores[str(i)].cache_mode = True


    def turn_on_all_V(self,placeholder=None):
        self.turn_off_all()
        for i, v in self.ii.items():
            self.current_update_pointers.append(i)
            if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
                self.TT_cores[str(i)].kernel_train_mode_off()
                self.TT_cores[str(i)].turn_on()
            else:
                self.TT_cores[str(i)].turn_on()
            self.TT_cores[str(i)].cache_mode = False
            if not self.old_setup:
                self.TT_cores_prime[str(i)].turn_off()

    def turn_on_all_prime(self,placeholder=None):
        self.turn_off_all()
        for i, v in self.ii.items():
            self.current_update_pointers.append(i)
            if not self.old_setup:
                if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
                    self.TT_cores[str(i)].kernel_train_mode_off()
                    self.TT_cores[str(i)].turn_off()
                else:
                    self.TT_cores[str(i)].turn_off()
                self.TT_cores_prime[str(i)].turn_on()
                self.TT_cores_prime[str(i)].cache_mode = False

    def turn_on_all_kernels(self,placeholder=None):
        self.turn_off_all()
        for i, v in self.ii.items():
            self.current_update_pointers.append(i)
            if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
                self.TT_cores[str(i)].kernel_train_mode_on()
                self.TT_cores[str(i)].turn_off()
            else:
                self.TT_cores[str(i)].turn_off()
            if not self.old_setup:
                self.TT_cores_prime[str(i)].turn_off()
            self.TT_cores[str(i)].cache_mode = False

    def turn_on_V(self,i):
        self.turn_off_all()
        self.current_update_pointers.append(i)
        if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
            self.TT_cores[str(i)].kernel_train_mode_off()
            self.TT_cores[str(i)].turn_on()
        else:
            self.TT_cores[str(i)].turn_on()
        self.TT_cores[str(i)].cache_mode = False
        if not self.old_setup:
            self.TT_cores_prime[str(i)].turn_off()

    def turn_on_prime(self,i):
        self.turn_off_all()
        self.current_update_pointers.append(i)
        if not self.old_setup:
            if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
                self.TT_cores[str(i)].kernel_train_mode_off()
                self.TT_cores[str(i)].turn_off()
            else:
                self.TT_cores[str(i)].turn_off()
            self.TT_cores_prime[str(i)].turn_on()
            self.TT_cores_prime[str(i)].cache_mode = False

    def has_dual_kernel_component(self,i):
        if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name and self.TT_cores[str(i)].dual:
            return True

    def turn_on_kernel_mode(self,i):
        self.turn_off_all()
        self.current_update_pointers.append(i)
        if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
            self.TT_cores[str(i)].kernel_train_mode_on()
            self.TT_cores[str(i)].turn_off()
        else:
            self.TT_cores[str(i)].turn_off()
        if not self.old_setup:
            self.TT_cores_prime[str(i)].turn_off()
        self.TT_cores[str(i)].cache_mode = False

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
                pred_outputs.append(pred * prime_pred)
            else:
                pred, reg = tt(ix)
                pred_outputs.append(pred)
            if self.config['dual']:
                if not self.old_setup:
                    reg_output += tt.reg_para * torch.mean(reg*reg_prime) #numerical issue with fp 16 how fix, sum of square terms, serves as fp 16 fix
                else:
                    reg_output += torch.mean(reg) * tt.reg_para#numerical issue with fp 16 how fix, sum of square terms, serves as fp 16 fix
            else:
                if not self.old_setup:
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
        indices = indices[:,self.shape_permutation]
        preds_list,regularization = self.collect_core_outputs(indices)
        if self.full_grad:
            preds = self.edge_mode_collate(preds_list=preds_list)
            return preds[torch.unbind(indices,dim=1)],regularization
        else:
            preds = self.bmm_collate(preds_list=preds_list)
            return preds,regularization

class KFT_scale(torch.nn.Module):
    def __init__(self, initialization_data,lambdas,shape_permutation,  cuda=None, config=None, old_setup=False): #decomposition_data = {0:{'ii':[0,1],'lambda':0.01,r_1:1 n_list=[10,10],r_2:10,'has_side_info':True, side_info:{1:x_1,2:x_2},kernel_para:{'ls_factor':0.5, 'kernel_type':'RBF','nu':2.5} },1:{}}
        super(KFT_scale, self).__init__()
        self.shape_permutation = shape_permutation
        self.kernel_class_name = ['TT_kernel_component']
        self.cuda = cuda
        self.config = config
        self.old_setup = old_setup
        tmp_dict = {}
        tmp_dict_s = {}
        tmp_dict_b = {}
        self.full_grad = config['full_grad']
        self.ii = {}
        self.current_update_pointers = []
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
                                                       n_list= v['n_list'],
                                                       r_2=v['r_2'],
                                                       side_information_dict=v['side_info'],
                                                       kernel_para_dict=v['kernel_para'],
                                                       cuda=cuda,
                                                       config=config,
                                                       init_scale=v['init_scale'],
                                                       reg_para=lambdas[f'reg_para_{i}'])
            else:
                tmp_dict[str(i)] = TT_component(r_1=v['r_1'],n_list= v['n_list'],r_2=v['r_2'],cuda=cuda,config=config,init_scale=v['init_scale'],old_setup=old_setup)
        self.TT_cores = torch.nn.ModuleDict(tmp_dict)
        self.TT_cores_s = torch.nn.ModuleDict(tmp_dict_s)
        self.TT_cores_b = torch.nn.ModuleDict(tmp_dict_b)

    def has_dual_kernel_component(self,i):
        if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name and self.TT_cores[str(i)].dual:
            return True

    def turn_on_all(self):
        for i, v in self.ii.items():
            if not self.old_setup:
                self.TT_cores_b[str(i)].turn_on()
                self.TT_cores_s[str(i)].turn_on()
            self.TT_cores[str(i)].turn_on()
            if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
                self.TT_cores[str(i)].kernel_train_mode_on()
            self.TT_cores_s[str(i)].cache_mode = False
            self.TT_cores_b[str(i)].cache_mode = False
            self.TT_cores[str(i)].cache_mode = False

    def turn_off_all(self):
        for i, v in self.ii.items():
            if not self.old_setup:
                self.TT_cores_b[str(i)].turn_off()
                self.TT_cores_s[str(i)].turn_off()
            self.TT_cores[str(i)].turn_off()
            if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
                self.TT_cores[str(i)].kernel_train_mode_off()
            self.TT_cores_s[str(i)].cache_results()
            self.TT_cores_b[str(i)].cache_results()
            self.TT_cores[str(i)].cache_results()
            self.TT_cores_s[str(i)].cache_mode = True
            self.TT_cores_b[str(i)].cache_mode = True
            self.TT_cores[str(i)].cache_mode = True

    def turn_on_V(self,i):
        self.turn_off_all()
        self.current_update_pointers.append(i)
        if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
            self.TT_cores[str(i)].kernel_train_mode_off()
            self.TT_cores[str(i)].turn_on()
        else:
            self.TT_cores[str(i)].turn_on()
        self.TT_cores_s[str(i)].turn_off()
        self.TT_cores_b[str(i)].turn_off()
        self.TT_cores[str(i)].cache_mode=False

    def turn_on_prime(self,i):
        self.turn_off_all()
        self.current_update_pointers.append(i)
        if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
            self.TT_cores[str(i)].kernel_train_mode_off()
            self.TT_cores[str(i)].turn_off()
        else:
            self.TT_cores[str(i)].turn_off()
        self.TT_cores_s[str(i)].turn_on()
        self.TT_cores_b[str(i)].turn_on()
        self.TT_cores_s[str(i)].cache_mode = False
        self.TT_cores_b[str(i)].cache_mode = False

    def has_kernel_component(self):
        for i, v in self.ii.items():
            if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
                for v in self.TT_cores[str(i)].n_dict.values():
                    if v is not None:
                        return True
        return False

    def turn_on_kernel_mode(self,i):
        self.turn_off_all()
        self.current_update_pointers.append(i)
        if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
            self.TT_cores[str(i)].kernel_train_mode_on()
            self.TT_cores[str(i)].turn_off()
        else:
            self.TT_cores[str(i)].turn_off()
        self.TT_cores_s[str(i)].turn_off()
        self.TT_cores_b[str(i)].turn_off()
        self.TT_cores[str(i)].cache_mode = False

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
            reg_output += tt.reg_para*torch.mean(reg)+tt_s.reg_para*torch.mean(reg_s)+tt_b.reg_para*torch.mean(reg_b) #numerical issue with fp 16 how fix, sum of square terms, serves as fp 16 fix
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
        indices = indices[:,self.shape_permutation]
        pred, reg = self.collect_core_outputs(indices)
        return pred,reg
