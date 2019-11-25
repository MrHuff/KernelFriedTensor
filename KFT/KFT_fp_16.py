import torch
import tensorly
tensorly.set_backend('pytorch')
import gpytorch
from tensorly.base import fold,unfold,partial_fold
from tensorly.tenalg import multi_mode_dot,mode_dot
from KFT.FLOWS.flows import IAF_no_h
import math
import timeit
from apex import amp
import apex
PI  = math.pi
torch.set_printoptions(profile="full")
def row_outer_prod(x,y):
    """
    :param x: n x c
    :param y: n x d
    :return: n x (cd)
    """
    y = y.unsqueeze(-1).expand(-1,-1,y.shape[1]).permute(0,2,1)
    return torch.einsum('ij,ijk->ijk', x, y).flatten(1)

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
        self.n_feat = int(round(math.log(X.shape[0])))#Michaels paper!

        print(f'I= {self.n_feat}')
        self.raw_lengthscale = torch.nn.Parameter(lengtscale,requires_grad=False)
        self.register_buffer('w' ,torch.randn(*(self.n_feat,self.n_input_feat)))
        self.register_buffer('b' ,torch.rand(*(self.n_feat, 1),device=device)*2.0*PI)
    def forward(self,X,dum_2=None):
        return torch.transpose(math.sqrt(2./float(self.n_feat))*torch.cos(torch.mm(self.w/self.raw_lengthscale, X.t()) + self.b),0,1)

class TT_component(torch.nn.Module):
    def __init__(self,r_1,n_list,r_2,cuda=None,config=None,init_scale=1.0,old_setup=False):
        super(TT_component, self).__init__()
        self.n_list = n_list
        self.amp = None
        self.V_mode = True
        self.device = cuda
        self.old_setup = old_setup
        self.full_grad = config['full_grad']
        self.n_dict = {i + 1: None for i in range(len(n_list))}
        self.RFF_dict = {i + 1: False for i in range(len(n_list))}
        self.shape_list  = [r_1]+[n for n in n_list] + [r_2]
        self.permutation_list = [i + 1 for i in range(len(n_list))] + [0, -1]
        self.core_param = torch.nn.Parameter(init_scale*torch.ones(*self.shape_list), requires_grad=not self.old_setup)
        self.init_scale = init_scale
        self.numel = self.core_param.numel()
        for i, n in enumerate(n_list):
            self.register_buffer(f'reg_ones_{i}',torch.ones((n,1)))

    def index_select(self,indices,T):
        return T.permute(self.permutation_list)[indices]

    def turn_off(self):
        for p in self.parameters():
            p.requires_grad = False
        self.V_mode=False

    def turn_on(self):
        for p in self.parameters():
            p.requires_grad = True
        self.V_mode=True

    def forward(self,indices):
        if self.full_grad:
            return self.core_param, self.get_aux_reg_term()
        else:
            if len(indices.shape)>1:
                indices = indices.unbind(1)
            return self.core_param.permute(self.permutation_list)[indices], self.get_aux_reg_term()

    def forward_scale(self,indices):
        if self.full_grad:
            return self.core_param, self.core_param**2
        else:
            if len(indices.shape)>1:
                indices = indices.unbind(1)
            return self.core_param.permute(self.permutation_list)[indices], self.core_param**2

    def get_aux_reg_term(self):
        if self.old_setup:
            return 1.
        else:
            T = (self.core_param ** 2)/self.numel
            for mode,ones in enumerate(self.n_list):
                ones = getattr(self,f'reg_ones_{mode}')
                T = lazy_mode_product(T, ones.t(), mode+1)
                T = lazy_mode_product(T, ones, mode+1)
            return T

class TT_kernel_component(TT_component): #for tensors with full or "mixed" side info
    def __init__(self,r_1,n_list,r_2,side_information_dict,kernel_para_dict,cuda=None,config=None,init_scale=1.0):
        super(TT_kernel_component, self).__init__(r_1,n_list,r_2,cuda,config,init_scale)
        self.core_param = torch.nn.Parameter(init_scale*torch.randn(*self.shape_list), requires_grad=True)
        self.deep_kernel = config['deep_kernel']
        self.deep_mode = False
        for key,value in side_information_dict.items(): #Should be on the form {mode: side_info}'
            self.assign_kernel(key,value,kernel_para_dict,config['deep_kernel'])

    def kernel_train_mode_on(self):
        self.turn_off()
        for key,val in self.n_dict.items():
            if val is not None:
                k = getattr(self,f'kernel_{key}')
                k.raw_lengthscale.requires_grad = True
                if k.__class__.__name__=='PeriodicKernel':
                    k.raw_period_length.requires_grad = True
        return 0

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
                    if  k.__class__.__name__=='RFF':
                        self.n_dict[key] = k(value)
                    else:
                        self.n_dict[key] = k(value).evaluate()
        return 0

    def deep_kernel_mode_on(self):
        self.deep_mode = True
        for key in self.n_dict.keys():
            f = getattr(self, f'transformation_{key}')
            for p in f.parameters():
                p.requires_grad = True
        return 0

    def deep_kernel_mode_off(self):
        self.deep_mode = False
        for key in self.n_dict.keys():
            f = getattr(self, f'transformation_{key}')
            for p in f.parameters():
                p.requires_grad = False
            for key,val in self.n_dict.items(): #Issue is probably here!
                k = getattr(self,f'kernel_{key}')
                f = getattr(self, f'transformation_{key}')
                input = getattr(self,f'kernel_data_{key}')
                with torch.no_grad():
                    X = f(input)
                    if  k.__class__.__name__=='RFF':
                        self.n_dict[key] = k(X)
                    else:
                        self.n_dict[key] = k(X).evaluate()
        return 0

    def get_median_ls(self,X,key):  # Super LS and init value sensitive wtf
        base = gpytorch.kernels.Kernel()
        if X.shape[0] > 5000:
            self.RFF_dict[key] = True
            idx = torch.randperm(5000)
            X = X[idx, :]
        d = base.covar_dist(X, X)
        return torch.sqrt(torch.median(d[d > 0])).unsqueeze(0)

    def assign_kernel(self,key,value,kernel_dict_input,deep_kernel=False):
        kernel_para_dict = kernel_dict_input[key]
        gwidth0 = self.get_median_ls(value,key)
        self.gamma_sq_init = gwidth0 * kernel_para_dict['ls_factor']
        ard_dims = None if not kernel_para_dict['ARD'] else value.shape[1]
        if self.RFF_dict[key]:
            setattr(self, f'kernel_{key}', RFF(value,lengtscale=self.gamma_sq_init))
            # value = value.to(self.device)
        else:
            if kernel_para_dict['kernel_type']=='rbf':
                setattr(self, f'kernel_{key}', gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims))
                getattr(self, f'kernel_{key}').raw_lengthscale = torch.nn.Parameter(
                    self.gamma_sq_init * torch.ones(*(1, 1 if ard_dims is None else ard_dims)),
                    requires_grad=False)

            elif kernel_para_dict['kernel_type']=='matern':
                setattr(self, f'kernel_{key}', gpytorch.kernels.MaternKernel(ard_num_dims=ard_dims,nu=kernel_para_dict['nu']))
                getattr(self, f'kernel_{key}').raw_lengthscale = torch.nn.Parameter(
                    self.gamma_sq_init * torch.ones(*(1, 1 if ard_dims is None else ard_dims)),
                    requires_grad=False)
            elif kernel_para_dict['kernel_type']=='periodic':
                setattr(self, f'kernel_{key}', gpytorch.kernels.PeriodicKernel())
                getattr(self, f'kernel_{key}').raw_lengthscale = torch.nn.Parameter(
                    self.gamma_sq_init * torch.ones(*(1, 1)),
                    requires_grad=False)
                getattr(self,f'kernel_{key}').raw_period_length = torch.nn.Parameter(kernel_para_dict['p']*torch.ones(*(1,1)),requires_grad=False)

        tmp_kernel_func = getattr(self,f'kernel_{key}')
        if tmp_kernel_func.__class__.__name__ in 'RFF':
            self.n_dict[key] =  tmp_kernel_func(value).to(self.device)
        else:
            self.n_dict[key] =  tmp_kernel_func(value).evaluate().to(self.device)
        self.register_buffer(f'kernel_data_{key}',value)
        if deep_kernel:
            setattr(self,f'transformation_{key}',IAF_no_h(latent_size=value.shape[1],depth=2,tanh_flag_h=True,C=10))

    def side_data_eval(self,key):
        X = getattr(self, f'kernel_data_{key}')
        tmp_kernel_func = getattr(self, f'kernel_{key}')
        if self.deep_mode:
            f = getattr(self, f'transformation_{key}')
            X = f(X)
        if tmp_kernel_func.__class__.__name__=='RFF':
            val = tmp_kernel_func(X)
        else:
            val = tmp_kernel_func(X).evaluate()
        return val
    def apply_kernels(self,T):
        for key, val in self.n_dict.items():
            if val is not None:
                if not self.V_mode:
                    val = self.side_data_eval(key)
                if not self.RFF_dict[key]:
                    T = lazy_mode_product(T, val, key)
                else:
                    T = lazy_mode_product(T, val.t(), key)
                    T = lazy_mode_product(T, val, key)
        return T

    def forward(self,indices):
        """Do tensor ops"""
        T = self.apply_kernels(self.core_param)
        if self.full_grad:
            return T,T*self.core_param
        else:
            if len(indices.shape)>1:
                indices = indices.unbind(1)
            return T.permute(self.permutation_list)[indices], T*self.core_param  #return both to calculate regularization when doing frequentist

class KFT(torch.nn.Module):
    def __init__(self, initialization_data, lambda_reg=1e-6, cuda=None, config=None, old_setup=False): #decomposition_data = {0:{'ii':[0,1],'lambda':0.01,r_1:1 n_list=[10,10],r_2:10,'has_side_info':True, side_info:{1:x_1,2:x_2},kernel_para:{'ls_factor':0.5, 'kernel_type':'RBF','nu':2.5} },1:{}}
        super(KFT, self).__init__()
        self.kernel_class_name = ['TT_kernel_component']
        self.cuda = cuda
        self.amp = None
        self.config = config
        self.old_setup = old_setup
        self.register_buffer('lambda_reg',torch.tensor(lambda_reg))
        tmp_dict = {}
        tmp_dict_prime = {}
        self.full_grad = config['full_grad']
        self.ii = {}
        for i,v in initialization_data.items():
            self.ii[i] = v['ii']
            tmp_dict_prime[str(i)] = TT_component(r_1=v['r_1'],n_list=v['n_list'],r_2=v['r_2'],cuda=cuda,config=config,init_scale=v['init_scale'])
            if v['has_side_info']:
                tmp_dict[str(i)] = TT_kernel_component(r_1=v['r_1'],
                                                       n_list=v['n_list'],
                                                       r_2=v['r_2'],
                                                       side_information_dict=v['side_info'],
                                                       kernel_para_dict=v['kernel_para'],cuda=cuda,config=config,init_scale=v['init_scale'])
            else:
                tmp_dict[str(i)] = TT_component(r_1=v['r_1'],n_list=v['n_list'],r_2=v['r_2'],cuda=cuda,config=config,init_scale=v['init_scale'],old_setup=old_setup)
        self.TT_cores = torch.nn.ModuleDict(tmp_dict)
        self.TT_cores_prime = torch.nn.ModuleDict(tmp_dict_prime)

    def turn_on_V(self):
        for i, v in self.ii.items():
            if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
                self.TT_cores[str(i)].kernel_train_mode_off()
                if self.TT_cores[str(i)].deep_kernel:
                    self.TT_cores[str(i)].deep_kernel_mode_off()
            else:
                self.TT_cores[str(i)].turn_on()
            self.TT_cores_prime[str(i)].turn_off()
        return 0

    def turn_on_prime(self):
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
            self.TT_cores_prime[str(i)].turn_off()
        return 0

    def turn_off_kernel_mode(self):
        for i,v in self.ii.items():
            if self.TT_cores[str(i)].__class__.__name__ in self.kernel_class_name:
                self.TT_cores[str(i)].kernel_train_mode_off()
            else:
                self.TT_cores[str(i)].turn_on()
            self.TT_cores_prime[str(i)].turn_on()
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
            tt_prime = self.TT_cores_prime[str(i)]
            prime_pred,reg_prime = tt_prime(ix)
            pred, reg = tt(ix)
            pred_outputs.append(pred*prime_pred)
            reg_output += torch.sum(reg.float()*reg_prime.float()) #numerical issue with fp 16 how fix, sum of square terms, serves as fp 16 fix

        return pred_outputs,reg_output*self.lambda_reg

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
            reg_output += torch.sum(
                reg.float() * reg_prime.float())  # numerical issue with fp 16 how fix, sum of square terms, serves as fp 16 fix

        return pred_outputs, reg_output * self.lambda_reg

class KFT_scale(torch.nn.Module):
    def __init__(self, initialization_data, lambda_reg=1e-6, cuda=None, config=None, old_setup=False): #decomposition_data = {0:{'ii':[0,1],'lambda':0.01,r_1:1 n_list=[10,10],r_2:10,'has_side_info':True, side_info:{1:x_1,2:x_2},kernel_para:{'ls_factor':0.5, 'kernel_type':'RBF','nu':2.5} },1:{}}
        super(KFT_scale, self).__init__()
        self.kernel_class_name = ['TT_kernel_component']
        self.cuda = cuda
        self.config = config
        self.amp = None
        self.old_setup = old_setup
        self.register_buffer('lambda_reg',torch.tensor(lambda_reg))
        tmp_dict = {}
        tmp_dict_s = {}
        tmp_dict_b = {}
        self.full_grad = config['full_grad']
        self.ii = {}
        for i,v in initialization_data.items():
            self.ii[i] = v['ii']
            tmp_dict_b[str(i)] = TT_component(r_1=v['r_1_latent'],n_list=v['n_list'],r_2=v['r_2_latent'],cuda=cuda,config=config,init_scale=v['init_scale'])
            tmp_dict_s[str(i)] = TT_component(r_1=v['r_1_latent'],n_list=v['n_list'],r_2=v['r_2_latent'],cuda=cuda,config=config,init_scale=v['init_scale'])
            if v['has_side_info']:
                tmp_dict[str(i)] = TT_kernel_component(r_1=v['r_1'],
                                                       n_list=v['n_list'],
                                                       r_2=v['r_2'],
                                                       side_information_dict=v['side_info'],
                                                       kernel_para_dict=v['kernel_para'],cuda=cuda,config=config,init_scale=v['init_scale'])
            else:
                tmp_dict[str(i)] = TT_component(r_1=v['r_1'],n_list=v['n_list'],r_2=v['r_2'],cuda=cuda,config=config,init_scale=v['init_scale'],old_setup=old_setup)
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
            reg_output += torch.mean(reg.float())*torch.mean(reg_s.float())+torch.mean(reg_b.float()) #numerical issue with fp 16 how fix, sum of square terms, serves as fp 16 fix
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
            return pred[torch.unbind(indices, dim=1)], reg_output * self.lambda_reg
        else:
            return pred,reg_output * self.lambda_reg

    def forward(self,indices):
        pred, reg = self.collect_core_outputs(indices)
        return pred,reg


class variational_TT_component(TT_component):
    def __init__(self,r_1,n_list,r_2,cuda=None,config=None,init_scale=1.0,old_setup=False):
        super(variational_TT_component, self).__init__(r_1,n_list,r_2,cuda,config,init_scale,old_setup)
        self.variance_parameters = torch.nn.Parameter(-init_scale*torch.ones(*self.shape_list),requires_grad=True)

    def calculate_KL(self,mean,sig):
        KL = torch.mean(0.5*(sig.exp()+mean**2-sig-1))
        return KL

    def forward_reparametrization(self,indices):
        if self.full_grad:
            mean = self.core_param
            sig = self.variance_parameters
            T = mean + sig.exp()*torch.randn_like(mean)
            return T,  self.calculate_KL(mean,sig)
        else:
            if len(indices.shape)>1:
                indices = indices.unbind(1)
            mean = self.core_param.permute(self.permutation_list)[indices]
            sig = self.variance_parameters.permute(self.permutation_list)[indices]
            z = mean + torch.randn_like(mean)*sig.exp()
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
        return mean,sig.exp()**2,KL

class univariate_variational_kernel_TT(TT_kernel_component):
    def __init__(self, r_1, n_list, r_2, side_information_dict, kernel_para_dict, cuda=None,config=None,init_scale=1.0):
        super(univariate_variational_kernel_TT, self).__init__(r_1, n_list, r_2, side_information_dict,
                                                                 kernel_para_dict, cuda, config, init_scale)
        self.variance_parameters = torch.nn.Parameter(-10.*torch.ones(*self.shape_list),requires_grad=True)

    def calculate_KL(self,mean,sig):
        KL = torch.mean(0.5*(sig.exp()+mean**2-sig-1))
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
                if not self.V_mode:
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
                    val = row_outer_prod(val,val)
                    T = lazy_mode_product(T,val.t(), key)
                    T = lazy_mode_product(T, val, key)
        return T

    def forward(self,indices):
        if self.full_grad:
            KL = self.calculate_KL(self.core_param, self.variance_parameters)
        else:
            mean = self.core_param.permute(self.permutation_list)[indices]
            sig = self.variance_parameters.permute(self.permutation_list)[indices]
            KL = self.calculate_KL(mean, sig)
        T = self.apply_kernels(self.core_param)
        T_additional = self.apply_square_kernel(self.variance_parameters.exp())
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
                if not self.V_mode:
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
    def __init__(self, r_1, n_list, r_2, side_information_dict, kernel_para_dict, cuda=None,config=None,init_scale=1.0):
        super(multivariate_variational_kernel_TT, self).__init__(r_1, n_list, r_2, side_information_dict,
                                                                 kernel_para_dict, cuda, config, init_scale)
        self.register_buffer('ones',torch.ones_like(self.core_param))
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
        self.noise_shape.append(R)
        self.register_buffer(f'prior_log_det_{key}',prior_log_det)
        self.register_buffer(f'n_const_{key}',torch.tensor(self.shape_list[key]).float())
        setattr(self,f'B_{key}',torch.nn.Parameter(torch.zeros(self.shape_list[key],R),requires_grad=True))

    def fast_log_det(self,L):
        return torch.log(torch.prod(L.diag()).abs()+1e-5)*2

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
        T = self.core_param
        log_term_1 = 0
        log_term_2 = 0
        for key in self.n_dict.keys():
            trace,D,cov,B = self.get_trace_term_KL(key)
            tr_term = tr_term*trace
            fix_det = getattr(self,f'prior_log_det_{key}')
            log_term_1 = log_term_1 + fix_det
            log_term_2 = log_term_2 + self.get_log_term(key,cov,D,B)
            if self.RFF_dict[key]:
                T = lazy_mode_product(T,B.t(),key)
                T = lazy_mode_product(T,B,key)
                T = T + D*self.core_param
            else:
                T = lazy_mode_product(T,cov,key)
        log_term = log_term_1 - log_term_2
        middle_term = torch.sum(T * self.core_param)
        return tr_term + middle_term + log_term

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
        if not self.V_mode:
            self.recalculate_priors()
        KL = self.calculate_KL()
        if self.full_grad:
            return T,KL
        else:
            if len(indices.shape) > 1:
                indices = indices.unbind(1)
            return T.permute(self.permutation_list)[indices], KL

    def forward(self,indices):
        T = self.apply_kernels(self.core_param)
        T_cross_sigma = self.apply_cross_kernel()
        T_additional = self.apply_kernels(T_cross_sigma)
        if not self.V_mode:
            self.recalculate_priors()
        KL = self.calculate_KL()
        if self.full_grad:
            return T,T_additional, KL
        else:
            if len(indices.shape) > 1:
                indices = indices.unbind(1)
            return T.permute(self.permutation_list)[indices],T_additional.permute(self.permutation_list)[indices], KL

    def apply_cross_kernel(self):
        T = self.ones
        T_D = self.ones
        for key, val in self.n_dict.items():
            if val is not None:
                D = getattr(self, f'D_{key}') ** 2
                T_D = lazy_mode_hadamard(T_D,D,key)
                if not self.V_mode:
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
                    cov,_,_ = self.build_cov(key)
                    T = lazy_mode_product(T, val*cov, key)
                else:
                    B = getattr(self, f'B_{key}')
                    val = row_outer_prod(B,val)
                    T = lazy_mode_product(T,val.t(), key)
                    T = lazy_mode_product(T, val, key)
        return T+T_D


    def mean_forward(self,indices):
        """Do tensor ops"""
        T = self.core_param
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
        if self.full_grad:
            return T
        else:
            if len(indices.shape) > 1:
                indices = indices.unbind(1)
            return T.permute(self.permutation_list)[indices]


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
            if v['has_side_info'] and v['multivariate']:
                tmp_dict[str(i)] = multivariate_variational_kernel_TT(r_1=v['r_1'],
                                                                      n_list=v['n_list'],
                                                                      r_2=v['r_2'],
                                                                      side_information_dict=v['side_info'],
                                                                      kernel_para_dict=v['kernel_para'],
                                                                      cuda=cuda,
                                                                      config=config,
                                                                      init_scale=1.0)
            else:
                tmp_dict[str(i)] = univariate_variational_kernel_TT(r_1=v['r_1'],
                                                                      n_list=v['n_list'],
                                                                      r_2=v['r_2'],
                                                                      side_information_dict=v['side_info'],
                                                                      kernel_para_dict=v['kernel_para'],
                                                                      cuda=cuda,
                                                                      config=config,
                                                                      init_scale=1.0)
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
            if v['has_side_info'] and v['multivariate']:
                tmp_dict[str(i)] = multivariate_variational_kernel_TT(r_1=v['r_1'],
                                                                      n_list=v['n_list'],
                                                                      r_2=v['r_2'],
                                                                      side_information_dict=v['side_info'],
                                                                      kernel_para_dict=v['kernel_para'],
                                                                      cuda=cuda,
                                                                      config=config)
            else:
                tmp_dict[str(i)] = univariate_variational_kernel_TT(r_1=v['r_1'],
                                                                    n_list=v['n_list'],
                                                                    r_2=v['r_2'],
                                                                    side_information_dict=v['side_info'],
                                                                    kernel_para_dict=v['kernel_para'],
                                                                    cuda=cuda,
                                                                    config=config)
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